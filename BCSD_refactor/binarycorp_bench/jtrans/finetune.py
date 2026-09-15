"""Fine-tune jTrans with InfoNCE contrastive loss on the bench training set (jtrans*).

Mirrors `binarycorp_bench/clap/finetune.py` but for jTrans:
  * Trains on the jTrans-strict bench train split (tokens are jTrans-native and
    are fed to the tokenizer DIRECTLY — no regex normalization, no CLAP dict
    formatting).
  * No instruction prefix on the query side (jTrans has no instruction tuning);
    queries and targets are encoded symmetrically.
  * CLS pooling; the position-embedding table is bumped to 2902 to match the
    jTrans-finetune checkpoint.

Pairs are round-robin within each function id (all variants are mutual positives).
Saves a best-val-loss checkpoint and a final checkpoint so eval.py can be pointed
at the result with `--model`.

Example:
    python finetune.py
    python eval.py --model ./jtrans_finetuned \
        --embeddings_path embeddings_jtrans_bench_finetuned.pt \
        --report_path eval_jtrans_bench_finetuned_report.json
"""

import argparse
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, BertModel

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../../")))

from shared.data_utils import set_seed, load_jsonl, parse_bench_opt

DEFAULT_MODEL = "/home/ra72yeq/projects/Embedding_Paper/jTrans/models/jTrans-finetune"
DEFAULT_OUTPUT_DIR = os.path.join(script_dir, "jtrans_finetuned")

BATCH_SIZE = 32
LR = 1e-5
EPOCHS = 1
TEMPERATURE = 0.05
MAX_LENGTH = 512
SEED = 42


def resolve_train_path():
    candidates = [
        "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_train_jtrans_strict.jsonl",
        os.path.abspath(os.path.join(script_dir, "../../../nvemb/output_benchset_rebalanced_train_jtrans_strict.jsonl")),
        os.path.abspath(os.path.join(script_dir, "../../../output_benchset_rebalanced_train_jtrans_strict.jsonl")),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _asm_to_text(asm) -> str:
    return "\n".join(asm) if isinstance(asm, list) else asm


def build_pairs_by_split(samples, val_frac, seed) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Round-robin (query, positive) pairs within each function id; val held out by id."""
    import random

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        parse_bench_opt(sample["opt"])  # validate label schema
        grouped[sample["id"]].append(sample)

    fids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(fids)

    num_val = int(len(fids) * val_frac)
    val_fids = set(fids[:num_val])

    def make_pairs(selected_fids):
        pairs = []
        skipped = 0
        for fid in selected_fids:
            variants = grouped[fid]
            if len(variants) < 2:
                skipped += 1
                continue
            ordered = variants[:]
            rng.shuffle(ordered)
            for idx, query in enumerate(ordered):
                positive = ordered[(idx + 1) % len(ordered)]
                pairs.append((_asm_to_text(query["asm"]), _asm_to_text(positive["asm"])))
        return pairs, skipped

    train_fids = [f for f in fids if f not in val_fids]
    val_fids_list = [f for f in fids if f in val_fids]

    train_pairs, train_skipped = make_pairs(train_fids)
    val_pairs, val_skipped = make_pairs(val_fids_list)

    print(
        f"Functions: {len(fids)} total -> {len(train_fids)} train / {len(val_fids_list)} val. "
        f"Pairs: {len(train_pairs)} train ({train_skipped} fns skipped <2 variants), "
        f"{len(val_pairs)} val ({val_skipped} fns skipped <2 variants)."
    )
    return train_pairs, val_pairs


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_pairs(batch):
    queries = [q for q, _ in batch]
    targets = [t for _, t in batch]
    return queries, targets


# ---------------------------------------------------------------------------
# jTrans model + tokenizer (port of supervisor's BCSD/models.py approach)
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["[SEP]", "[PAD]", "[CLS]", "[MASK]"]
UNKNOWN_TOKEN_ID = 512


class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.config = config
        self.embeddings.position_embeddings = self.embeddings.word_embeddings


def _resolve_tokenizer_dir(model_path, tokenizer_path=None):
    if tokenizer_path is not None:
        candidate = Path(tokenizer_path)
        if (candidate / "vocab.txt").exists():
            return candidate
        raise FileNotFoundError(f"JTrans tokenizer vocab not found under {candidate}")
    model_dir = Path(model_path)
    candidates = [
        model_dir,
        model_dir.parent / "jtrans_tokenizer",
        Path(__file__).resolve().parent.parent.parent.parent / "jtrans_tokenizer",
    ]
    for candidate in candidates:
        if (candidate / "vocab.txt").exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate jTrans tokenizer vocab.txt for model path {model_path}")


def _load_vocab(tokenizer_dir):
    vocab_path = Path(tokenizer_dir) / "vocab.txt"
    vocab_data = vocab_path.read_text(encoding="utf-8").strip().split("\n")
    vocab_data.extend(SPECIAL_TOKENS)
    return defaultdict(
        lambda: UNKNOWN_TOKEN_ID,
        {token: idx for idx, token in enumerate(vocab_data)},
    )


def _jtrans_tokenize(vocab, text, max_length):
    split_line = text.strip().split(" ") if text.strip() else []
    if len(split_line) <= max_length - 3:
        split_line = ["[CLS]"] + split_line + ["[SEP]"]
        attention_mask = [1] * len(split_line) + [0] * (max_length - len(split_line))
        split_line = split_line + (max_length - len(split_line)) * ["[PAD]"]
    else:
        split_line = ["[CLS]"] + split_line[: max_length - 2] + ["[SEP]"]
        attention_mask = [1] * max_length
    input_ids = [vocab[token] for token in split_line]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def load_jtrans(model_path, device, tokenizer_path=None):
    tokenizer_dir = _resolve_tokenizer_dir(model_path, tokenizer_path)
    vocab = _load_vocab(tokenizer_dir)
    config = AutoConfig.from_pretrained(model_path)
    config.max_position_embeddings = len(vocab)
    model = BinBertModel.from_pretrained(model_path, config=config).to(device)
    return model, vocab, tokenizer_dir


def save_vocab(tokenizer_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    src = Path(tokenizer_dir) / "vocab.txt"
    dst = Path(output_dir) / "vocab.txt"
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode(model, vocab, texts, max_length, device, use_amp):
    input_ids_list, attention_masks_list = [], []
    for text in texts:
        t = _jtrans_tokenize(vocab, text, max_length)
        input_ids_list.append(t["input_ids"])
        attention_masks_list.append(t["attention_mask"])
    inputs = {
        "input_ids": torch.stack(input_ids_list).to(device),
        "attention_mask": torch.stack(attention_masks_list).to(device),
    }
    with torch.autocast(device_type="cuda", enabled=use_amp):
        out = model(**inputs)
        embeddings = out.pooler_output
    return F.normalize(embeddings.float(), p=2, dim=1)


def info_nce_loss(embeddings, temperature):
    """InfoNCE over interleaved [q0, t0, q1, t1, ...]; positives are adjacent."""
    n = embeddings.size(0)
    sim = (embeddings @ embeddings.t()) / temperature
    diag = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    sim = sim.masked_fill(diag, float("-inf"))
    idx = torch.arange(n, device=embeddings.device)
    labels = idx + 1 - 2 * (idx % 2)  # 0<->1, 2<->3, ...
    return F.cross_entropy(sim, labels)


def build_interleaved_embeddings(model, vocab, queries, targets, max_length, device, use_amp):
    interleaved = []
    for q_text, t_text in zip(queries, targets):
        interleaved.append(q_text)
        interleaved.append(t_text)
    return encode(model, vocab, interleaved, max_length, device, use_amp)


@torch.no_grad()
def evaluate(model, vocab, loader, max_length, device, use_amp, temperature, max_batches=None) -> float:
    model.eval()
    total_loss = 0.0
    total_pairs = 0
    for i, (queries, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        embeddings = build_interleaved_embeddings(model, vocab, queries, targets, max_length, device, use_amp)
        loss = info_nce_loss(embeddings, temperature)
        total_loss += loss.item() * len(queries)
        total_pairs += len(queries)
    model.train()
    return total_loss / max(1, total_pairs)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune jTrans with InfoNCE on the bench train split.")
    parser.add_argument("--data", default=None, help="jTrans-strict bench train JSONL path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="jTrans model dir to start from.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Final checkpoint dir.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--val_batches", type=int, default=64, help="Cap val to N batches (0 = full).")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.startswith("cuda")
    best_dir = args.output_dir.rstrip("/") + "_best"
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Final checkpoint -> {args.output_dir}")
    print(f"Best-val checkpoint -> {best_dir}")

    data_path = args.data or resolve_train_path()
    samples = load_jsonl(data_path)
    train_pairs, val_pairs = build_pairs_by_split(samples, args.val_frac, args.seed)
    if not train_pairs:
        raise RuntimeError("No training pairs were built; check dataset path and id/opt schema.")

    print(f"Loading jTrans from {args.model}...")
    model, vocab, tokenizer_dir = load_jtrans(args.model, device)
    model.train()

    train_loader = DataLoader(
        PairDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        PairDataset(val_pairs),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pairs,
        num_workers=args.num_workers,
    ) if val_pairs else None

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    val_cap = args.val_batches if args.val_batches and args.val_batches > 0 else None

    best_val = float("inf")
    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        running = []
        for queries, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            embeddings = build_interleaved_embeddings(model, vocab, queries, targets, args.max_length, device, use_amp)
            loss = info_nce_loss(embeddings, args.temperature)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            running.append(loss.item())

            if global_step % 100 == 0:
                avg = sum(running[-100:]) / min(100, len(running))
                elapsed = (time.time() - start_time) / 60
                print(
                    f"epoch {epoch + 1} step {global_step} | loss {loss.item():.4f} "
                    f"| avg100 {avg:.4f} | {elapsed:.1f} min",
                    flush=True,
                )

            if val_loader is not None and args.val_every > 0 and global_step % args.val_every == 0:
                val_loss = evaluate(model, vocab, val_loader, args.max_length, device, use_amp, args.temperature, max_batches=val_cap)
                print(f"[val] step {global_step} | val_loss {val_loss:.4f} (best {best_val:.4f})", flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    os.makedirs(best_dir, exist_ok=True)
                    model.save_pretrained(best_dir, safe_serialization=False)
                    save_vocab(tokenizer_dir, best_dir)
                    print(f"[val] new best ({best_val:.4f}) saved to {best_dir}", flush=True)

        if running:
            print(f"epoch {epoch + 1} done | avg loss {sum(running) / len(running):.4f}", flush=True)

    if val_loader is not None:
        val_loss = evaluate(model, vocab, val_loader, args.max_length, device, use_amp, args.temperature, max_batches=val_cap)
        print(f"[val] final | val_loss {val_loss:.4f} (best {best_val:.4f})", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir, safe_serialization=False)
            save_vocab(tokenizer_dir, best_dir)
            print(f"[val] new best ({best_val:.4f}) saved to {best_dir}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=False)
    save_vocab(tokenizer_dir, args.output_dir)
    print(f"Final model saved to {args.output_dir}")
    if best_val != float("inf"):
        print(f"Best val loss: {best_val:.4f} (checkpoint: {best_dir})")


if __name__ == "__main__":
    main()
