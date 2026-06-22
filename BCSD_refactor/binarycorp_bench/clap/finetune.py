"""Fine-tune hustcw/clap-asm with InfoNCE contrastive loss on the bench training set.

Pairs are built round-robin within each function id (all opt/arch/compiler variants of
the same function are mutual positives). Trains for 1 epoch and saves both the
best-val-loss checkpoint and the final checkpoint so the existing eval.py can point
--model at the result.

Input format (must match eval.py exactly):
  * Trains on the **CLAP-strict** bench train split
    (`output_benchset_rebalanced_train_clap_strict.jsonl`), whose `asm` is a list of
    IDA-style instructions — the same representation eval.py consumes. (The older
    version trained on the Nova-format file, which is a different tokenization and
    caused a train/eval mismatch.)
  * Symmetric encoding with **no instruction prefix** — CLAP's asm tokenizer has no
    notion of natural-language instructions, and eval.py adds no prefix either.

Example:
    python finetune.py
    python eval.py --model ./clap_finetuned \
        --embeddings_path embeddings_clap_bench_finetuned.pt \
        --report_path eval_clap_bench_finetuned_report.json
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# Reduce CUDA fragmentation OOMs on shared GPUs (must be set before CUDA init).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# shared.data_utils lives under BCSD_refactor (two levels up from clap/).
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../../")))

from shared.data_utils import set_seed, load_jsonl, parse_bench_opt

MODEL_ID = "hustcw/clap-asm"

DEFAULT_OUTPUT_DIR = os.path.join(script_dir, "clap_finetuned")


def resolve_train_path():
    """Locate the CLAP-strict bench train split (matches the format eval.py tests on)."""
    candidates = [
        "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_train_clap_strict.jsonl",
        os.path.abspath(os.path.join(script_dir, "../../../nvemb/output_benchset_rebalanced_train_clap_strict.jsonl")),
        os.path.abspath(os.path.join(script_dir, "../../../output_benchset_rebalanced_train_clap_strict.jsonl")),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]

# Hyperparameters (per task spec).
BATCH_SIZE = 32
LR = 1e-5
EPOCHS = 1
TEMPERATURE = 0.05
MAX_LENGTH = 1024
SEED = 42


def build_pairs_by_split(
    samples: List[Dict[str, Any]],
    val_frac: float,
    seed: int,
) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
    """Round-robin (query, positive) pairs within each function id.

    Each element of a pair is the raw `asm` field (a list of instructions for the
    CLAP-strict format). The val split is held out by *function id* (not by pair)
    so that no function leaks between train and val.
    """
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

    def make_pairs(selected_fids: List[str]) -> List[Tuple[str, str]]:
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
                pairs.append((query["asm"], positive["asm"]))
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
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_pairs(batch: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    queries = [q for q, _ in batch]
    targets = [t for _, t in batch]
    return queries, targets


def format_asm(text: str, max_length: int) -> Dict[str, str]:
    asm_list = text.split("\n") if isinstance(text, str) else text
    asm_list = asm_list[:max_length]
    return {str(i): inst for i, inst in enumerate(asm_list)}


def prepare_input(
    tokenizer,
    texts: List[str],
    max_length: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Tokenize and pad a list of asm texts to max_length (CLAP input format)."""
    formatted_batch = [format_asm(t, max_length) for t in texts]
    raw_inputs = tokenizer(formatted_batch)
    pad_id = tokenizer.pad_token_id or 0
    final_inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

    for i in range(len(texts)):
        input_ids = raw_inputs["input_ids"][i][:max_length]
        attention_mask = raw_inputs["attention_mask"][i][:max_length]
        token_type_ids = raw_inputs["token_type_ids"][i][:max_length]
        pad_len = max_length - len(input_ids)

        final_inputs["input_ids"].append(input_ids + [pad_id] * pad_len)
        final_inputs["attention_mask"].append(attention_mask + [0] * pad_len)
        final_inputs["token_type_ids"].append(token_type_ids + [0] * pad_len)

    return {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in final_inputs.items()}


def encode(model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(**inputs)
    embeddings = outputs[:, 0, :] if outputs.ndim == 3 else outputs
    return F.normalize(embeddings.float(), p=2, dim=1)


def info_nce_loss(embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
    """InfoNCE over interleaved [q0, t0, q1, t1, ...] embeddings.

    Positives are adjacent (labels = [1, 0, 3, 2, ...]); the self-similarity diagonal
    is masked out before cross entropy.
    """
    n = embeddings.size(0)
    sim = (embeddings @ embeddings.t()) / temperature
    diag = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    sim = sim.masked_fill(diag, float("-inf"))
    idx = torch.arange(n, device=embeddings.device)
    labels = idx + 1 - 2 * (idx % 2)  # 0<->1, 2<->3, ...
    return F.cross_entropy(sim, labels)


def build_interleaved_embeddings(model, tokenizer, queries, targets, max_length, device, use_amp):
    """Encode queries + targets interleaved q0,t0,q1,t1,...

    Symmetric encoding with NO instruction prefix, matching binarycorp_bench/clap/eval.py.
    Each query/target is a raw asm value (list or string); prepare_input/format_asm
    handle both representations identically to eval.
    """
    interleaved: List[Any] = []
    for q_asm, t_asm in zip(queries, targets):
        interleaved.append(q_asm)
        interleaved.append(t_asm)

    inputs = prepare_input(tokenizer, interleaved, max_length, device)
    with torch.autocast(device_type="cuda", enabled=use_amp):
        embeddings = encode(model, inputs)
    return embeddings


@torch.no_grad()
def evaluate(model, tokenizer, loader, max_length, device, use_amp, temperature, max_batches=None) -> float:
    """Average InfoNCE loss over the val loader.

    `max_batches` caps the number of batches evaluated. The loader uses shuffle=False,
    so a cap yields the same deterministic subset every call -> comparable best-ckpt
    selection without paying for the full (40k+ pair) val set each time.
    """
    model.eval()
    total_loss = 0.0
    total_pairs = 0
    for i, (queries, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        embeddings = build_interleaved_embeddings(
            model, tokenizer, queries, targets, max_length, device, use_amp
        )
        loss = info_nce_loss(embeddings, temperature)
        total_loss += loss.item() * len(queries)
        total_pairs += len(queries)
    model.train()
    return total_loss / max(1, total_pairs)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLAP-asm with InfoNCE on the bench train split.")
    parser.add_argument("--data", default=None, help="Bench train JSONL path (default: CLAP-strict train split).")
    parser.add_argument("--model", default=MODEL_ID, help="CLAP model id or local path to start from.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Final checkpoint dir.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val_frac", type=float, default=0.10, help="Fraction of function ids held out for val.")
    parser.add_argument("--val_every", type=int, default=1000, help="Run validation every N optimizer steps.")
    parser.add_argument(
        "--val_batches",
        type=int,
        default=64,
        help="Cap periodic/final validation to this many batches (deterministic subset). 0 = full val set.",
    )
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

    print(f"Loading CLAP from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(device)
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
            embeddings = build_interleaved_embeddings(
                model, tokenizer, queries, targets, args.max_length, device, use_amp
            )
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
                val_loss = evaluate(
                    model, tokenizer, val_loader, args.max_length, device, use_amp,
                    args.temperature, max_batches=val_cap,
                )
                print(f"[val] step {global_step} | val_loss {val_loss:.4f} (best {best_val:.4f})", flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    os.makedirs(best_dir, exist_ok=True)
                    # CLAP ties word/token_type embeddings -> safetensors refuses shared tensors.
                    model.save_pretrained(best_dir, safe_serialization=False)
                    tokenizer.save_pretrained(best_dir)
                    print(f"[val] new best ({best_val:.4f}) saved to {best_dir}", flush=True)

        if running:
            print(f"epoch {epoch + 1} done | avg loss {sum(running) / len(running):.4f}", flush=True)

    # Final validation pass (same deterministic subset so it's comparable to periodic vals).
    if val_loader is not None:
        val_loss = evaluate(
            model, tokenizer, val_loader, args.max_length, device, use_amp,
            args.temperature, max_batches=val_cap,
        )
        print(f"[val] final | val_loss {val_loss:.4f} (best {best_val:.4f})", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir, safe_serialization=False)
            tokenizer.save_pretrained(best_dir)
            print(f"[val] new best ({best_val:.4f}) saved to {best_dir}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Final model saved to {args.output_dir}")
    if best_val != float("inf"):
        print(f"Best val loss: {best_val:.4f} (checkpoint: {best_dir})")


if __name__ == "__main__":
    main()
