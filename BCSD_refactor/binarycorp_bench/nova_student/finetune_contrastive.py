import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.collators import DistillCollator
from shared.data_utils import load_jsonl, parse_bench_opt, set_seed
from shared.losses import contrastive_loss_positive_aware
from shared.nova_utils import setup_nova_tokenizer
from shared.student_model import LatentAttentionLayer, StudentDistillationModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
BATCH_SIZE = 32
GRAD_ACCUM = 8
LR = 1e-4
NUM_EPOCHS = 10
MAX_LENGTH = 1024
VAL_SPLIT = 0.1
EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_DELTA = 1e-4

script_dir_file = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir_file, "../../../"))
DATA_PATH = os.path.join(repo_root, "nvemb/output_benchset_rebalanced_train_nova.jsonl")
DEFAULT_SOURCE_DIR = "/home/ra72yeq/BCSD/model_checkpoints/nova_student_bench/student_best"
DEFAULT_OUTPUT_DIR = "/home/ra72yeq/BCSD/model_checkpoints/nova_student_bench_contrastive_ft"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue a Nova student checkpoint with contrastive-only training."
    )
    parser.add_argument("--source_dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    return parser.parse_args()


def log_factory(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"log_contrastive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def log(msg: str, print_to_stdout: bool = True) -> None:
        if print_to_stdout:
            print(msg)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    return log


def group_bench_samples(samples):
    grouped = defaultdict(list)
    for sample in samples:
        parse_bench_opt(sample["opt"])
        grouped[sample["id"]].append(sample)
    return grouped


def build_bench_pairs(grouped, func_ids, func_id_to_int, seed=42):
    rng = random.Random(seed)
    pairs = []
    skipped = 0

    for fid in func_ids:
        variants = grouped[fid]
        if len(variants) < 2:
            skipped += 1
            continue

        shuffled = variants[:]
        rng.shuffle(shuffled)
        func_int = func_id_to_int[fid]

        for idx, query in enumerate(shuffled):
            positive = shuffled[(idx + 1) % len(shuffled)]
            pairs.append((query["asm"], positive["asm"], func_int))

    rng.shuffle(pairs)
    return pairs, skipped


def resolve_ckpt(source_dir, name):
    best = os.path.join(source_dir, name.replace(".pt", "_best.pt"))
    plain = os.path.join(source_dir, name)
    if os.path.exists(best):
        return best
    if os.path.exists(plain):
        return plain
    raise FileNotFoundError(f"Missing checkpoint for {name} in {source_dir}")


@torch.no_grad()
def evaluate_contrastive(student_model, lal_head, data_loader, device):
    student_model.eval()
    lal_head.eval()
    total = 0.0
    steps = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        func_ids = batch["func_ids"].to(device)
        key_padding_mask = batch["key_padding_mask"].to(device)
        pool_mask = batch["pool_mask"].to(device)

        h_student = student_model(input_ids, key_padding_mask=key_padding_mask)
        embeddings = lal_head(h_student, key_padding_mask=key_padding_mask, pool_mask=pool_mask)
        loss = contrastive_loss_positive_aware(embeddings, func_ids)

        total += loss.item()
        steps += 1

        del input_ids, func_ids, key_padding_mask, pool_mask, h_student, embeddings, loss

    student_model.train()
    lal_head.train()
    return total / max(1, steps)


def save_checkpoint(student_model, lal_head, cfg, out_dir, suffix):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(out_dir, f"student_model{suffix}.pt"))
    torch.save(lal_head.state_dict(), os.path.join(out_dir, f"lal_head{suffix}.pt"))
    with open(os.path.join(out_dir, "student_config.json"), "w") as f:
        json.dump(cfg, f)


def main():
    args = parse_args()
    log = log_factory(args.output_dir)
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")
    log(f"Source checkpoint: {args.source_dir}")
    log(f"Output directory:  {args.output_dir}")
    log("Training objective: contrastive only (no teacher forward, no MSE distillation).")

    log("Loading tokenizer...")
    base_tokenizer, nova_tokenizer, _ = setup_nova_tokenizer()
    pad_id = base_tokenizer.pad_token_id or 0

    log("Loading data...")
    train_samples = load_jsonl(DATA_PATH)
    grouped_samples = group_bench_samples(train_samples)
    valid_func_ids = [fid for fid, variants in grouped_samples.items() if len(variants) >= 2]
    random.shuffle(valid_func_ids)

    val_func_count = max(1, int(len(valid_func_ids) * VAL_SPLIT)) if len(valid_func_ids) > 1 else 0
    val_func_ids = valid_func_ids[:val_func_count]
    train_func_ids = valid_func_ids[val_func_count:]
    func_id_to_int = {fid: i for i, fid in enumerate(sorted(valid_func_ids))}

    train_pairs, train_skipped = build_bench_pairs(grouped_samples, train_func_ids, func_id_to_int, seed=SEED)
    val_pairs, val_skipped = build_bench_pairs(grouped_samples, val_func_ids, func_id_to_int, seed=SEED + 1) if val_func_ids else ([], 0)
    if not train_pairs:
        raise RuntimeError("No bench train pairs were built; check dataset path and opt/id schema.")

    log(
        f"Train functions: {len(train_func_ids)} | Val functions: {len(val_func_ids)} | "
        f"Train pairs: {len(train_pairs)} ({train_skipped} skipped) | "
        f"Val pairs: {len(val_pairs)} ({val_skipped} skipped)"
    )

    with open(os.path.join(args.source_dir, "student_config.json"), "r") as f:
        cfg = json.load(f)

    log("Loading student and LAL head from checkpoint...")
    student_model = StudentDistillationModule(
        vocab_size=cfg["vocab_size"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        pad_id=pad_id,
    ).to(device).to(torch.bfloat16)
    lal_head = LatentAttentionLayer(hidden_dim=cfg["hidden_dim"]).to(device).to(torch.bfloat16)

    student_model.load_state_dict(torch.load(resolve_ckpt(args.source_dir, "student_model.pt"), map_location="cpu"))
    lal_head.load_state_dict(torch.load(resolve_ckpt(args.source_dir, "lal_head.pt"), map_location="cpu"))

    pair_collator = DistillCollator(nova_tokenizer, max_length=args.max_length)
    pair_loader = DataLoader(
        train_pairs,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pair_collator,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_pairs,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pair_collator,
        num_workers=4,
        pin_memory=True,
    ) if val_pairs else None

    total_steps = math.ceil(len(pair_loader) / args.grad_accum) * args.epochs
    optimizer = AdamW(list(student_model.parameters()) + list(lal_head.parameters()), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)

    log(f"Starting contrastive fine-tune for {args.epochs} epochs / {total_steps} optimizer steps.")
    student_model.train()
    lal_head.train()
    optimizer.zero_grad()

    global_step = 0
    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch in range(args.epochs):
        running_loss = 0.0
        running_count = 0

        for step, batch in enumerate(tqdm(pair_loader, desc=f"Epoch {epoch + 1}")):
            input_ids = batch["input_ids"].to(device)
            func_ids = batch["func_ids"].to(device)
            key_padding_mask = batch["key_padding_mask"].to(device)
            pool_mask = batch["pool_mask"].to(device)

            h_student = student_model(input_ids, key_padding_mask=key_padding_mask)
            embeddings = lal_head(h_student, key_padding_mask=key_padding_mask, pool_mask=pool_mask)
            loss = contrastive_loss_positive_aware(embeddings, func_ids)

            running_loss += loss.item()
            running_count += 1
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(pair_loader):
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(lal_head.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    log(
                        f" Step {global_step}: "
                        f"Contrastive={running_loss / max(1, running_count):.4f}",
                        print_to_stdout=False,
                    )
                    running_loss = 0.0
                    running_count = 0

            del input_ids, func_ids, key_padding_mask, pool_mask, h_student, embeddings, loss

        if val_loader is not None:
            val_loss = evaluate_contrastive(student_model, lal_head, val_loader, device)
            log(f"Epoch {epoch + 1} Val Contrastive Loss: {val_loss:.4f}")

            if val_loss + EARLY_STOP_MIN_DELTA < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                save_checkpoint(
                    student_model,
                    lal_head,
                    cfg,
                    os.path.join(args.output_dir, "student_best"),
                    "_best",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    log("Early stopping triggered.")
                    break

    save_checkpoint(student_model, lal_head, cfg, os.path.join(args.output_dir, "student_final"), "")
    log("Contrastive fine-tune complete.")


if __name__ == "__main__":
    main()
