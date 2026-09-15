#!/usr/bin/env python3
"""Train pooling-only BCSD heads over a frozen native-causal Nova backbone.

The mean baseline has no trainable parameters.  This script jointly trains two
independent heads from the same frozen Nova forward pass:

1. mean pooling -> identity-initialized linear projection;
2. scalar attention pooling over the same assembly-token positions.

Nova receives its official tokenizer-produced hierarchical causal mask without
symmetrization.  No Nova weight, adapter, or LM head is trained.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from binarycorp_bench.nova_causal_pooling.core import (  # noqa: E402
    CausalNovaPairCollator,
    FrozenCausalNova,
    LinearMeanProbe,
    MaskedAttentionPooling,
    build_bench_pairs,
    masked_mean_pool,
)
from shared.data_utils import group_samples_by_id, load_jsonl, set_seed  # noqa: E402
from shared.losses import contrastive_loss_positive_aware  # noqa: E402
from shared.nova_utils import MODEL_ID, NOVA_CACHE_DIR, load_nova  # noqa: E402


@dataclass
class RunConfig:
    data: str
    output_dir: str
    seed: int
    epochs: int
    pair_batch_size: int
    grad_accum: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    temperature: float
    max_length: int
    val_fraction: float
    num_workers: int
    device_id: int
    max_train_pairs: Optional[int]
    max_val_pairs: Optional[int]
    train_heads: tuple
    resume_linear: Optional[str]
    resume_attention: Optional[str]
    backbone: str = MODEL_ID
    backbone_snapshot: str = Path(NOVA_CACHE_DIR).name
    backbone_trainable: bool = False
    adapter: Optional[str] = None
    mask: str = "official_native_hierarchical_causal"
    mask_symmetrized: bool = False
    query_instruction: bool = True
    pooling_positions: str = "assembly_tokens_only"


def parse_args() -> argparse.Namespace:
    default_data = BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_train_nova.jsonl"
    default_root = REPO_ROOT / "model_checkpoints" / "nova_causal_pooling"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(default_data))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=8,
        help="Positive pairs per forward; Nova sees twice this many sequences.",
    )
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument(
        "--train-heads",
        nargs="+",
        choices=("linear", "attention"),
        default=("linear", "attention"),
        help="Which trainable pooling heads to update. Use '--train-heads linear' to continue only the linear probe.",
    )
    parser.add_argument(
        "--resume-linear",
        default=None,
        help="Optional linear_head_*.pt checkpoint to initialize the linear probe before training.",
    )
    parser.add_argument(
        "--resume-attention",
        default=None,
        help="Optional attention_head_*.pt checkpoint to initialize the attention pooling head before training.",
    )
    return parser.parse_args()


def log(message: str, log_path: Path) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def load_official_nova(device_id: int):
    """Load the unmodified HF tokenizer/model without adding repository tokens."""
    if NOVA_CACHE_DIR not in sys.path:
        sys.path.insert(0, NOVA_CACHE_DIR)
    from modeling_nova import NovaTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=NOVA_CACHE_DIR,
        local_files_only=True,
    )
    nova_tokenizer = NovaTokenizer(base_tokenizer)
    causal_lm = load_nova(device_id=device_id, base_tokenizer=None)
    return base_tokenizer, nova_tokenizer, causal_lm


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def validate(
    backbone: FrozenCausalNova,
    linear_head: LinearMeanProbe,
    attention_head: MaskedAttentionPooling,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> Dict[str, float]:
    linear_head.eval()
    attention_head.eval()
    totals = {"mean": 0.0, "linear": 0.0, "attention": 0.0}
    steps = 0
    for cpu_batch in tqdm(loader, desc="validation", leave=False):
        batch = move_batch(cpu_batch, device)
        hidden = backbone(
            batch["input_ids"],
            batch["attention_mask"],
            batch["nova_attention_mask"],
        )
        mean_embedding = masked_mean_pool(hidden, batch["pool_mask"])
        embeddings = {
            "mean": mean_embedding,
            "linear": linear_head(mean_embedding),
            "attention": attention_head(hidden, batch["pool_mask"]),
        }
        for name, embedding in embeddings.items():
            totals[name] += contrastive_loss_positive_aware(
                embedding.float(), batch["func_ids"], temperature=temperature
            ).item()
        steps += 1
    linear_head.train()
    attention_head.train()
    return {name: total / max(steps, 1) for name, total in totals.items()}


def save_head(head: torch.nn.Module, path: Path, metadata: Dict) -> None:
    torch.save(
        {"state_dict": head.state_dict(), "metadata": metadata},
        path,
    )


def load_head_checkpoint(head: torch.nn.Module, checkpoint: str, expected: str) -> Path:
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing {expected} checkpoint: {checkpoint_path}")
    if expected not in checkpoint_path.name:
        raise ValueError(
            f"Expected a {expected} checkpoint, got: {checkpoint_path.name}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu")
    head.load_state_dict(payload["state_dict"])
    return checkpoint_path


def selected_trainables(
    linear_head: LinearMeanProbe,
    attention_head: MaskedAttentionPooling,
    train_heads: Tuple[str, ...],
) -> list:
    params = []
    for parameter in linear_head.parameters():
        parameter.requires_grad = "linear" in train_heads
    for parameter in attention_head.parameters():
        parameter.requires_grad = "attention" in train_heads
    if "linear" in train_heads:
        params.extend(linear_head.parameters())
    if "attention" in train_heads:
        params.extend(attention_head.parameters())
    if not params:
        raise ValueError("At least one trainable head must be selected")
    return params


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This 1.3B Nova experiment requires a CUDA device")
    if args.epochs < 1 or args.pair_batch_size < 1 or args.grad_accum < 1:
        raise ValueError("epochs, pair-batch-size, and grad-accum must be positive")
    train_heads = tuple(args.train_heads)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or (REPO_ROOT / "model_checkpoints" / "nova_causal_pooling" / run_id))
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "train.log"

    config = RunConfig(
        data=str(Path(args.data).resolve()),
        output_dir=str(output_dir.resolve()),
        seed=args.seed,
        epochs=args.epochs,
        pair_batch_size=args.pair_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        temperature=args.temperature,
        max_length=args.max_length,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        device_id=args.device_id,
        max_train_pairs=args.max_train_pairs,
        max_val_pairs=args.max_val_pairs,
        train_heads=train_heads,
        resume_linear=str(Path(args.resume_linear).resolve()) if args.resume_linear else None,
        resume_attention=str(Path(args.resume_attention).resolve()) if args.resume_attention else None,
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}")
    log("Loading bench data and creating a function-disjoint validation split...", log_path)
    samples = load_jsonl(args.data)
    grouped = group_samples_by_id(samples)
    function_ids = sorted(fid for fid, rows in grouped.items() if len(rows) >= 2)
    random.Random(args.seed).shuffle(function_ids)
    val_count = max(1, int(len(function_ids) * args.val_fraction))
    val_ids = function_ids[:val_count]
    train_ids = function_ids[val_count:]
    train_pairs = build_bench_pairs(samples, train_ids, seed=args.seed)
    val_pairs = build_bench_pairs(samples, val_ids, seed=args.seed + 1)
    if args.max_train_pairs is not None:
        train_pairs = train_pairs[: args.max_train_pairs]
    if args.max_val_pairs is not None:
        val_pairs = val_pairs[: args.max_val_pairs]
    log(
        f"Functions train/val: {len(train_ids):,}/{len(val_ids):,}; "
        f"pairs train/val: {len(train_pairs):,}/{len(val_pairs):,}",
        log_path,
    )

    log("Loading official Nova-1.3B weights and tokenizer (no added tokens)...", log_path)
    _, nova_tokenizer, causal_lm = load_official_nova(args.device_id)
    backbone = FrozenCausalNova(causal_lm)
    collator = CausalNovaPairCollator(nova_tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_pairs,
        batch_size=args.pair_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_pairs,
        batch_size=args.pair_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    hidden_dim = backbone.hidden_dim
    linear_head = LinearMeanProbe(hidden_dim).to(device=device, dtype=torch.bfloat16)
    attention_head = MaskedAttentionPooling(hidden_dim).to(device=device, dtype=torch.bfloat16)
    resumed = {}
    if args.resume_linear:
        resumed["linear"] = load_head_checkpoint(
            linear_head, args.resume_linear, "linear_head"
        )
    if args.resume_attention:
        resumed["attention"] = load_head_checkpoint(
            attention_head, args.resume_attention, "attention_head"
        )
    for name, checkpoint_path in resumed.items():
        log(f"Initialized {name} head from: {checkpoint_path}", log_path)

    trainable = selected_trainables(linear_head, attention_head, train_heads)
    optimizer = AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-7,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_updates),
        num_training_steps=total_updates,
    )
    log(
        f"Frozen Nova params: {sum(p.numel() for p in causal_lm.parameters()):,}; "
        f"trainable linear: {sum(p.numel() for p in linear_head.parameters()):,}; "
        f"trainable attention: {sum(p.numel() for p in attention_head.parameters()):,}",
        log_path,
    )
    log(f"Updating heads: {', '.join(train_heads)}", log_path)
    log(
        "Mask contract: official tokenizer mask, strict lower triangular, "
        "no symmetrization; pooling contract: assembly tokens only.",
        log_path,
    )

    history = []
    best = {"linear": float("inf"), "attention": float("inf")}
    optimizer.zero_grad(set_to_none=True)
    global_update = 0

    for epoch in range(args.epochs):
        linear_head.train("linear" in train_heads)
        attention_head.train("attention" in train_heads)
        running = {"linear": 0.0, "attention": 0.0}
        running_steps = 0

        for step, cpu_batch in enumerate(
            tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"), start=1
        ):
            batch = move_batch(cpu_batch, device)
            hidden = backbone(
                batch["input_ids"],
                batch["attention_mask"],
                batch["nova_attention_mask"],
            )
            mean_embedding = masked_mean_pool(hidden, batch["pool_mask"])
            losses = {}
            linear_embedding = attention_embedding = None
            if "linear" in train_heads:
                linear_embedding = linear_head(mean_embedding)
                losses["linear"] = contrastive_loss_positive_aware(
                    linear_embedding.float(), batch["func_ids"], temperature=args.temperature
                )
            if "attention" in train_heads:
                attention_embedding = attention_head(hidden, batch["pool_mask"])
                losses["attention"] = contrastive_loss_positive_aware(
                    attention_embedding.float(), batch["func_ids"], temperature=args.temperature
                )

            loss = sum(losses.values()) / args.grad_accum
            loss.backward()
            for name, head_loss in losses.items():
                running[name] += head_loss.item()
            running_steps += 1

            should_update = step % args.grad_accum == 0 or step == len(train_loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if global_update % 20 == 0:
                    metrics = []
                    for name in train_heads:
                        metrics.append(f"{name}={running[name] / running_steps:.4f}")
                    log(
                        f"update={global_update}/{total_updates} "
                        + " ".join(metrics),
                        log_path,
                    )
                    running = {"linear": 0.0, "attention": 0.0}
                    running_steps = 0

            del hidden, mean_embedding, linear_embedding, attention_embedding
            del losses, loss, batch

        validation = validate(
            backbone,
            linear_head,
            attention_head,
            val_loader,
            device,
            args.temperature,
        )
        record = {"epoch": epoch + 1, "global_update": global_update, **validation}
        history.append(record)
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        log(
            f"epoch={epoch + 1} validation mean={validation['mean']:.4f} "
            f"linear={validation['linear']:.4f} attention={validation['attention']:.4f}",
            log_path,
        )

        metadata = {"epoch": epoch + 1, "validation": validation, "config": asdict(config)}
        if "linear" in train_heads and validation["linear"] < best["linear"]:
            best["linear"] = validation["linear"]
            save_head(linear_head, output_dir / "linear_head_best.pt", metadata)
        if "attention" in train_heads and validation["attention"] < best["attention"]:
            best["attention"] = validation["attention"]
            save_head(attention_head, output_dir / "attention_head_best.pt", metadata)

    final_metadata = {"epoch": args.epochs, "validation": history[-1], "config": asdict(config)}
    if "linear" in train_heads:
        save_head(linear_head, output_dir / "linear_head_final.pt", final_metadata)
    if "attention" in train_heads:
        save_head(attention_head, output_dir / "attention_head_final.pt", final_metadata)
    log(f"Training complete: {output_dir}", log_path)


if __name__ == "__main__":
    main()
