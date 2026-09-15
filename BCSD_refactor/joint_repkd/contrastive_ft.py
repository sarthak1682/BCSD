#!/usr/bin/env python3
"""Contrastive retrieval-head fine-tune on a frozen joint_repkd backbone.

The repkd pretraining trains the shared backbone (causal NTP + bidirectional MLM
+ hidden-state KD) but leaves the retrieval head (attention pooling + linear
projection) at its random initialization.  This script loads that checkpoint,
freezes the backbone, and fits the retrieval head with a positive-aware
contrastive loss on Nova-format assembly pairs, producing the paper-comparable
"after CL" model.

The backbone is kept in eval mode throughout (dropout off) so the frozen
representations seen during fine-tuning match those used at inference.  Only
``pool_attention`` and ``retrieval_projection`` receive gradients.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from joint_distill.data import (  # noqa: E402
    JointPairDataset,
    build_joint_pairs,
    split_function_ids,
)
from joint_repkd.data import NovaRetrievalCollator  # noqa: E402
from joint_scratch.losses import retrieval_recall_at_1  # noqa: E402
from joint_scratch.model import ScratchJointModel  # noqa: E402
from joint_scratch.tokenizer import load_nova_tokenizer  # noqa: E402
from shared.data_utils import load_jsonl, set_seed  # noqa: E402
from shared.losses import contrastive_loss_positive_aware  # noqa: E402


def default_checkpoint() -> Path:
    return REPO_ROOT / "model_checkpoints" / "joint_repkd" / "20260822_153450"


def default_train_path() -> str:
    candidates = [
        "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_train_nova.jsonl",
        str(BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_train_nova.jsonl"),
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return candidates[0]


@dataclass
class RunConfig:
    checkpoint: str
    data: str
    output_dir: str
    seed: int
    epochs: int
    pair_batch_size: int
    grad_accum: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    contrastive_temperature: float
    val_fraction: float
    val_batches: int
    num_workers: int
    device: str
    amp: bool
    train_task_embedding: bool
    max_train_pairs: Optional[int]
    max_val_pairs: Optional[int]
    max_length: int
    trainable_parameters: int
    patience: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(default_checkpoint()))
    parser.add_argument("--data", default=default_train_path())
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--contrastive-temperature", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-batches", type=int, default=0)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use bfloat16 autocast on CUDA.",
    )
    parser.add_argument(
        "--train-task-embedding",
        action="store_true",
        help="Also keep the retrieval task embedding trainable.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop after this many epochs without validation improvement (0 disables).",
    )
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    path = path.resolve()
    if (path / "model_config.json").is_file() and (path / "model.pt").is_file():
        return path
    for split in ("best", "final"):
        candidate = path / split
        if (candidate / "model_config.json").is_file() and (candidate / "model.pt").is_file():
            return candidate
    raise FileNotFoundError(f"could not find checkpoint at {path} (best/final)")


def log(message: str, path: Path) -> None:
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def append_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def cosine_schedule(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


def compute_retrieval_step(
    model: ScratchJointModel,
    cpu_batch: Dict,
    device: torch.device,
    *,
    temperature: float,
    amp: bool,
):
    input_ids = cpu_batch["input_ids"].to(device, non_blocking=True)
    attention = cpu_batch["attention_mask"].to(device, non_blocking=True)
    pool_mask = cpu_batch["pool_mask"].to(device, non_blocking=True)
    func_ids = cpu_batch["func_ids"].to(device, non_blocking=True)
    with autocast_context(device, amp):
        output = model.forward_retrieval(input_ids, attention, pool_mask)
    embeddings = output.embeddings.float()
    loss = contrastive_loss_positive_aware(
        embeddings, func_ids, temperature=temperature
    )
    return loss, embeddings.detach().cpu(), func_ids.detach().cpu()


@torch.no_grad()
def validate(
    model: ScratchJointModel,
    loader: DataLoader,
    device: torch.device,
    *,
    temperature: float,
    amp: bool,
) -> Dict[str, float]:
    queries: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    query_ids: List[torch.Tensor] = []
    target_ids: List[torch.Tensor] = []
    total_loss = 0.0
    steps = 0
    for cpu_batch in loader:
        loss, embeddings, func_ids = compute_retrieval_step(
            model, cpu_batch, device, temperature=temperature, amp=amp
        )
        total_loss += float(loss.item())
        queries.append(embeddings[0::2])
        targets.append(embeddings[1::2])
        query_ids.append(func_ids[0::2])
        target_ids.append(func_ids[1::2])
        steps += 1
    metrics = {"loss": total_loss / max(steps, 1)}
    if queries:
        metrics["recall_at_1"] = retrieval_recall_at_1(
            torch.cat(queries),
            torch.cat(targets),
            torch.cat(query_ids),
            torch.cat(target_ids),
            compute_device=device if device.type == "cuda" else None,
        )
    return metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_seed(args.seed)

    checkpoint = resolve_checkpoint(Path(args.checkpoint))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = REPO_ROOT / "model_checkpoints" / "joint_repkd_cl" / run_id
    output_dir = Path(args.output_dir or default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "train.log"
    metrics_path = output_dir / "metrics.jsonl"

    _, nova_tokenizer = load_nova_tokenizer()
    model = ScratchJointModel.from_pretrained(checkpoint, map_location="cpu")
    model = model.to(device)
    model.freeze_backbone(train_task_embedding=args.train_task_embedding)
    model.eval()
    max_length = model.config.max_length
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    log(
        f"Checkpoint={checkpoint}; max_length={max_length}; "
        f"trainable={trainable_parameters:,} ({trainable_parameters / 1e6:.2f}M); "
        f"device={device}",
        log_path,
    )

    log(f"Loading Nova train pairs from {args.data}", log_path)
    samples = load_jsonl(args.data)
    train_ids, val_ids = split_function_ids(
        samples, val_fraction=args.val_fraction, seed=args.seed
    )
    train_pairs = build_joint_pairs(samples, train_ids, seed=args.seed)
    val_pairs = build_joint_pairs(samples, val_ids, seed=args.seed + 1)
    if args.max_train_pairs is not None:
        train_pairs = train_pairs[: args.max_train_pairs]
    if args.max_val_pairs is not None:
        val_pairs = val_pairs[: args.max_val_pairs]
    if not train_pairs:
        raise RuntimeError("training requires at least one pair")
    log(
        f"Function-disjoint train/val: {len(train_ids):,}/{len(val_ids):,}; "
        f"pairs: {len(train_pairs):,}/{len(val_pairs):,}",
        log_path,
    )

    collator = NovaRetrievalCollator(nova_tokenizer, max_length=max_length)
    loader_args = dict(
        batch_size=args.pair_batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        JointPairDataset(train_pairs), shuffle=True, drop_last=True, **loader_args
    )
    val_loader = None
    if val_pairs:
        val_loader = DataLoader(
            JointPairDataset(val_pairs), shuffle=False, drop_last=False, **loader_args
        )

    optimizer = AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_updates = updates_per_epoch * args.epochs
    scheduler = cosine_schedule(
        optimizer,
        warmup_steps=int(args.warmup_ratio * total_updates),
        total_steps=total_updates,
    )

    run_config = RunConfig(
        checkpoint=str(checkpoint),
        data=str(Path(args.data).resolve()),
        output_dir=str(output_dir),
        seed=args.seed,
        epochs=args.epochs,
        pair_batch_size=args.pair_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        contrastive_temperature=args.contrastive_temperature,
        val_fraction=args.val_fraction,
        val_batches=args.val_batches,
        num_workers=args.num_workers,
        device=str(device),
        amp=bool(args.amp and device.type == "cuda"),
        train_task_embedding=args.train_task_embedding,
        max_train_pairs=args.max_train_pairs,
        max_val_pairs=args.max_val_pairs,
        max_length=max_length,
        trainable_parameters=trainable_parameters,
        patience=args.patience,
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(run_config), handle, indent=2)

    best_recall = float("-inf")
    best_loss = float("inf")
    best_epoch: Optional[int] = None
    epochs_without_improvement = 0
    global_update = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        running_steps = 0
        for step, cpu_batch in enumerate(
            tqdm(train_loader, desc=f"contrastive ft epoch {epoch}/{args.epochs}"),
            start=1,
        ):
            loss, _, _ = compute_retrieval_step(
                model,
                cpu_batch,
                device,
                temperature=args.contrastive_temperature,
                amp=args.amp and device.type == "cuda",
            )
            (loss / args.grad_accum).backward()
            running_loss += float(loss.detach().item())
            running_steps += 1

            should_update = step % args.grad_accum == 0 or step == len(train_loader)
            if not should_update:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1

            if global_update % args.log_every == 0:
                record = {
                    "event": "train",
                    "epoch": epoch,
                    "update": global_update,
                    "loss": running_loss / max(running_steps, 1),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                append_jsonl(metrics_path, record)
                log(
                    f"update={global_update}/{total_updates} "
                    f"loss={record['loss']:.4f} lr={record['learning_rate']:.6f}",
                    log_path,
                )
                running_loss = 0.0
                running_steps = 0

        if val_loader is not None:
            validation = validate(
                model,
                val_loader,
                device,
                temperature=args.contrastive_temperature,
                amp=args.amp and device.type == "cuda",
            )
            validation_record = {
                "event": "validation",
                "epoch": epoch,
                "update": global_update,
                **validation,
            }
            append_jsonl(metrics_path, validation_record)
            log(
                f"validation epoch={epoch} loss={validation['loss']:.4f} "
                f"recall_at_1={validation.get('recall_at_1', float('nan')):.4f}",
                log_path,
            )
            recall = validation.get("recall_at_1", float("nan"))
            val_loss = validation["loss"]
            improved = val_loss < best_loss
            if improved:
                best_recall = recall
                best_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                model.save_pretrained(output_dir / "best")
                log(
                    f"saved best epoch={epoch} recall_at_1={recall:.4f} "
                    f"loss={val_loss:.4f}",
                    log_path,
                )
            else:
                epochs_without_improvement += 1
                if args.patience > 0 and epochs_without_improvement >= args.patience:
                    log(
                        f"early stopping after {epochs_without_improvement} epochs "
                        "without validation improvement",
                        log_path,
                    )
                    break

    model.save_pretrained(output_dir / "final")
    summary = {
        "checkpoint": str(checkpoint),
        "selection_metric": "validation_loss",
        "trainable_parameters": trainable_parameters,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "best_recall_at_1": None if val_loader is None else best_recall,
        "best_validation_loss": None if val_loader is None else best_loss,
        "best_epoch": best_epoch,
        "updates": global_update,
        "checkpoints": ["best", "final"] if val_loader is not None else ["final"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(f"Training complete: {output_dir}", log_path)


if __name__ == "__main__":
    main()
