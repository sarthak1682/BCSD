#!/usr/bin/env python3
"""Train a pure randomly initialized BCSD student with contrastive loss only.

Experiment 2 from the student-ablation task:
same deployed student architecture as the distilled BCSD student, no teacher,
no token MSE, and only positive-aware BCSD InfoNCE.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import socket
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.collators import DistillCollator  # noqa: E402
from shared.data_utils import load_jsonl, parse_bench_opt, set_seed  # noqa: E402
from shared.losses import contrastive_loss_positive_aware  # noqa: E402
from shared.nova_utils import setup_nova_tokenizer  # noqa: E402
from shared.student_model import LatentAttentionLayer, StudentDistillationModule  # noqa: E402


DEFAULT_DATA = BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_train_nova.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "model_checkpoints" / "nova_student_scratch_bcsd"
DEFAULT_HIDDEN_DIM = 2048
DEFAULT_STUDENT_LAYERS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--refine-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--student-layers", type=int, default=DEFAULT_STUDENT_LAYERS)
    parser.add_argument("--num-latents", type=int, default=512)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is unavailable")
    return device


def maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {"requested_device": str(device), "type": device.type}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(props.total_memory),
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return metadata


def safe_torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def assert_safe_output_dir(output_dir: Path, args: argparse.Namespace) -> None:
    if args.resume or args.allow_existing_output:
        return
    if not output_dir.exists():
        return
    protected = {
        "config.json",
        "training_history.csv",
        "training_history.json",
        "latest_training.pt",
        "SUMMARY.md",
        "student_best",
        "student_final",
        "phase_a_best",
        "phase_a_final",
    }
    existing = [item.name for item in output_dir.iterdir() if item.name in protected]
    if existing:
        preview = ", ".join(sorted(existing)[:8])
        raise RuntimeError(
            "Refusing fresh pure-BCSD training in a non-empty output directory: "
            f"{output_dir} ({preview}). Use --resume or choose a new --output-dir."
        )


def group_bench_samples(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        parse_bench_opt(sample["opt"])
        grouped.setdefault(str(sample["id"]), []).append(sample)
    return grouped


def build_bench_pairs(
    grouped: dict[str, list[dict[str, Any]]],
    func_ids: list[str],
    func_id_to_int: dict[str, int],
    *,
    seed: int,
) -> list[tuple[Any, Any, int]]:
    rng = random.Random(seed)
    pairs: list[tuple[Any, Any, int]] = []
    skipped = 0
    for fid in func_ids:
        variants = grouped[fid]
        if len(variants) < 2:
            skipped += 1
            continue
        ordered = variants[:]
        rng.shuffle(ordered)
        func_int = func_id_to_int[fid]
        for index, query in enumerate(ordered):
            positive = ordered[(index + 1) % len(ordered)]
            pairs.append((query["asm"], positive["asm"], func_int))
    rng.shuffle(pairs)
    print(
        f"built {len(pairs)} pairs from {len(func_ids)} functions "
        f"({skipped} skipped with <2 variants)",
        flush=True,
    )
    return pairs


def build_function_disjoint_pairs(
    samples: list[dict[str, Any]],
    *,
    val_split: float,
    seed: int,
    max_train_pairs: int | None,
    max_val_pairs: int | None,
) -> tuple[list[tuple[Any, Any, int]], list[tuple[Any, Any, int]], dict[str, Any]]:
    grouped = group_bench_samples(samples)
    valid_func_ids = [fid for fid, variants in grouped.items() if len(variants) >= 2]
    rng = random.Random(seed)
    rng.shuffle(valid_func_ids)
    val_func_count = max(1, int(len(valid_func_ids) * val_split)) if len(valid_func_ids) > 1 else 0
    val_func_ids = valid_func_ids[:val_func_count]
    train_func_ids = valid_func_ids[val_func_count:]
    func_id_to_int = {fid: index for index, fid in enumerate(sorted(valid_func_ids))}
    train_pairs = build_bench_pairs(grouped, train_func_ids, func_id_to_int, seed=seed)
    val_pairs = build_bench_pairs(grouped, val_func_ids, func_id_to_int, seed=seed + 1) if val_func_ids else []
    if max_train_pairs is not None:
        train_pairs = train_pairs[:max_train_pairs]
    if max_val_pairs is not None:
        val_pairs = val_pairs[:max_val_pairs]
    if not train_pairs:
        raise RuntimeError("No train pairs were built; check data path and schema")
    metadata = {
        "rows": len(samples),
        "valid_functions": len(valid_func_ids),
        "train_functions": len(train_func_ids),
        "val_functions": len(val_func_ids),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "function_disjoint_validation": True,
    }
    return train_pairs, val_pairs, metadata


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def save_student_checkpoint(
    out_dir: Path,
    *,
    student: StudentDistillationModule,
    lal_head: LatentAttentionLayer,
    config: dict[str, int],
    suffix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), out_dir / f"student_model{suffix}.pt")
    torch.save(lal_head.state_dict(), out_dir / f"lal_head{suffix}.pt")
    out_dir.joinpath("student_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_training_state(
    output_dir: Path,
    *,
    student: StudentDistillationModule,
    lal_head: LatentAttentionLayer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    phase: str,
    epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    best_phase_a_val_loss: float,
    best_refine_val_loss: float,
    bad_refine_epochs: int,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "phase": phase,
            "epoch": epoch,
            "global_step": global_step,
            "student_state_dict": student.state_dict(),
            "lal_head_state_dict": lal_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "history": history,
            "best_phase_a_val_loss": best_phase_a_val_loss,
            "best_refine_val_loss": best_refine_val_loss,
            "bad_refine_epochs": bad_refine_epochs,
            "config": config,
        },
        output_dir / "latest_training.pt",
    )


def write_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    output_dir.joinpath("training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not history:
        return
    preferred = [
        "phase",
        "epoch",
        "global_step",
        "train_loss",
        "val_loss",
        "improved",
        "epoch_seconds",
    ]
    observed = {key for row in history for key in row}
    fieldnames = [key for key in preferred if key in observed]
    fieldnames.extend(sorted(observed.difference(fieldnames)))
    with output_dir.joinpath("training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


@torch.no_grad()
def evaluate_contrastive(
    student: StudentDistillationModule,
    lal_head: LatentAttentionLayer,
    loader: DataLoader | None,
    *,
    device: torch.device,
    temperature: float,
    max_batches: int | None = None,
) -> float | None:
    if loader is None:
        return None
    student.eval()
    lal_head.eval()
    total = 0.0
    steps = 0
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        func_ids = batch["func_ids"].to(device, non_blocking=True)
        key_padding_mask = batch["key_padding_mask"].to(device, non_blocking=True)
        pool_mask = batch["pool_mask"].to(device, non_blocking=True)
        with maybe_autocast(device):
            hidden = student(input_ids, key_padding_mask=key_padding_mask)
            embeddings = lal_head(
                hidden,
                key_padding_mask=key_padding_mask,
                pool_mask=pool_mask,
            )
            loss = contrastive_loss_positive_aware(
                embeddings,
                func_ids,
                temperature=temperature,
            )
        total += float(loss.item())
        steps += 1
    student.train()
    lal_head.train()
    return total / max(1, steps)


def write_summary(
    output_dir: Path,
    *,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    best_refine_val_loss: float,
    completed: bool,
) -> None:
    latest_phase_a = next(
        (row for row in reversed(history) if row.get("phase") == "phase_a"),
        None,
    )
    latest_refine = next(
        (row for row in reversed(history) if row.get("phase") == "refine"),
        None,
    )

    def fmt(value: float | None) -> str:
        if value is None or math.isinf(value):
            return "pending"
        return f"{value:.4f}"

    lines = [
        "# Pure BCSD Scratch Student",
        "",
        "## Comparison Target",
        "",
        "| Model | Pool50 R@1 | Pool100 R@1 | Pool200 R@1 | Pool500 R@1 | Full R@1 | Full R@5 | Full R@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| Table I distilled BCSD student | 0.882 | 0.847 | 0.803 | 0.737 | 0.514 | 0.702 | 0.758 |",
        "| Pure BCSD scratch student | pending eval | pending eval | pending eval | pending eval | pending eval | pending eval | pending eval |",
        "",
        "## Recipe",
        "",
        f"- Status: {'training completed' if completed else 'configured / in progress'}",
        "- Teacher: none loaded.",
        "- Loss: positive-aware BCSD contrastive/InfoNCE only; no masked token MSE.",
        f"- Phase A: {config['training']['epochs']} contrastive-only epochs.",
        f"- Refine: up to {config['training']['refine_epochs']} contrastive-only epochs, early-stop patience {config['training']['early_stop_patience']}.",
        f"- Latest Phase A val loss: {fmt(None if latest_phase_a is None else latest_phase_a.get('val_loss'))}",
        f"- Latest refine val loss: {fmt(None if latest_refine is None else latest_refine.get('val_loss'))}",
        f"- Best refine val loss: {fmt(best_refine_val_loss)}",
        f"- Architecture: StudentDistillationModule(vocab_size={config['student']['vocab_size']}, hidden_dim={config['student']['hidden_dim']}, num_layers={config['student']['num_layers']}) + LatentAttentionLayer(num_latents={config['student']['num_latents']}).",
        f"- Parameter counts: backbone={config['parameter_counts']['backbone']:,}, LAL={config['parameter_counts']['latent_attention']:,}, total={config['parameter_counts']['total']:,}.",
        f"- Data: function-disjoint train/val split from `{config['data']['path']}`; query preprocessing uses the shared DistillCollator retrieval instruction and pool mask.",
    ]
    output_dir.joinpath("SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def train_phase(
    *,
    phase: str,
    start_epoch: int,
    epochs: int,
    student: StudentDistillationModule,
    lal_head: LatentAttentionLayer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    output_dir: Path,
    student_config: dict[str, int],
    config: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    best_phase_a_val_loss: float,
    best_refine_val_loss: float,
    bad_refine_epochs: int,
    resume_state: dict[str, Any] | None = None,
) -> tuple[int, float, float, int, list[dict[str, Any]], bool]:
    total_batches = len(train_loader)
    if args.max_train_batches is not None:
        total_batches = min(total_batches, args.max_train_batches)
    total_steps = max(1, math.ceil(total_batches / args.grad_accum_steps) * max(epochs, 1))
    optimizer = AdamW(
        list(student.parameters()) + list(lal_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(0.05 * total_steps),
        total_steps,
    )
    global_step = 0
    if resume_state is not None and resume_state.get("phase") == phase:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler_state = resume_state.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        global_step = int(resume_state.get("global_step", 0))

    completed = False
    for epoch in range(start_epoch + 1, epochs + 1):
        student.train()
        lal_head.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.monotonic()
        running = 0.0
        processed = 0
        print(f"{phase} epoch {epoch}/{epochs} start", flush=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            if args.max_train_batches is not None and batch_index > args.max_train_batches:
                break
            processed += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            func_ids = batch["func_ids"].to(device, non_blocking=True)
            key_padding_mask = batch["key_padding_mask"].to(device, non_blocking=True)
            pool_mask = batch["pool_mask"].to(device, non_blocking=True)
            with maybe_autocast(device):
                hidden = student(input_ids, key_padding_mask=key_padding_mask)
                embeddings = lal_head(
                    hidden,
                    key_padding_mask=key_padding_mask,
                    pool_mask=pool_mask,
                )
                loss = contrastive_loss_positive_aware(
                    embeddings,
                    func_ids,
                    temperature=args.temperature,
                )
            (loss / args.grad_accum_steps).backward()
            running += float(loss.item())
            if batch_index % args.grad_accum_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(lal_head.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if args.log_every > 0 and (
                batch_index == 1 or batch_index % args.log_every == 0
            ):
                print(
                    f"{phase} epoch {epoch}: batch {batch_index}/{len(train_loader)} "
                    f"loss={running / max(processed, 1):.4f}",
                    flush=True,
                )

        if processed and processed % args.grad_accum_steps != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(lal_head.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        val_loss = evaluate_contrastive(
            student,
            lal_head,
            val_loader,
            device=device,
            temperature=args.temperature,
            max_batches=args.max_val_batches,
        )
        improved = False
        if phase == "phase_a" and val_loss is not None:
            improved = val_loss + args.early_stop_min_delta < best_phase_a_val_loss
            if improved:
                best_phase_a_val_loss = float(val_loss)
                save_student_checkpoint(
                    output_dir / "phase_a_best",
                    student=student,
                    lal_head=lal_head,
                    config=student_config,
                    suffix="_best",
                )
        elif phase == "refine" and val_loss is not None:
            improved = val_loss + args.early_stop_min_delta < best_refine_val_loss
            if improved:
                best_refine_val_loss = float(val_loss)
                bad_refine_epochs = 0
                save_student_checkpoint(
                    output_dir / "student_best",
                    student=student,
                    lal_head=lal_head,
                    config=student_config,
                    suffix="_best",
                )
            else:
                bad_refine_epochs += 1

        row = {
            "phase": phase,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running / max(processed, 1),
            "val_loss": val_loss,
            "improved": improved,
            "epoch_seconds": time.monotonic() - epoch_start,
        }
        history.append(row)
        write_history(output_dir, history)
        save_training_state(
            output_dir,
            student=student,
            lal_head=lal_head,
            optimizer=optimizer,
            scheduler=scheduler,
            phase=phase,
            epoch=epoch,
            global_step=global_step,
            history=history,
            best_phase_a_val_loss=best_phase_a_val_loss,
            best_refine_val_loss=best_refine_val_loss,
            bad_refine_epochs=bad_refine_epochs,
            config=config,
        )
        write_summary(
            output_dir,
            config=config,
            history=history,
            best_refine_val_loss=best_refine_val_loss,
            completed=False,
        )
        print(
            f"{phase} epoch {epoch} done: train_loss={row['train_loss']:.4f} "
            f"val_loss={fmt_loss(val_loss)} improved={improved}",
            flush=True,
        )
        if phase == "refine" and bad_refine_epochs >= args.early_stop_patience:
            print("refine early stopping triggered", flush=True)
            break
    else:
        completed = True
    return (
        global_step,
        best_phase_a_val_loss,
        best_refine_val_loss,
        bad_refine_epochs,
        history,
        completed,
    )


def fmt_loss(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.epochs < 0 or args.refine_epochs < 0:
        raise ValueError("--epochs and --refine-epochs must be non-negative")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_safe_output_dir(output_dir, args)
    set_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(
        f"pure BCSD scratch starting: device={device} dtype={dtype} "
        f"batch_size={args.batch_size} grad_accum={args.grad_accum_steps}",
        flush=True,
    )
    base_tokenizer, nova_tokenizer, _ = setup_nova_tokenizer()
    pad_id = base_tokenizer.pad_token_id or 0
    student_config = {
        "vocab_size": len(base_tokenizer),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.student_layers,
    }
    student = StudentDistillationModule(
        vocab_size=student_config["vocab_size"],
        hidden_dim=student_config["hidden_dim"],
        num_layers=student_config["num_layers"],
        pad_id=pad_id,
    ).to(device=device, dtype=dtype)
    lal_head = LatentAttentionLayer(
        hidden_dim=student_config["hidden_dim"],
        num_latents=args.num_latents,
    ).to(device=device, dtype=dtype)
    parameter_counts = {
        "backbone": count_trainable_parameters(student),
        "latent_attention": count_trainable_parameters(lal_head),
    }
    parameter_counts["total"] = (
        parameter_counts["backbone"] + parameter_counts["latent_attention"]
    )
    print(f"model ready: params={parameter_counts['total']:,}", flush=True)

    samples = load_jsonl(args.data)
    train_pairs, val_pairs, data_metadata = build_function_disjoint_pairs(
        samples,
        val_split=args.val_split,
        seed=args.seed,
        max_train_pairs=args.max_train_pairs,
        max_val_pairs=args.max_val_pairs,
    )
    collator = DistillCollator(nova_tokenizer, max_length=args.max_length)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_pairs,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    val_loader = (
        DataLoader(
            val_pairs,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        if val_pairs
        else None
    )
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "experiment": "nova_student_scratch_bcsd",
        "data": {"path": str(Path(args.data)), **data_metadata},
        "device": device_metadata(device),
        "student": {
            **student_config,
            "num_latents": args.num_latents,
            "pad_id": pad_id,
        },
        "parameter_counts": parameter_counts,
        "training": {
            "epochs": args.epochs,
            "refine_epochs": args.refine_epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "temperature": args.temperature,
            "max_length": args.max_length,
            "seed": args.seed,
            "val_split": args.val_split,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "objective": "contrastive_loss_positive_aware only",
            "teacher": None,
            "token_mse": False,
        },
    }
    output_dir.joinpath("config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        write_summary(
            output_dir,
            config=config,
            history=[],
            best_refine_val_loss=float("inf"),
            completed=False,
        )
        print(json.dumps({"dry_run": True, "config": config}, indent=2, sort_keys=True))
        return

    history: list[dict[str, Any]] = []
    best_phase_a_val_loss = float("inf")
    best_refine_val_loss = float("inf")
    bad_refine_epochs = 0
    resume_state = None
    phase_a_start_epoch = 0
    refine_start_epoch = 0
    if args.resume and output_dir.joinpath("latest_training.pt").is_file():
        resume_state = safe_torch_load(output_dir / "latest_training.pt", map_location=device)
        student.load_state_dict(resume_state["student_state_dict"])
        lal_head.load_state_dict(resume_state["lal_head_state_dict"])
        history = list(resume_state.get("history", []))
        best_phase_a_val_loss = float(
            resume_state.get("best_phase_a_val_loss", float("inf"))
        )
        best_refine_val_loss = float(
            resume_state.get("best_refine_val_loss", float("inf"))
        )
        bad_refine_epochs = int(resume_state.get("bad_refine_epochs", 0))
        if resume_state.get("phase") == "phase_a":
            phase_a_start_epoch = int(resume_state.get("epoch", 0))
        elif resume_state.get("phase") == "refine":
            phase_a_start_epoch = args.epochs
            refine_start_epoch = int(resume_state.get("epoch", 0))
        print(
            f"resumed phase={resume_state.get('phase')} epoch={resume_state.get('epoch')}",
            flush=True,
        )

    if phase_a_start_epoch < args.epochs:
        _, best_phase_a_val_loss, best_refine_val_loss, bad_refine_epochs, history, _ = train_phase(
            phase="phase_a",
            start_epoch=phase_a_start_epoch,
            epochs=args.epochs,
            student=student,
            lal_head=lal_head,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            student_config=student_config,
            config=config,
            device=device,
            args=args,
            history=history,
            best_phase_a_val_loss=best_phase_a_val_loss,
            best_refine_val_loss=best_refine_val_loss,
            bad_refine_epochs=bad_refine_epochs,
            resume_state=resume_state,
        )
    save_student_checkpoint(
        output_dir / "phase_a_final",
        student=student,
        lal_head=lal_head,
        config=student_config,
    )

    if refine_start_epoch < args.refine_epochs:
        _, best_phase_a_val_loss, best_refine_val_loss, bad_refine_epochs, history, _ = train_phase(
            phase="refine",
            start_epoch=refine_start_epoch,
            epochs=args.refine_epochs,
            student=student,
            lal_head=lal_head,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            student_config=student_config,
            config=config,
            device=device,
            args=args,
            history=history,
            best_phase_a_val_loss=best_phase_a_val_loss,
            best_refine_val_loss=best_refine_val_loss,
            bad_refine_epochs=bad_refine_epochs,
            resume_state=resume_state,
        )

    if not output_dir.joinpath("student_best").is_dir():
        save_student_checkpoint(
            output_dir / "student_best",
            student=student,
            lal_head=lal_head,
            config=student_config,
            suffix="_best",
        )
    save_student_checkpoint(
        output_dir / "student_final",
        student=student,
        lal_head=lal_head,
        config=student_config,
    )
    write_summary(
        output_dir,
        config=config,
        history=history,
        best_refine_val_loss=best_refine_val_loss,
        completed=True,
    )
    print("pure BCSD scratch training finished", flush=True)


if __name__ == "__main__":
    main()

