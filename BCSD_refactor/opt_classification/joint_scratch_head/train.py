#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from opt_classification.joint_scratch_head.data import (
    JointScratchAssemblyDataset,
    build_collator,
    collate_batch,
)
from opt_classification.joint_scratch_head.model import JointScratchOptClassifier
from opt_classification.scratch_baseline.data import CLASS_LABELS, prepare_split_jsonls
from opt_classification.scratch_baseline.eval import (
    add_bootstrap,
    autocast_context,
    binary_o0_vs_o3_from_4way,
    predict_loader,
    serializable_test_result,
)
from shared.data_utils import set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "opt_classification" / "outputs" / "table1_opt_probe"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "joint_scratch_83m_precl_opt_head"
DEFAULT_MANIFEST = DEFAULT_RUN_ROOT / "manifest.jsonl"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "model_checkpoints"
    / "joint_scratch"
    / "clap_strict_83m_lm_mlm_pretrain10_trainonly_gpu1_seed42"
    / "final"
)
DEFAULT_TRAIN_SOURCE = (
    REPO_ROOT.parent / "nvemb"
    / "output_benchset_rebalanced_train_clap_strict.jsonl"
)
DEFAULT_TEST_SOURCE = (
    REPO_ROOT.parent / "nvemb"
    / "output_benchset_rebalanced_test_clap_strict.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--train-source", default=str(DEFAULT_TRAIN_SOURCE))
    parser.add_argument("--test-source", default=str(DEFAULT_TEST_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--selection-metric",
        choices=("val_acc", "val_loss", "val_macro_f1"),
        default="val_acc",
        help="Validation metric used to select the best checkpoint.",
    )
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-checkpoint",
        default="best",
        help="'latest', 'best', or a checkpoint path for --eval-only.",
    )
    parser.add_argument("--rebuild-data-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested {device_arg}, but torch.cuda.is_available() is false"
        )
    return device


def make_loader(
    path: str | Path,
    *,
    collator,
    pad_id: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    dataset = JointScratchAssemblyDataset(path, collator)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_batch(batch, pad_id=pad_id),
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best_score,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def selection_score(metrics: dict[str, Any], metric: str) -> float:
    key = {"val_acc": "accuracy", "val_loss": "loss", "val_macro_f1": "macro_f1"}[metric]
    value = float(metrics[key])
    return -value if metric == "val_loss" else value


def run_test_evaluation(
    *,
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    precision: str,
    output_dir: Path,
    counts: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
    checkpoint_epoch: int,
    log_every: int,
    max_batches: int | None = None,
) -> None:
    result = predict_loader(
        model,
        test_loader,
        device=device,
        precision=precision,
        max_batches=max_batches,
        log_every=log_every,
        log_prefix="test",
    )
    bootstrap = add_bootstrap(result, samples=bootstrap_samples, seed=seed)
    test_4way = serializable_test_result(
        result, bootstrap=bootstrap, epoch=checkpoint_epoch, counts=counts
    )
    output_dir.joinpath("test_4way.json").write_text(
        json.dumps(test_4way, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    binary_result = binary_o0_vs_o3_from_4way(result)
    binary_bootstrap = add_bootstrap(
        {
            "true": binary_result["true"],
            "predicted": binary_result["predicted"],
            "ids": binary_result["ids"],
        },
        samples=bootstrap_samples,
        seed=seed,
        labels=("O0", "O3"),
        positive_scores=np.asarray(binary_result["positive_scores"], dtype=float),
    )
    test_binary = {
        "task": binary_result["task"],
        "class_labels": binary_result["class_labels"],
        "checkpoint_epoch": checkpoint_epoch,
        "test_metrics": binary_result["metrics"],
        "bootstrap_95": binary_bootstrap,
        "counts": {
            "test_rows": len(binary_result["true"]),
            "test_functions": len(set(binary_result["ids"])),
        },
    }
    output_dir.joinpath("test_o0_vs_o3_from_4way.json").write_text(
        json.dumps(test_binary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "checkpoint_epoch": checkpoint_epoch,
                "test_4way": test_4way["test_metrics"],
                "test_o0_vs_o3_from_4way": test_binary["test_metrics"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda was set, but no CUDA device is available")
    precision = args.precision if device.type == "cuda" else "fp32"
    print(
        f"frozen-head opt-classifier: device={device} precision={precision} "
        f"batch_size={args.batch_size} grad_accum={args.grad_accum_steps}",
        flush=True,
    )

    data_cache = prepare_split_jsonls(
        manifest_path=args.manifest,
        train_source=args.train_source,
        test_source=args.test_source,
        output_dir=output_dir,
        rebuild=args.rebuild_data_cache,
    )
    print(
        "data cache ready: "
        f"train={data_cache['counts']['train']['rows']} "
        f"val={data_cache['counts']['val']['rows']} "
        f"test={data_cache['counts']['test']['rows']}",
        flush=True,
    )

    collator = build_collator(max_length=args.max_length)
    pad_id = collator.pad_id
    eval_batch_size = args.eval_batch_size or args.batch_size
    loaders: dict[str, DataLoader] = {}
    if not args.eval_only:
        loaders["train"] = make_loader(
            data_cache["paths"]["train"],
            collator=collator,
            pad_id=pad_id,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
        )
        loaders["val"] = make_loader(
            data_cache["paths"]["val"],
            collator=collator,
            pad_id=pad_id,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
        )
    loaders["test"] = make_loader(
        data_cache["paths"]["test"],
        collator=collator,
        pad_id=pad_id,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    print(
        "loaders ready: "
        + " ".join(f"{split}={len(loader)}" for split, loader in loaders.items()),
        flush=True,
    )

    model = JointScratchOptClassifier.from_pretrained(
        args.checkpoint, num_classes=len(CLASS_LABELS)
    )
    dtype = torch.bfloat16 if device.type == "cuda" and precision == "bf16" else torch.float32
    if device.type == "cuda" and precision == "fp16":
        dtype = torch.float16
    model = model.to(device=device, dtype=dtype)
    counts = model.parameter_counts().as_dict()
    print(
        f"model ready: trainable={counts['total_trainable']:,} "
        f"(pool_attention={counts['pool_attention']:,}, classifier={counts['classifier']:,})",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_metric": args.selection_metric,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "train_source": str(Path(args.train_source).resolve()),
        "test_source": str(Path(args.test_source).resolve()),
        "parameter_counts": counts,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "grad_accum_steps": args.grad_accum_steps,
            "max_length": args.max_length,
            "seed": args.seed,
            "patience": args.patience,
            "precision": precision,
            "bootstrap_samples": args.bootstrap_samples,
        },
    }
    if not args.eval_only:
        output_dir.joinpath("config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": config}, indent=2, sort_keys=True))
        return

    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    if args.eval_only:
        checkpoint_path = (
            best_path if args.eval_checkpoint == "best" else latest_path
        ) if args.eval_checkpoint in {"latest", "best"} else Path(args.eval_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        print(
            f"eval-only: loaded {checkpoint_path} epoch={checkpoint_epoch}",
            flush=True,
        )
        run_test_evaluation(
            model=model,
            test_loader=loaders["test"],
            device=device,
            precision=precision,
            output_dir=output_dir,
            counts=data_cache["counts"],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            checkpoint_epoch=checkpoint_epoch,
            log_every=args.log_every,
            max_batches=args.max_eval_batches,
        )
        return

    start_epoch = 0
    best_score = float("-inf")
    best_epoch: int | None = None
    if args.resume and latest_path.is_file():
        checkpoint = load_checkpoint(
            latest_path, model=model, optimizer=optimizer, map_location=device
        )
        start_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint.get("best_score", float("-inf")))
        print(
            f"resumed: start_epoch={start_epoch} best_score={best_score:.4f}",
            flush=True,
        )
    elif args.resume:
        print("resume requested, but no latest.pt; starting fresh", flush=True)

    criterion = torch.nn.CrossEntropyLoss()
    epochs_without_improvement = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_true: list[int] = []
        train_pred: list[int] = []
        train_loss = 0.0
        train_rows = 0
        processed_batches = 0
        for batch_index, batch in enumerate(loaders["train"], start=1):
            if args.max_train_batches is not None and batch_index > args.max_train_batches:
                break
            processed_batches += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            key_padding_mask = batch["key_padding_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast_context(device, precision):
                logits = model(input_ids, key_padding_mask=key_padding_mask)
                loss = criterion(logits.float(), labels)
            (loss / args.grad_accum_steps).backward()
            if batch_index % args.grad_accum_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters(model), args.grad_clip
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            batch_rows = int(labels.numel())
            train_loss += float(loss.item()) * batch_rows
            train_rows += batch_rows
            train_true.extend(labels.detach().cpu().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
            if args.log_every > 0 and batch_index % args.log_every == 0:
                print(
                    f"epoch {epoch}: batch {batch_index}/{len(loaders['train'])} "
                    f"loss={train_loss / max(train_rows, 1):.4f}",
                    flush=True,
                )
        if train_rows and processed_batches % args.grad_accum_steps != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_metrics = {
            "loss": train_loss / max(train_rows, 1),
            "accuracy": float(accuracy_score(
                [CLASS_LABELS[i] for i in train_true],
                [CLASS_LABELS[i] for i in train_pred],
            )),
            "macro_f1": float(f1_score(
                [CLASS_LABELS[i] for i in train_true],
                [CLASS_LABELS[i] for i in train_pred],
                labels=list(CLASS_LABELS), average="macro", zero_division=0,
            )),
        }
        val_result = predict_loader(
            model,
            loaders["val"],
            device=device,
            precision=precision,
            max_batches=args.max_eval_batches,
            log_every=max(args.log_every, 0),
            log_prefix=f"epoch {epoch} val",
        )
        val_metrics = val_result["metrics"]
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": {
                        k: val_metrics[k]
                        for k in ("loss", "accuracy", "balanced_accuracy", "macro_f1")
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

        score = selection_score(val_metrics, args.selection_metric)
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_score,
                config=config,
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_score=best_score,
            config=config,
        )
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {epochs_without_improvement} epochs without "
                f"validation {args.selection_metric} improvement.",
                flush=True,
            )
            break

    if best_path.is_file():
        load_checkpoint(best_path, model=model, map_location=device)
        best_epoch = int(
            torch.load(best_path, map_location="cpu", weights_only=False)["epoch"]
        )
    run_test_evaluation(
        model=model,
        test_loader=loaders["test"],
        device=device,
        precision=precision,
        output_dir=output_dir,
        counts=data_cache["counts"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        checkpoint_epoch=best_epoch or args.epochs,
        log_every=args.log_every,
        max_batches=args.max_eval_batches,
    )


if __name__ == "__main__":
    main()
