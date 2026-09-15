#!/usr/bin/env python3
"""Train a randomly initialized student+LAL baseline for opt classification."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from opt_classification.scratch_baseline.data import (
    CLASS_LABELS,
    NovaAssemblyDataset,
    collate_batch,
    prepare_split_jsonls,
)
from opt_classification.scratch_baseline.eval import (
    add_bootstrap,
    autocast_context,
    binary_o0_vs_o3_from_4way,
    predict_loader,
    serializable_test_result,
)
from opt_classification.scratch_baseline.model import build_model_from_config
from shared.data_utils import set_seed
from shared.nova_utils import setup_nova_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "opt_classification" / "outputs" / "table1_opt_probe"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "scratch_student_opt"
DEFAULT_MANIFEST = DEFAULT_RUN_ROOT / "manifest.jsonl"
DEFAULT_STUDENT_CONFIG = Path(
    "/home/ra72yeq/BCSD/model_checkpoints/"
    "nova_student_bench_contrastive_ft_bs64/student_best/student_config.json"
)
DEFAULT_TRAIN_SOURCE = Path(
    "/home/ra72yeq/BCSD/nvemb/output_benchset_rebalanced_train_nova.jsonl"
)
DEFAULT_TEST_SOURCE = Path(
    "/home/ra72yeq/BCSD/nvemb/output_benchset_rebalanced_test_nova.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--train-source", default=str(DEFAULT_TRAIN_SOURCE))
    parser.add_argument("--test-source", default=str(DEFAULT_TEST_SOURCE))
    parser.add_argument("--student-config", default=str(DEFAULT_STUDENT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-latents", type=int, default=512)
    parser.add_argument("--normalize-embeddings", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-checkpoint",
        default="latest",
        help="'latest', 'best', or a checkpoint path for --eval-only.",
    )
    parser.add_argument(
        "--eval-tag",
        default=None,
        help="Suffix for eval-only result files; defaults to checkpoint name.",
    )
    parser.add_argument(
        "--save-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save checkpoints/epoch_XXX.pt after each completed epoch.",
    )
    parser.add_argument(
        "--test-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the full test split after each epoch; never used for checkpoint selection.",
    )
    parser.add_argument(
        "--epoch-test-bootstrap-samples",
        type=int,
        default=0,
        help="Bootstrap samples for per-epoch test evals; 0 records scalar metrics only.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow fresh training to use an output dir that already has run artifacts.",
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


def device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {"requested_device": str(device), "type": device.type}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(properties.total_memory),
                "capability": f"{properties.major}.{properties.minor}",
            }
        )
    return metadata


def assert_safe_fresh_output_dir(output_dir: Path, args: argparse.Namespace) -> None:
    if args.eval_only or args.resume or args.allow_existing_output:
        return
    if not output_dir.exists():
        return

    dangerous_names = {
        "best.pt",
        "latest.pt",
        "metrics_history.csv",
        "convergence.png",
        "config.json",
        "SUMMARY.md",
    }
    dangerous_prefixes = (
        "test_4way",
        "test_o0_vs_o3",
        "eval_config_",
        "eval_",
    )
    existing = []
    for item in output_dir.iterdir():
        if item.name in dangerous_names or item.name == "checkpoints":
            existing.append(item.name)
        elif any(item.name.startswith(prefix) for prefix in dangerous_prefixes):
            existing.append(item.name)
    if existing:
        preview = ", ".join(sorted(existing)[:8])
        raise RuntimeError(
            "Refusing to start fresh training in a non-empty run directory with "
            f"existing artifacts: {output_dir} ({preview}). Use --resume to "
            "continue that run, choose a new --output-dir for a fresh run, or "
            "pass --allow-existing-output if you really intend to reuse it."
        )


def make_loader(
    path: str | Path,
    *,
    max_length: int,
    pad_id: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    dataset = NovaAssemblyDataset(path, max_length=max_length, pad_id=pad_id)
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


def write_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    preferred = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "test_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_o0_vs_o3_accuracy",
        "test_o0_vs_o3_macro_f1",
        "test_o0_vs_o3_roc_auc",
        "epoch_seconds",
    ]
    observed = {key for row in history for key in row}
    fieldnames = [key for key in preferred if key in observed]
    fieldnames.extend(sorted(observed.difference(fieldnames)))
    history_path = output_dir / "metrics_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def plot_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    epochs = [row["epoch"] for row in history]
    fig, axes = pyplot.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_macro_f1"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("macro-F1")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=180)
    pyplot.close(fig)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_macro_f1: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "history": history,
            "config": config,
        },
        path,
    )


def save_epoch_checkpoint(
    output_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_macro_f1: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    save_checkpoint(
        epoch_path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_val_macro_f1=best_val_macro_f1,
        history=history,
        config=config,
    )
    return epoch_path


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


def resolve_eval_checkpoint(output_dir: Path, checkpoint: str) -> tuple[Path, str]:
    if checkpoint in {"latest", "best"}:
        return output_dir / f"{checkpoint}.pt", checkpoint
    path = Path(checkpoint)
    tag = path.stem.replace(" ", "_").replace(".", "_")
    return path, tag


def summarize_training_batch(
    true_indices: list[int],
    pred_indices: list[int],
    *,
    total_loss: float,
    total_rows: int,
) -> dict[str, float]:
    y_true = [CLASS_LABELS[index] for index in true_indices]
    y_pred = [CLASS_LABELS[index] for index in pred_indices]
    return {
        "loss": total_loss / max(total_rows, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(CLASS_LABELS), average="macro", zero_division=0)
        ),
    }


def read_probe_metric(run_root: Path, model_name: str, task: str, metric: str) -> float | None:
    path = run_root / "probes" / model_name / f"{task}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("test_metrics", {}).get(metric)


def read_result_metric(path: Path, metric: str) -> float | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("test_metrics", {}).get(metric)


def write_summary(
    output_dir: Path,
    *,
    config: dict[str, Any],
    scratch_4way: dict[str, Any] | None,
    scratch_binary: dict[str, Any] | None,
    best_epoch: int | None,
    completed: bool,
) -> None:
    run_root = output_dir.parent
    rows = []
    for model_name, label in [
        ("clap", "CLAP probe"),
        ("jtrans", "jTrans probe"),
        ("ebm-full", "EBM-full probe"),
        ("ebm-no-causal", "EBM-no-causal probe"),
        ("student", "BCSD Student probe"),
    ]:
        rows.append(
            {
                "model": label,
                "acc": read_probe_metric(run_root, model_name, "4way", "accuracy"),
                "f1": read_probe_metric(run_root, model_name, "4way", "macro_f1"),
                "binary_acc": read_probe_metric(run_root, model_name, "o0-vs-o3", "accuracy"),
                "binary_auc": read_probe_metric(run_root, model_name, "o0-vs-o3", "roc_auc"),
            }
        )
    for pure_dir in (output_dir, run_root / "scratch_student_opt_cuda0_fresh_5ep"):
        pure_4way = pure_dir / "test_4way_epoch_001.json"
        pure_binary = pure_dir / "test_o0_vs_o3_from_4way_epoch_001.json"
        if pure_4way.is_file() and pure_binary.is_file():
            rows.append(
                {
                    "model": "Pure baseline (epoch 1)",
                    "acc": read_result_metric(pure_4way, "accuracy"),
                    "f1": read_result_metric(pure_4way, "macro_f1"),
                    "binary_acc": read_result_metric(pure_binary, "accuracy"),
                    "binary_auc": read_result_metric(pure_binary, "roc_auc"),
                }
            )
            break
    if scratch_4way is not None:
        rows.append(
            {
                "model": "Scratch Student end-to-end",
                "acc": scratch_4way["test_metrics"]["accuracy"],
                "f1": scratch_4way["test_metrics"]["macro_f1"],
                "binary_acc": (
                    None
                    if scratch_binary is None
                    else scratch_binary["test_metrics"]["accuracy"]
                ),
                "binary_auc": (
                    None
                    if scratch_binary is None
                    else scratch_binary["test_metrics"].get("roc_auc")
                ),
            }
        )

    def fmt(value: float | None) -> str:
        return "pending" if value is None else f"{value:.3f}"

    lines = [
        "# Scratch Student Opt Baseline",
        "",
        "## jTrans vs CLAP check",
        "",
        "The claim holds under the frozen-probe protocol: Table I BCSD full-corpus R@1 has CLAP at 0.493 and jTrans at 0.425, while the matched opt-classification probe has jTrans above CLAP.",
        "",
        "## Comparison",
        "",
        "| Model | 4-way Acc | 4-way Macro-F1 | O0-vs-O3 Acc | O0-vs-O3 AUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {fmt(row['acc'])} | {fmt(row['f1'])} | {fmt(row['binary_acc'])} | {fmt(row.get('binary_auc'))} |"
        )

    lines.extend(
        [
            "",
            "## Scratch Recipe",
            "",
            f"- Status: {'completed test evaluation' if completed else 'training artifacts initialized / not fully evaluated'}",
            f"- Best epoch: {best_epoch if best_epoch is not None else 'pending'}",
            f"- Architecture: StudentDistillationModule(vocab_size={config['student_config']['vocab_size']}, hidden_dim={config['student_config']['hidden_dim']}, num_layers={config['student_config']['num_layers']}) + LatentAttentionLayer(num_latents={config['training']['num_latents']}) + Linear(2048, 4).",
            f"- Parameter counts: backbone={config['parameter_counts']['backbone']:,}, LAL={config['parameter_counts']['latent_attention']:,}, classifier={config['parameter_counts']['classifier']:,}, total={config['parameter_counts']['total']:,}.",
            f"- Optimizer: AdamW, lr={config['training']['lr']}, weight_decay={config['training']['weight_decay']}, grad_clip={config['training']['grad_clip']}, batch_size={config['training']['batch_size']}, grad_accum_steps={config['training']['grad_accum_steps']}, precision={config['training']['precision']}.",
            f"- Data: manifest-aligned Nova assembly only, max_length={config['training']['max_length']}, no BCSD retrieval instruction prefix.",
            "",
            "## Caveats",
            "",
            "- The scratch O0-vs-O3 number, when present, is computed from the trained 4-way model by restricting logits to O0/O3; it is not a separately trained binary head.",
            "- The frozen student probe is a linear probe on BCSD-trained embeddings; the scratch student is trained end-to-end for the opt-class objective, so this is an intentionally strong baseline.",
        ]
    )
    output_dir.joinpath("SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    tag: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    suffix = "" if tag is None else f"_{tag}"
    log_prefix = "test" if tag is None else f"test {tag}"
    test_result = predict_loader(
        model,
        test_loader,
        device=device,
        precision=precision,
        max_batches=max_batches,
        log_every=max(log_every, 0),
        log_prefix=log_prefix,
    )
    bootstrap = add_bootstrap(
        test_result,
        samples=bootstrap_samples,
        seed=seed,
    )
    test_4way = serializable_test_result(
        test_result,
        bootstrap=bootstrap,
        epoch=checkpoint_epoch,
        counts=counts,
    )
    output_dir.joinpath(f"test_4way{suffix}.json").write_text(
        json.dumps(test_4way, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    binary_result = binary_o0_vs_o3_from_4way(test_result)
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
        "test_predictions": {
            "ids": binary_result["ids"],
            "true": binary_result["true"],
            "predicted": binary_result["predicted"],
            "positive_scores": binary_result["positive_scores"],
        },
    }
    output_dir.joinpath(f"test_o0_vs_o3_from_4way{suffix}.json").write_text(
        json.dumps(test_binary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checkpoint_epoch": checkpoint_epoch,
                "tag": tag or "final",
                "test_4way": test_4way["test_metrics"],
                "test_o0_vs_o3_from_4way": test_binary["test_metrics"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return test_4way, test_binary


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.epochs < 0:
        raise ValueError("--epochs must be non-negative")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_safe_fresh_output_dir(output_dir, args)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda was set, but no CUDA device is available")
    precision = args.precision if device.type == "cuda" else "fp32"
    print(
        f"scratch baseline starting: device={device} precision={precision} "
        f"batch_size={args.batch_size} grad_accum={args.grad_accum_steps} "
        f"workers={args.num_workers}",
        flush=True,
    )

    base_tokenizer, _, _ = setup_nova_tokenizer()
    pad_id = base_tokenizer.pad_token_id or 0
    print(f"tokenizer ready: pad_id={pad_id}", flush=True)
    student_config = json.loads(Path(args.student_config).read_text(encoding="utf-8"))
    model = build_model_from_config(
        student_config,
        pad_id=pad_id,
        num_latents=args.num_latents,
        num_classes=len(CLASS_LABELS),
        normalize_embeddings=args.normalize_embeddings,
    )
    parameter_counts = model.parameter_counts().as_dict()
    dtype = torch.bfloat16 if device.type == "cuda" and precision == "bf16" else torch.float32
    if device.type == "cuda" and precision == "fp16":
        dtype = torch.float16
    model = model.to(device=device, dtype=dtype)
    print(
        f"model ready: params={parameter_counts['total']:,} dtype={dtype} "
        f"device={next(model.parameters()).device}",
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
        f"reused={data_cache['reused']} "
        f"train={data_cache['counts']['train']['rows']} "
        f"val={data_cache['counts']['val']['rows']} "
        f"test={data_cache['counts']['test']['rows']}",
        flush=True,
    )
    eval_batch_size = args.eval_batch_size or args.batch_size
    loaders: dict[str, DataLoader] = {}
    if not args.eval_only:
        loaders["train"] = make_loader(
            data_cache["paths"]["train"],
            max_length=args.max_length,
            pad_id=pad_id,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
        )
        loaders["val"] = make_loader(
            data_cache["paths"]["val"],
            max_length=args.max_length,
            pad_id=pad_id,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
        )
    loaders["test"] = make_loader(
        data_cache["paths"]["test"],
        max_length=args.max_length,
        pad_id=pad_id,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    loader_counts = {split: len(loader) for split, loader in loaders.items()}
    print(
        "loaders ready: "
        + " ".join(f"{split}_batches={count}" for split, count in loader_counts.items()),
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "manifest": str(Path(args.manifest)),
        "train_source": str(Path(args.train_source)),
        "test_source": str(Path(args.test_source)),
        "student_config_path": str(Path(args.student_config)),
        "student_config": student_config,
        "parameter_counts": parameter_counts,
        "device": device_metadata(device),
        "data_cache": data_cache,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": eval_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "grad_accum_steps": args.grad_accum_steps,
            "max_length": args.max_length,
            "seed": args.seed,
            "patience": args.patience,
            "precision": precision,
            "num_latents": args.num_latents,
            "normalize_embeddings": args.normalize_embeddings,
            "bootstrap_samples": args.bootstrap_samples,
            "log_every": args.log_every,
            "save_every_epoch": args.save_every_epoch,
            "test_every_epoch": args.test_every_epoch,
            "epoch_test_bootstrap_samples": args.epoch_test_bootstrap_samples,
        },
    }
    if not args.eval_only:
        output_dir.joinpath("config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_summary(
            output_dir,
            config=config,
            scratch_4way=None,
            scratch_binary=None,
            best_epoch=None,
            completed=False,
        )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": config}, indent=2, sort_keys=True))
        return

    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_val_macro_f1 = float("-inf")
    best_epoch: int | None = None
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    if args.eval_only:
        checkpoint_path, default_tag = resolve_eval_checkpoint(
            output_dir, args.eval_checkpoint
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        tag = args.eval_tag or default_tag
        eval_config = dict(config)
        eval_config["eval_only"] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "tag": tag,
        }
        output_dir.joinpath(f"eval_config_{tag}.json").write_text(
            json.dumps(eval_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"eval-only: loaded {checkpoint_path} epoch={checkpoint_epoch} tag={tag}",
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
            tag=tag,
        )
        return

    if args.resume and latest_path.is_file():
        checkpoint = load_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        start_epoch = int(checkpoint["epoch"])
        best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", float("-inf")))
        history = list(checkpoint.get("history", []))
        if history:
            best_epoch = max(history, key=lambda row: row["val_macro_f1"])["epoch"]
        print(
            f"resumed from {latest_path}: start_epoch={start_epoch} "
            f"best_val_macro_f1={best_val_macro_f1:.4f}",
            flush=True,
        )
    elif args.resume:
        print("resume requested, but no latest.pt exists; starting fresh", flush=True)

    criterion = torch.nn.CrossEntropyLoss()
    start_time = time.monotonic()
    epochs_without_improvement = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_rows = 0
        train_true: list[int] = []
        train_pred: list[int] = []
        epoch_start = time.monotonic()
        processed_batches = 0
        log_window_loss = 0.0
        log_window_rows = 0
        print(
            f"epoch {epoch}/{args.epochs} train start: "
            f"{len(loaders['train'])} batches",
            flush=True,
        )

        for batch_index, batch in enumerate(loaders["train"], start=1):
            if args.max_train_batches is not None and batch_index > args.max_train_batches:
                break
            processed_batches += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            key_padding_mask = batch["key_padding_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            if batch_index == 1:
                print(
                    f"epoch {epoch}: first batch loaded "
                    f"shape={tuple(input_ids.shape)}",
                    flush=True,
                )
            with autocast_context(device, precision):
                logits = model(input_ids, key_padding_mask=key_padding_mask)
                loss = criterion(logits.float(), labels)
            (loss / args.grad_accum_steps).backward()
            if batch_index % args.grad_accum_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_rows = int(labels.numel())
            train_loss += float(loss.item()) * batch_rows
            train_rows += batch_rows
            log_window_loss += float(loss.item()) * batch_rows
            log_window_rows += batch_rows
            train_true.extend(labels.detach().cpu().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
            if args.log_every > 0 and batch_index % args.log_every == 0:
                elapsed = time.monotonic() - epoch_start
                print(
                    f"epoch {epoch}: batch {batch_index}/{len(loaders['train'])} "
                    f"rows={train_rows} "
                    f"loss={log_window_loss / max(log_window_rows, 1):.4f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                log_window_loss = 0.0
                log_window_rows = 0

        if train_rows and processed_batches % args.grad_accum_steps != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_metrics = summarize_training_batch(
            train_true,
            train_pred,
            total_loss=train_loss,
            total_rows=train_rows,
        )
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
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": time.monotonic() - epoch_start,
        }
        history.append(row)
        write_history(output_dir, history)
        plot_history(output_dir, history)

        improved = val_metrics["macro_f1"] > best_val_macro_f1
        if improved:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_macro_f1=best_val_macro_f1,
                history=history,
                config=config,
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_macro_f1=best_val_macro_f1,
            history=history,
            config=config,
        )
        if args.save_every_epoch:
            epoch_path = save_epoch_checkpoint(
                output_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_macro_f1=best_val_macro_f1,
                history=history,
                config=config,
            )
            print(f"saved epoch checkpoint: {epoch_path}", flush=True)
        if args.test_every_epoch:
            test_4way_epoch, test_binary_epoch = run_test_evaluation(
                model=model,
                test_loader=loaders["test"],
                device=device,
                precision=precision,
                output_dir=output_dir,
                counts=data_cache["counts"],
                bootstrap_samples=args.epoch_test_bootstrap_samples,
                seed=args.seed,
                checkpoint_epoch=epoch,
                log_every=args.log_every,
                max_batches=args.max_eval_batches,
                tag=f"epoch_{epoch:03d}",
            )
            row.update(
                {
                    "test_loss": test_4way_epoch["test_metrics"]["loss"],
                    "test_accuracy": test_4way_epoch["test_metrics"]["accuracy"],
                    "test_balanced_accuracy": test_4way_epoch["test_metrics"][
                        "balanced_accuracy"
                    ],
                    "test_macro_f1": test_4way_epoch["test_metrics"]["macro_f1"],
                    "test_o0_vs_o3_accuracy": test_binary_epoch["test_metrics"][
                        "accuracy"
                    ],
                    "test_o0_vs_o3_macro_f1": test_binary_epoch["test_metrics"][
                        "macro_f1"
                    ],
                    "test_o0_vs_o3_roc_auc": test_binary_epoch["test_metrics"].get(
                        "roc_auc"
                    ),
                }
            )
            history[-1] = row
            write_history(output_dir, history)
            save_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_macro_f1=best_val_macro_f1,
                history=history,
                config=config,
            )
            if improved:
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_macro_f1=best_val_macro_f1,
                    history=history,
                    config=config,
                )
            if args.save_every_epoch:
                epoch_path = save_epoch_checkpoint(
                    output_dir,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_macro_f1=best_val_macro_f1,
                    history=history,
                    config=config,
                )
                print(f"updated epoch checkpoint with test metrics: {epoch_path}", flush=True)
        print(json.dumps(row, sort_keys=True), flush=True)
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {epochs_without_improvement} epochs without validation macro-F1 improvement.",
                flush=True,
            )
            break

    if best_path.is_file():
        best_checkpoint = load_checkpoint(best_path, model=model, map_location=device)
        best_epoch = int(best_checkpoint["epoch"])
    test_4way, test_binary = run_test_evaluation(
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
        tag=None,
    )

    config["finished_at"] = datetime.now(timezone.utc).isoformat()
    config["wall_clock_seconds"] = time.monotonic() - start_time
    config["best_epoch"] = best_epoch
    output_dir.joinpath("config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(
        output_dir,
        config=config,
        scratch_4way=test_4way,
        scratch_binary=test_binary,
        best_epoch=best_epoch,
        completed=True,
    )


if __name__ == "__main__":
    main()
