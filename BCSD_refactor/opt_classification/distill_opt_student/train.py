#!/usr/bin/env python3
"""Train a Nova-distilled student for optimization classification.

Experiment 1 from the student-ablation task:
random student + LAL + Linear(2048, 4), frozen EBM-full Nova teacher, and
loss = opt-class CE + lambda(s) * masked token MSE. There is intentionally no
BCSD contrastive/InfoNCE loss in this run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from opt_classification.scratch_baseline.data import (  # noqa: E402
    CLASS_LABELS,
    LABEL_TO_INDEX,
    prepare_split_jsonls,
)
from opt_classification.scratch_baseline.eval import (  # noqa: E402
    add_bootstrap,
    autocast_context,
    binary_o0_vs_o3_from_4way,
    predict_loader,
    serializable_test_result,
)
from opt_classification.probe import compute_metrics  # noqa: E402
from opt_classification.distill_opt_student.model import (  # noqa: E402
    build_model_from_config,
)
from shared.data_utils import asm_to_text, set_seed  # noqa: E402
from shared.losses import masked_mse_loss  # noqa: E402
from shared.nova_utils import NOVA_CACHE_DIR, setup_nova_tokenizer  # noqa: E402


DEFAULT_RUN_ROOT = REPO_ROOT / "opt_classification" / "outputs" / "table1_opt_probe"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "distill_opt_student"
DEFAULT_MANIFEST = DEFAULT_RUN_ROOT / "manifest.jsonl"
DEFAULT_SPLIT_DATA_DIR = DEFAULT_RUN_ROOT / "scratch_student_opt" / "data"
DEFAULT_STUDENT_CONFIG = Path(
    BCSD_ROOT
    / "model_checkpoints"
    / "nova_student_bench_contrastive_ft_bs64"
    / "student_best"
    / "student_config.json"
)
DEFAULT_TEACHER_DIR = BCSD_ROOT / "model_checkpoints" / "nova_ebm_bench" / "s3_final"
DEFAULT_TRAIN_SOURCE = BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_train_nova.jsonl"
DEFAULT_TEST_SOURCE = BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_test_nova.jsonl"
SCRATCH_OPT_BASELINE_ACC = 0.612
SCRATCH_OPT_BASELINE_F1 = 0.567


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split-data-dir", default=str(DEFAULT_SPLIT_DATA_DIR))
    parser.add_argument("--train-source", default=str(DEFAULT_TRAIN_SOURCE))
    parser.add_argument("--test-source", default=str(DEFAULT_TEST_SOURCE))
    parser.add_argument("--teacher-dir", default=str(DEFAULT_TEACHER_DIR))
    parser.add_argument("--student-config", default=str(DEFAULT_STUDENT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-latents", type=int, default=512)
    parser.add_argument("--normalize-embeddings", action="store_true")
    parser.add_argument("--lambda-start", type=float, default=1.0)
    parser.add_argument("--lambda-end", type=float, default=0.05)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epoch-test-bootstrap-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-checkpoint", default="latest")
    parser.add_argument("--eval-tag", default=None)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--rebuild-data-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is unavailable")
    return device


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


def precision_for_device(device: torch.device, requested: str) -> str:
    return requested if device.type == "cuda" else "fp32"


def dtype_for_precision(device: torch.device, precision: str) -> torch.dtype:
    if device.type == "cuda" and precision == "bf16":
        return torch.bfloat16
    if device.type == "cuda" and precision == "fp16":
        return torch.float16
    return torch.float32


def maybe_autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def safe_torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def assert_safe_output_dir(output_dir: Path, args: argparse.Namespace) -> None:
    if args.resume or args.eval_only or args.allow_existing_output:
        return
    if not output_dir.exists():
        return
    protected_names = {
        "best.pt",
        "latest.pt",
        "config.json",
        "metrics_history.csv",
        "SUMMARY.md",
    }
    protected_prefixes = ("test_4way", "test_o0_vs_o3", "eval_config_")
    existing = []
    for item in output_dir.iterdir():
        if item.name in protected_names or item.name == "checkpoints":
            existing.append(item.name)
        elif any(item.name.startswith(prefix) for prefix in protected_prefixes):
            existing.append(item.name)
    if existing:
        preview = ", ".join(sorted(existing)[:8])
        raise RuntimeError(
            "Refusing fresh opt-distill training in a non-empty run directory: "
            f"{output_dir} ({preview}). Use --resume or choose a new --output-dir."
        )


def count_jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def split_counts_from_paths(paths: Mapping[str, str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split, raw_path in paths.items():
        path = Path(raw_path)
        rows = 0
        ids: set[str] = set()
        binary_groups: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                ids.add(str(row["id"]))
                binary_groups.add(str(row.get("binary_group", row["id"])))
        counts[split] = {
            "rows": rows,
            "functions": len(ids),
            "binary_groups": len(binary_groups),
        }
    return counts


def resolve_split_data(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    split_data_dir = Path(args.split_data_dir) if args.split_data_dir else None
    if split_data_dir is not None and not args.rebuild_data_cache:
        paths = {split: split_data_dir / f"{split}.jsonl" for split in ("train", "val", "test")}
        if all(path.is_file() for path in paths.values()):
            metadata_path = split_data_dir / "cache_metadata.json"
            metadata = {}
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return {
                "paths": {split: str(path) for split, path in paths.items()},
                "counts": metadata.get("counts") or split_counts_from_paths(
                    {split: str(path) for split, path in paths.items()}
                ),
                "metadata_path": str(metadata_path) if metadata_path.is_file() else None,
                "manifest_sha256": metadata.get("manifest_sha256"),
                "reused": True,
                "source": "existing_split_data_dir",
            }

    prepared = prepare_split_jsonls(
        manifest_path=args.manifest,
        train_source=args.train_source,
        test_source=args.test_source,
        output_dir=output_dir,
        rebuild=args.rebuild_data_cache,
    )
    prepared["source"] = "prepared_under_output_dir"
    return prepared


class DistillOptDataset(Dataset):
    """Random-access split JSONL dataset that keeps the Nova attention mask."""

    def __init__(self, path: str | Path, *, max_length: int = 1024) -> None:
        self.path = Path(path)
        self.max_length = max_length
        self.offsets = self._build_offsets()
        self._handle = None
        self._tokenizer = None

    def _build_offsets(self) -> list[int]:
        offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    offsets.append(offset)
                offset += len(line)
        if not offsets:
            raise ValueError(f"Dataset is empty: {self.path}")
        return offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_tokenizer"] = None
        return state

    @property
    def handle(self):
        if self._handle is None:
            self._handle = self.path.open("r", encoding="utf-8")
        return self._handle

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            _, self._tokenizer, _ = setup_nova_tokenizer()
        return self._tokenizer

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.handle.seek(self.offsets[index])
        row = json.loads(self.handle.readline())
        text = asm_to_text(row["asm"])
        if not text.strip():
            raise ValueError(f"Empty assembly in {self.path}:{index}")
        encoded = self.tokenizer.encode("", text, "1" * len(text))
        ids = encoded["input_ids"][: self.max_length]
        input_ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        if not input_ids:
            raise ValueError(f"Nova tokenizer produced no ids in {self.path}:{index}")
        length = len(input_ids)
        nova_mask = encoded["nova_attention_mask"][:length, :length]
        label_name = str(row["optimization"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "nova_attention_mask": torch.tensor(nova_mask, dtype=torch.float32),
            "label": torch.tensor(LABEL_TO_INDEX[label_name], dtype=torch.long),
            "label_name": label_name,
            "id": str(row["id"]),
            "opt": str(row["opt"]),
        }


def collate_distill_opt(
    batch: Sequence[Mapping[str, Any]],
    *,
    pad_id: int,
    mask_dtype: torch.dtype,
) -> dict[str, Any]:
    width = max(int(item["input_ids"].numel()) for item in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    nova_mask = torch.zeros((len(batch), width, width), dtype=mask_dtype)
    ids: list[str] = []
    opts: list[str] = []
    label_names: list[str] = []
    for row_index, item in enumerate(batch):
        ids_tensor = item["input_ids"]
        length = int(ids_tensor.numel())
        input_ids[row_index, :length] = ids_tensor
        nova_mask[row_index, :length, :length] = item["nova_attention_mask"].to(mask_dtype)
        labels[row_index] = item["label"]
        ids.append(str(item["id"]))
        opts.append(str(item["opt"]))
        label_names.append(str(item["label_name"]))
    return {
        "input_ids": input_ids,
        "nova_attention_mask": nova_mask,
        "key_padding_mask": input_ids == pad_id,
        "pool_mask": input_ids == pad_id,
        "labels": labels,
        "ids": ids,
        "opts": opts,
        "label_names": label_names,
    }


def make_loader(
    path: str | Path,
    *,
    max_length: int,
    pad_id: int,
    mask_dtype: torch.dtype,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    dataset = DistillOptDataset(path, max_length=max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_distill_opt(
            batch,
            pad_id=pad_id,
            mask_dtype=mask_dtype,
        ),
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def load_teacher(
    teacher_dir: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
) -> torch.nn.Module:
    if not teacher_dir.is_dir():
        raise FileNotFoundError(f"Teacher checkpoint directory not found: {teacher_dir}")
    if NOVA_CACHE_DIR not in sys.path:
        sys.path.insert(0, NOVA_CACHE_DIR)
    from modeling_nova import NovaForCausalLM

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        kwargs["device_map"] = {"": index}
    teacher = NovaForCausalLM.from_pretrained(str(teacher_dir), **kwargs)
    teacher.resize_token_embeddings(vocab_size)
    if device.type != "cuda":
        teacher = teacher.to(device=device, dtype=dtype)
    teacher.requires_grad_(False)
    teacher.eval()
    if hasattr(teacher.config, "use_cache"):
        teacher.config.use_cache = False
    return teacher


def lambda_mse_for_step(
    global_step: int,
    total_steps: int,
    *,
    start: float,
    end: float,
) -> float:
    if total_steps <= 1:
        return end
    progress = min(global_step, total_steps - 1) / (total_steps - 1)
    return start + (end - start) * (progress**2)


def write_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    preferred = [
        "epoch",
        "train_loss",
        "train_ce_loss",
        "train_mse_loss",
        "train_lambda_mse",
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
    with output_dir.joinpath("metrics_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def summarize_predictions(
    true_indices: list[int],
    pred_indices: list[int],
    *,
    total_loss: float,
    total_rows: int,
) -> dict[str, float]:
    y_true = [CLASS_LABELS[index] for index in true_indices]
    y_pred = [CLASS_LABELS[index] for index in pred_indices]
    metrics = compute_metrics(y_true, y_pred, labels=CLASS_LABELS)
    return {
        "loss": total_loss / max(total_rows, 1),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_macro_f1: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
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
    global_step: int,
    best_val_macro_f1: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        best_val_macro_f1=best_val_macro_f1,
        history=history,
        config=config,
    )
    return path


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = safe_torch_load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def resolve_eval_checkpoint(output_dir: Path, checkpoint: str) -> tuple[Path, str]:
    if checkpoint in {"latest", "best"}:
        return output_dir / f"{checkpoint}.pt", checkpoint
    path = Path(checkpoint)
    return path, path.stem.replace(" ", "_").replace(".", "_")


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
    test_result = predict_loader(
        model,
        test_loader,
        device=device,
        precision=precision,
        max_batches=max_batches,
        log_every=log_every,
        log_prefix=f"test {tag or 'final'}",
    )
    test_4way = serializable_test_result(
        test_result,
        bootstrap=add_bootstrap(test_result, samples=bootstrap_samples, seed=seed),
        epoch=checkpoint_epoch,
        counts=counts,
    )
    output_dir.joinpath(f"test_4way{suffix}.json").write_text(
        json.dumps(test_4way, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    binary = binary_o0_vs_o3_from_4way(test_result)
    binary_bootstrap = add_bootstrap(
        {
            "true": binary["true"],
            "predicted": binary["predicted"],
            "ids": binary["ids"],
        },
        samples=bootstrap_samples,
        seed=seed,
        labels=("O0", "O3"),
        positive_scores=np.asarray(binary["positive_scores"], dtype=float),
    )
    test_binary = {
        "task": binary["task"],
        "class_labels": binary["class_labels"],
        "checkpoint_epoch": checkpoint_epoch,
        "test_metrics": binary["metrics"],
        "bootstrap_95": binary_bootstrap,
        "counts": {
            "test_rows": len(binary["true"]),
            "test_functions": len(set(binary["ids"])),
        },
        "test_predictions": {
            "ids": binary["ids"],
            "true": binary["true"],
            "predicted": binary["predicted"],
            "positive_scores": binary["positive_scores"],
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


def fmt_metric(value: float | None) -> str:
    return "pending" if value is None else f"{value:.3f}"


def write_summary(
    output_dir: Path,
    *,
    config: dict[str, Any],
    latest_4way: dict[str, Any] | None,
    latest_binary: dict[str, Any] | None,
    best_epoch: int | None,
    completed: bool,
) -> None:
    run_root = output_dir.parent
    rows = [
        {
            "model": "Pure opt baseline epoch 1",
            "acc": SCRATCH_OPT_BASELINE_ACC,
            "f1": SCRATCH_OPT_BASELINE_F1,
            "note": "paper equal-budget reference",
        },
        {
            "model": "Frozen BCSD Student probe",
            "acc": 0.559,
            "f1": 0.554,
            "note": "reference from task",
        },
        {
            "model": "Frozen EBM-full probe",
            "acc": 0.588,
            "f1": 0.583,
            "note": "reference from task",
        },
    ]
    scratch_epoch1 = run_root / "scratch_student_opt_cuda0_fresh_5ep" / "test_4way_epoch_001.json"
    if scratch_epoch1.is_file():
        rows[0]["acc"] = read_result_metric(scratch_epoch1, "accuracy")
        rows[0]["f1"] = read_result_metric(scratch_epoch1, "macro_f1")
        rows[0]["note"] = "read from saved scratch epoch-1 result"
    student_probe_acc = read_probe_metric(run_root, "student", "4way", "accuracy")
    student_probe_f1 = read_probe_metric(run_root, "student", "4way", "macro_f1")
    if student_probe_acc is not None:
        rows[1]["acc"] = student_probe_acc
        rows[1]["f1"] = student_probe_f1
    ebm_probe_acc = read_probe_metric(run_root, "ebm-full", "4way", "accuracy")
    ebm_probe_f1 = read_probe_metric(run_root, "ebm-full", "4way", "macro_f1")
    if ebm_probe_acc is not None:
        rows[2]["acc"] = ebm_probe_acc
        rows[2]["f1"] = ebm_probe_f1
    if latest_4way is not None:
        rows.append(
            {
                "model": "Distilled opt student",
                "acc": latest_4way["test_metrics"]["accuracy"],
                "f1": latest_4way["test_metrics"]["macro_f1"],
                "note": f"checkpoint epoch {latest_4way['checkpoint_epoch']}",
            }
        )

    lines = [
        "# Distilled Opt Student",
        "",
        "## Comparison",
        "",
        "| Model | 4-way Acc | 4-way Macro-F1 | Note |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {fmt_metric(row['acc'])} | {fmt_metric(row['f1'])} | {row['note']} |"
        )
    if latest_4way is not None:
        acc = latest_4way["test_metrics"]["accuracy"]
        delta = acc - SCRATCH_OPT_BASELINE_ACC
        verdict = (
            "beats"
            if delta > 0
            else "ties"
            if abs(delta) < 1e-12
            else "does not beat"
        )
        lines.extend(
            [
                "",
                "## Opt-Class Verdict",
                "",
                f"- Distilled opt student {verdict} the equal-budget scratch baseline on 4-way accuracy ({acc:.3f} vs {SCRATCH_OPT_BASELINE_ACC:.3f}, delta {delta:+.3f}).",
            ]
        )
    lines.extend(
        [
            "",
            "## Recipe",
            "",
            f"- Status: {'completed test evaluation' if completed else 'configured / pending full evaluation'}",
            f"- Best validation epoch: {best_epoch if best_epoch is not None else 'pending'}",
            f"- Teacher: frozen EBM-full Nova from `{config['teacher_dir']}`.",
            "- Loss: opt-class cross entropy + scheduled masked token MSE; no BCSD contrastive/InfoNCE term.",
            f"- Lambda schedule: {config['training']['lambda_start']} -> {config['training']['lambda_end']} with quadratic progress.",
            f"- Architecture: StudentDistillationModule(vocab_size={config['student_config']['vocab_size']}, hidden_dim={config['student_config']['hidden_dim']}, num_layers={config['student_config']['num_layers']}) + LatentAttentionLayer(num_latents={config['training']['num_latents']}) + Linear(2048, 4).",
            f"- Parameter counts: backbone={config['parameter_counts']['backbone']:,}, LAL={config['parameter_counts']['latent_attention']:,}, classifier={config['parameter_counts']['classifier']:,}, total={config['parameter_counts']['total']:,}.",
            f"- Data: leakage-safe opt split from `{config['data_cache']['source']}`; assembly only, no retrieval instruction prefix for CE or MSE.",
        ]
    )
    if latest_binary is not None:
        lines.extend(
            [
                "",
                "## O0-vs-O3 From 4-Way Logits",
                "",
                f"- Accuracy: {latest_binary['test_metrics']['accuracy']:.3f}",
                f"- Macro-F1: {latest_binary['test_metrics']['macro_f1']:.3f}",
                f"- AUROC: {latest_binary['test_metrics'].get('roc_auc', float('nan')):.3f}",
            ]
        )
    output_dir.joinpath("SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.epochs < 0:
        raise ValueError("--epochs must be non-negative")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_safe_output_dir(output_dir, args)
    set_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    precision = precision_for_device(device, args.precision)
    model_dtype = dtype_for_precision(device, precision)
    mask_dtype = torch.bfloat16 if device.type == "cuda" and precision == "bf16" else torch.float32
    print(
        f"distilled opt student starting: device={device} precision={precision} "
        f"batch_size={args.batch_size} grad_accum={args.grad_accum_steps}",
        flush=True,
    )

    base_tokenizer, _, _ = setup_nova_tokenizer()
    pad_id = base_tokenizer.pad_token_id or 0
    vocab_size = len(base_tokenizer)
    student_config = json.loads(Path(args.student_config).read_text(encoding="utf-8"))
    model = build_model_from_config(
        student_config,
        pad_id=pad_id,
        num_latents=args.num_latents,
        num_classes=len(CLASS_LABELS),
        normalize_embeddings=args.normalize_embeddings,
    ).to(device=device, dtype=model_dtype)
    parameter_counts = model.parameter_counts().as_dict()
    print(
        f"student ready: params={parameter_counts['total']:,} dtype={model_dtype} "
        f"device={next(model.parameters()).device}",
        flush=True,
    )

    data_cache = resolve_split_data(args, output_dir)
    eval_batch_size = args.eval_batch_size or args.batch_size
    loaders: dict[str, DataLoader] = {}
    if not args.eval_only:
        loaders["train"] = make_loader(
            data_cache["paths"]["train"],
            max_length=args.max_length,
            pad_id=pad_id,
            mask_dtype=mask_dtype,
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
            mask_dtype=mask_dtype,
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
        mask_dtype=mask_dtype,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    print(
        "loaders ready: "
        + " ".join(f"{split}={len(loader)} batches" for split, loader in loaders.items()),
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
        "experiment": "distill_opt_student",
        "teacher_dir": str(Path(args.teacher_dir)),
        "manifest": str(Path(args.manifest)),
        "student_config_path": str(Path(args.student_config)),
        "student_config": student_config,
        "parameter_counts": parameter_counts,
        "device": device_metadata(device),
        "data_cache": data_cache,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": eval_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "max_length": args.max_length,
            "seed": args.seed,
            "precision": precision,
            "num_latents": args.num_latents,
            "normalize_embeddings": args.normalize_embeddings,
            "lambda_start": args.lambda_start,
            "lambda_end": args.lambda_end,
            "ce_weight": args.ce_weight,
            "mse_weight": args.mse_weight,
            "bootstrap_samples": args.bootstrap_samples,
            "test_every_epoch": args.test_every_epoch,
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
            latest_4way=None,
            latest_binary=None,
            best_epoch=None,
            completed=False,
        )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": config}, indent=2, sort_keys=True))
        return

    if args.eval_only:
        checkpoint_path, default_tag = resolve_eval_checkpoint(output_dir, args.eval_checkpoint)
        checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
        epoch = int(checkpoint.get("epoch", -1))
        tag = args.eval_tag or default_tag
        output_dir.joinpath(f"eval_config_{tag}.json").write_text(
            json.dumps(
                {"checkpoint": str(checkpoint_path), "checkpoint_epoch": epoch, "config": config},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
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
            checkpoint_epoch=epoch,
            log_every=args.log_every,
            max_batches=args.max_eval_batches,
            tag=tag,
        )
        return

    print(f"loading frozen teacher from {args.teacher_dir}", flush=True)
    teacher = load_teacher(
        Path(args.teacher_dir),
        device=device,
        dtype=model_dtype,
        vocab_size=vocab_size,
    )
    print(
        f"teacher ready: hidden_dim={teacher.config.hidden_size} "
        f"device={next(teacher.parameters()).device}",
        flush=True,
    )
    if int(teacher.config.hidden_size) != int(student_config["hidden_dim"]):
        raise ValueError(
            "Teacher/student hidden dims do not match: "
            f"{teacher.config.hidden_size} vs {student_config['hidden_dim']}"
        )

    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    global_step = 0
    best_val_macro_f1 = float("-inf")
    best_epoch: int | None = None
    if args.resume and latest_path.is_file():
        checkpoint = load_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", float("-inf")))
        history = list(checkpoint.get("history", []))
        if history:
            best_epoch = max(history, key=lambda row: row.get("val_macro_f1", -1.0))[
                "epoch"
            ]
        print(
            f"resumed from {latest_path}: start_epoch={start_epoch} global_step={global_step}",
            flush=True,
        )

    train_batches_per_epoch = len(loaders["train"])
    if args.max_train_batches is not None:
        train_batches_per_epoch = min(train_batches_per_epoch, args.max_train_batches)
    updates_per_epoch = max(1, math.ceil(train_batches_per_epoch / args.grad_accum_steps))
    total_updates = updates_per_epoch * max(args.epochs, 1)
    criterion = torch.nn.CrossEntropyLoss()

    latest_4way = None
    latest_binary = None
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_ce = 0.0
        train_mse = 0.0
        train_lambda = 0.0
        train_rows = 0
        train_true: list[int] = []
        train_pred: list[int] = []
        processed_batches = 0
        epoch_start = time.monotonic()
        print(f"epoch {epoch}/{args.epochs} train start", flush=True)

        for batch_index, batch in enumerate(loaders["train"], start=1):
            if args.max_train_batches is not None and batch_index > args.max_train_batches:
                break
            processed_batches += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            key_padding_mask = batch["key_padding_mask"].to(device, non_blocking=True)
            pool_mask = batch["pool_mask"].to(device, non_blocking=True)
            nova_mask = batch["nova_attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.no_grad(), maybe_autocast(device, precision):
                teacher_out = teacher(
                    input_ids=input_ids,
                    nova_attention_mask=nova_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                teacher_hidden = teacher_out.hidden_states[-1].detach()
                del teacher_out

            lambda_mse = lambda_mse_for_step(
                global_step,
                total_updates,
                start=args.lambda_start,
                end=args.lambda_end,
            )
            with autocast_context(device, precision):
                logits, student_hidden = model(
                    input_ids,
                    key_padding_mask=key_padding_mask,
                    pool_mask=pool_mask,
                    return_hidden=True,
                )
                ce_loss = criterion(logits.float(), labels)
                mse_loss = masked_mse_loss(
                    student_hidden.float(),
                    teacher_hidden.float(),
                    key_padding_mask,
                )
                loss = args.ce_weight * ce_loss + args.mse_weight * lambda_mse * mse_loss

            (loss / args.grad_accum_steps).backward()
            if batch_index % args.grad_accum_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            rows = int(labels.numel())
            train_rows += rows
            train_loss += float(loss.item()) * rows
            train_ce += float(ce_loss.item()) * rows
            train_mse += float(mse_loss.item()) * rows
            train_lambda += float(lambda_mse) * rows
            train_true.extend(labels.detach().cpu().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
            if args.log_every > 0 and (
                batch_index == 1 or batch_index % args.log_every == 0
            ):
                print(
                    f"epoch {epoch}: batch {batch_index}/{len(loaders['train'])} "
                    f"rows={train_rows} loss={train_loss / max(train_rows, 1):.4f} "
                    f"ce={train_ce / max(train_rows, 1):.4f} "
                    f"mse={train_mse / max(train_rows, 1):.4f} "
                    f"lambda={lambda_mse:.4f}",
                    flush=True,
                )

            del teacher_hidden, student_hidden, logits, ce_loss, mse_loss, loss
            del input_ids, key_padding_mask, pool_mask, nova_mask, labels

        if train_rows and processed_batches % args.grad_accum_steps != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        train_metrics = summarize_predictions(
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
            log_every=args.log_every,
            log_prefix=f"epoch {epoch} val",
        )
        val_metrics = val_result["metrics"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_ce / max(train_rows, 1),
            "train_mse_loss": train_mse / max(train_rows, 1),
            "train_lambda_mse": train_lambda / max(train_rows, 1),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": time.monotonic() - epoch_start,
        }
        history.append(row)

        improved = val_metrics["macro_f1"] > best_val_macro_f1
        if improved:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_val_macro_f1=best_val_macro_f1,
                history=history,
                config=config,
            )
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
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
                global_step=global_step,
                best_val_macro_f1=best_val_macro_f1,
                history=history,
                config=config,
            )
            print(f"saved epoch checkpoint: {epoch_path}", flush=True)
        if args.test_every_epoch:
            latest_4way, latest_binary = run_test_evaluation(
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
                    "test_loss": latest_4way["test_metrics"]["loss"],
                    "test_accuracy": latest_4way["test_metrics"]["accuracy"],
                    "test_balanced_accuracy": latest_4way["test_metrics"][
                        "balanced_accuracy"
                    ],
                    "test_macro_f1": latest_4way["test_metrics"]["macro_f1"],
                    "test_o0_vs_o3_accuracy": latest_binary["test_metrics"]["accuracy"],
                    "test_o0_vs_o3_macro_f1": latest_binary["test_metrics"]["macro_f1"],
                    "test_o0_vs_o3_roc_auc": latest_binary["test_metrics"].get("roc_auc"),
                }
            )
            history[-1] = row
        write_history(output_dir, history)
        write_summary(
            output_dir,
            config=config,
            latest_4way=latest_4way,
            latest_binary=latest_binary,
            best_epoch=best_epoch,
            completed=latest_4way is not None,
        )
        print(
            f"epoch {epoch} done: train_loss={row['train_loss']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} improved={improved}",
            flush=True,
        )

    final_epoch = best_epoch or args.epochs
    if best_path.is_file():
        best_checkpoint = load_checkpoint(best_path, model=model, map_location=device)
        final_epoch = int(best_checkpoint.get("epoch", final_epoch))
    latest_4way, latest_binary = run_test_evaluation(
        model=model,
        test_loader=loaders["test"],
        device=device,
        precision=precision,
        output_dir=output_dir,
        counts=data_cache["counts"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        checkpoint_epoch=final_epoch,
        log_every=args.log_every,
        max_batches=args.max_eval_batches,
        tag=None,
    )
    write_summary(
        output_dir,
        config=config,
        latest_4way=latest_4way,
        latest_binary=latest_binary,
        best_epoch=best_epoch,
        completed=True,
    )
    print("distilled opt student finished", flush=True)


if __name__ == "__main__":
    main()
