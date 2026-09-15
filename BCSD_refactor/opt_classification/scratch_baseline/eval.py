"""Evaluation helpers for the scratch student baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from opt_classification.probe import compute_metrics, function_clustered_bootstrap
from .data import CLASS_LABELS


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    max_batches: int | None = None,
    log_every: int = 0,
    log_prefix: str = "eval",
) -> dict[str, Any]:
    model.eval()
    all_ids: list[str] = []
    all_true: list[str] = []
    all_pred: list[str] = []
    all_logits: list[np.ndarray] = []
    total_loss = 0.0
    total_rows = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        key_padding_mask = batch["key_padding_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            logits = model(input_ids, key_padding_mask=key_padding_mask)
            loss = criterion(logits.float(), labels)
        predictions = logits.argmax(dim=1).detach().cpu().tolist()
        all_ids.extend(batch["ids"])
        all_true.extend(batch["label_names"])
        all_pred.extend(CLASS_LABELS[index] for index in predictions)
        all_logits.append(logits.detach().float().cpu().numpy())
        total_loss += float(loss.item())
        total_rows += int(labels.numel())
        if log_every > 0 and (
            batch_index == 1 or batch_index % log_every == 0
        ):
            print(
                f"{log_prefix}: batch {batch_index}/{len(loader)} "
                f"rows={total_rows} loss={total_loss / total_rows:.4f}",
                flush=True,
            )

    if total_rows == 0:
        raise ValueError("Evaluation loader produced no rows")
    logits_array = np.concatenate(all_logits, axis=0)
    probabilities = torch.softmax(torch.from_numpy(logits_array), dim=1).numpy()
    metrics = compute_metrics(all_true, all_pred, labels=CLASS_LABELS)
    metrics["loss"] = total_loss / total_rows
    return {
        "metrics": metrics,
        "ids": all_ids,
        "true": all_true,
        "predicted": all_pred,
        "logits": logits_array,
        "probabilities": probabilities,
    }


def add_bootstrap(
    result: dict[str, Any],
    *,
    samples: int,
    seed: int,
    labels: tuple[str, ...] = CLASS_LABELS,
    positive_scores: np.ndarray | None = None,
) -> dict[str, Any] | None:
    if samples <= 0:
        return None
    return function_clustered_bootstrap(
        result["true"],
        result["predicted"],
        result["ids"],
        labels=labels,
        positive_scores=positive_scores,
        samples=samples,
        seed=seed,
    )


def binary_o0_vs_o3_from_4way(result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate O0-vs-O3 by restricting a trained 4-way model's logits."""

    true = np.asarray(result["true"], dtype=object)
    ids = np.asarray(result["ids"], dtype=object)
    logits = np.asarray(result["logits"], dtype=np.float64)
    mask = np.isin(true, ["O0", "O3"])
    if not mask.any():
        raise ValueError("No O0/O3 rows available for binary evaluation")

    o0_index = CLASS_LABELS.index("O0")
    o3_index = CLASS_LABELS.index("O3")
    binary_logits = logits[mask][:, [o0_index, o3_index]]
    binary_probabilities = torch.softmax(torch.from_numpy(binary_logits), dim=1).numpy()
    pred = np.where(binary_logits[:, 1] > binary_logits[:, 0], "O3", "O0")
    true_binary = true[mask]
    metrics = compute_metrics(
        true_binary,
        pred,
        labels=("O0", "O3"),
        positive_scores=binary_probabilities[:, 1],
    )
    return {
        "task": "o0-vs-o3-from-4way-logits",
        "class_labels": ["O0", "O3"],
        "metrics": metrics,
        "ids": ids[mask].tolist(),
        "true": true_binary.tolist(),
        "predicted": pred.tolist(),
        "positive_scores": binary_probabilities[:, 1].astype(float).tolist(),
    }


def serializable_test_result(
    result: dict[str, Any],
    *,
    bootstrap: dict[str, Any] | None,
    epoch: int,
    counts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": "4way",
        "class_labels": list(CLASS_LABELS),
        "checkpoint_epoch": epoch,
        "test_metrics": result["metrics"],
        "bootstrap_95": bootstrap,
        "counts": counts,
        "test_predictions": {
            "ids": result["ids"],
            "true": result["true"],
            "predicted": result["predicted"],
            "probabilities": result["probabilities"].astype(float).tolist(),
        },
    }
