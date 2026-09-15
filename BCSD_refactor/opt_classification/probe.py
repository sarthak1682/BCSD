#!/usr/bin/env python3
"""Run deterministic linear probes on cached, frozen function embeddings.

Embedding caches use this deliberately small ``torch.save`` schema::

    {
        "ids": list[str],            # length N
        "opts": list[str],           # length N, full bench opt labels
        "embeddings": Tensor[N, D],  # finite floating-point values
        "metadata": dict,            # extraction/model provenance
    }

Extraction is intentionally outside this package. Cache rows are joined to the
manifest by the exact ``(id, opt)`` pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from .build_manifest import OPTIMIZATIONS, assert_split_disjointness, read_jsonl


TASK_LABELS = {
    "4way": OPTIMIZATIONS,
    "o0-vs-o3": ("O0", "O3"),
}
SELECTION_METRICS = {"accuracy", "balanced_accuracy", "macro_f1"}


def _normalize_task(task: str) -> str:
    aliases = {
        "4-way": "4way",
        "four-way": "4way",
        "o0_vs_o3": "o0-vs-o3",
        "binary": "o0-vs-o3",
    }
    normalized = aliases.get(task.lower(), task.lower())
    if normalized not in TASK_LABELS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASK_LABELS)}")
    return normalized


def load_embedding_cache(path: str | Path) -> dict[str, Any]:
    """Load and validate the documented torch-save embedding cache schema."""
    path = Path(path)
    try:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        cache = torch.load(path, map_location="cpu")

    if not isinstance(cache, Mapping):
        raise ValueError("Embedding cache must contain a mapping")
    required = {"ids", "opts", "embeddings", "metadata"}
    missing = required.difference(cache)
    if missing:
        raise ValueError(f"Embedding cache misses {sorted(missing)}")

    ids = list(cache["ids"])
    opts = list(cache["opts"])
    embeddings = cache["embeddings"]
    metadata = cache["metadata"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("cache['embeddings'] must be a rank-2 torch Tensor")
    if len(ids) != len(opts) or len(ids) != embeddings.shape[0]:
        raise ValueError("ids, opts, and embeddings must have the same length")
    if not isinstance(metadata, Mapping):
        raise ValueError("cache['metadata'] must be a mapping")
    if not torch.is_floating_point(embeddings):
        raise ValueError("cache['embeddings'] must be floating point")
    if not torch.isfinite(embeddings).all():
        raise ValueError("cache['embeddings'] contains non-finite values")

    return {
        "ids": [str(value) for value in ids],
        "opts": [str(value) for value in opts],
        "embeddings": embeddings.detach().cpu().float(),
        "metadata": dict(metadata),
    }


def align_embeddings_to_manifest(
    manifest_rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    *,
    task: str = "4way",
) -> dict[str, Any]:
    """Align cache embeddings to manifest order for the requested task."""
    task = _normalize_task(task)
    labels = TASK_LABELS[task]
    assert_split_disjointness(manifest_rows)

    ids = list(cache["ids"])
    opts = list(cache["opts"])
    embeddings = cache["embeddings"]
    if isinstance(embeddings, torch.Tensor):
        embedding_array = embeddings.detach().cpu().float().numpy()
    else:
        embedding_array = np.asarray(embeddings, dtype=np.float32)
    if embedding_array.ndim != 2 or len(ids) != len(opts) or len(ids) != len(
        embedding_array
    ):
        raise ValueError("Invalid cache dimensions")

    cache_index: dict[tuple[str, str], int] = {}
    for index, (function_id, opt) in enumerate(zip(ids, opts)):
        key = (str(function_id), str(opt))
        if key in cache_index:
            raise ValueError(f"Duplicate embedding cache key: {key}")
        cache_index[key] = index

    selected_rows = [
        row for row in manifest_rows if str(row["optimization"]) in labels
    ]
    if not selected_rows:
        raise ValueError(f"Manifest has no rows for task {task}")

    indices = []
    for row in selected_rows:
        key = (str(row["id"]), str(row["opt"]))
        if key not in cache_index:
            raise ValueError(f"Embedding cache is missing manifest key: {key}")
        indices.append(cache_index[key])

    return {
        "embeddings": np.asarray(embedding_array[indices], dtype=np.float64),
        "labels": np.asarray(
            [str(row["optimization"]) for row in selected_rows], dtype=object
        ),
        "splits": np.asarray([str(row["split"]) for row in selected_rows], dtype=object),
        "ids": np.asarray([str(row["id"]) for row in selected_rows], dtype=object),
        "binary_groups": np.asarray(
            [str(row["binary_group"]) for row in selected_rows], dtype=object
        ),
        "rows": selected_rows,
        "task": task,
        "class_labels": labels,
    }


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str],
    positive_scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute the fixed classification metric suite."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    recalls = recall_score(
        y_true_array,
        y_pred_array,
        labels=list(labels),
        average=None,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, y_pred_array)
        ),
        "macro_f1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": {
            label: float(value) for label, value in zip(labels, recalls)
        },
        "confusion_matrix": confusion_matrix(
            y_true_array, y_pred_array, labels=list(labels)
        ).tolist(),
    }
    if len(labels) == 2 and positive_scores is not None:
        binary_true = (y_true_array == labels[1]).astype(np.int64)
        result["roc_auc"] = float(roc_auc_score(binary_true, positive_scores))
    return result


def function_clustered_bootstrap(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    function_ids: Sequence[str],
    *,
    labels: Sequence[str],
    positive_scores: Sequence[float] | None = None,
    samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap metric CIs by resampling whole function-id clusters."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    function_array = np.asarray(function_ids)
    score_array = (
        None if positive_scores is None else np.asarray(positive_scores, dtype=float)
    )
    if not (
        len(y_true_array) == len(y_pred_array) == len(function_array)
        and (score_array is None or len(score_array) == len(y_true_array))
    ):
        raise ValueError("Bootstrap inputs must have equal lengths")

    unique_functions = np.asarray(sorted(set(function_array.tolist())), dtype=object)
    if not len(unique_functions):
        raise ValueError("Cannot bootstrap an empty evaluation set")
    cluster_indices = {
        function_id: np.flatnonzero(function_array == function_id)
        for function_id in unique_functions
    }
    scalar_samples: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
    }
    recall_samples = {label: [] for label in labels}
    if len(labels) == 2 and score_array is not None:
        scalar_samples["roc_auc"] = []

    rng = np.random.default_rng(seed)
    for _ in range(samples):
        sampled_functions = rng.choice(
            unique_functions, size=len(unique_functions), replace=True
        )
        indices = np.concatenate(
            [cluster_indices[function_id] for function_id in sampled_functions]
        )
        metrics = compute_metrics(
            y_true_array[indices],
            y_pred_array[indices],
            labels=labels,
            positive_scores=None if score_array is None else score_array[indices],
        )
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            scalar_samples[metric].append(metrics[metric])
        for label in labels:
            recall_samples[label].append(metrics["per_class_recall"][label])
        if "roc_auc" in scalar_samples and "roc_auc" in metrics:
            scalar_samples["roc_auc"].append(metrics["roc_auc"])

    alpha = (1 - confidence) / 2

    def interval(values: Sequence[float]) -> dict[str, float]:
        lower, upper = np.quantile(values, [alpha, 1 - alpha])
        return {"lower": float(lower), "upper": float(upper)}

    return {
        "method": "function_clustered_percentile",
        "confidence": confidence,
        "samples": samples,
        "clusters": len(unique_functions),
        "metrics": {
            metric: interval(values) for metric, values in scalar_samples.items()
        },
        "per_class_recall": {
            label: interval(values) for label, values in recall_samples.items()
        },
    }


def _positive_scores(
    model: LogisticRegression, embeddings: np.ndarray, labels: Sequence[str]
) -> np.ndarray | None:
    if len(labels) != 2:
        return None
    positive_index = list(model.classes_).index(labels[1])
    return model.predict_proba(embeddings)[:, positive_index]


def _fit_and_select(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    labels: Sequence[str],
    c_grid: Sequence[float],
    selection_metric: str,
    seed: int,
    max_iter: int,
    shuffle_labels: bool,
) -> tuple[LogisticRegression, float, list[dict[str, float]], dict[str, Any]]:
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {sorted(SELECTION_METRICS)}"
        )
    c_values = sorted({float(value) for value in c_grid})
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("c_grid must contain positive values")

    rng = np.random.default_rng(seed)
    candidate_y = rng.permutation(train_y) if shuffle_labels else train_y
    candidates: list[
        tuple[float, float, LogisticRegression, dict[str, Any]]
    ] = []
    selection_trace: list[dict[str, float]] = []
    for c_value in c_values:
        model = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            random_state=seed,
        )
        model.fit(train_x, candidate_y)
        val_pred = model.predict(val_x)
        val_scores = _positive_scores(model, val_x, labels)
        val_metrics = compute_metrics(
            val_y, val_pred, labels=labels, positive_scores=val_scores
        )
        score = float(val_metrics[selection_metric])
        selection_trace.append({"C": c_value, selection_metric: score})
        candidates.append((score, c_value, model, val_metrics))

    # A smaller C wins exact ties; only validation scores influence selection.
    _, selected_c, _, selection_val_metrics = max(
        candidates, key=lambda item: (item[0], -item[1])
    )
    final_x = np.concatenate([train_x, val_x], axis=0)
    final_y = np.concatenate([train_y, val_y], axis=0)
    if shuffle_labels:
        final_y = rng.permutation(final_y)
    final_model = LogisticRegression(
        C=selected_c,
        max_iter=max_iter,
        random_state=seed,
    )
    final_model.fit(final_x, final_y)
    return final_model, selected_c, selection_trace, selection_val_metrics


def run_probe(
    manifest_rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    *,
    task: str = "4way",
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    selection_metric: str = "macro_f1",
    seed: int = 42,
    max_iter: int = 1000,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    """Select C on validation only, refit on train+val, and evaluate test."""
    aligned = align_embeddings_to_manifest(manifest_rows, cache, task=task)
    labels = aligned["class_labels"]
    split_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        mask = aligned["splits"] == split
        if not mask.any():
            raise ValueError(f"Manifest has no aligned {split} rows")
        split_arrays[split] = (
            aligned["embeddings"][mask],
            aligned["labels"][mask],
            aligned["ids"][mask],
        )
        observed = set(split_arrays[split][1].tolist())
        if observed != set(labels):
            raise ValueError(
                f"{split} split classes {sorted(observed)} do not match {list(labels)}"
            )

    train_x, train_y, _ = split_arrays["train"]
    val_x, val_y, _ = split_arrays["val"]
    test_x, test_y, test_ids = split_arrays["test"]
    model, selected_c, selection_trace, validation_metrics = _fit_and_select(
        train_x,
        train_y,
        val_x,
        val_y,
        labels=labels,
        c_grid=c_grid,
        selection_metric=selection_metric,
        seed=seed,
        max_iter=max_iter,
        shuffle_labels=False,
    )
    test_pred = model.predict(test_x)
    test_scores = _positive_scores(model, test_x, labels)
    test_metrics = compute_metrics(
        test_y, test_pred, labels=labels, positive_scores=test_scores
    )
    bootstrap = function_clustered_bootstrap(
        test_y,
        test_pred,
        test_ids,
        labels=labels,
        positive_scores=test_scores,
        samples=bootstrap_samples,
        seed=seed,
    )

    shuffled_model, shuffled_c, shuffled_trace, shuffled_validation = _fit_and_select(
        train_x,
        train_y,
        val_x,
        val_y,
        labels=labels,
        c_grid=c_grid,
        selection_metric=selection_metric,
        seed=seed + 1,
        max_iter=max_iter,
        shuffle_labels=True,
    )
    shuffled_pred = shuffled_model.predict(test_x)
    shuffled_scores = _positive_scores(shuffled_model, test_x, labels)
    shuffled_test_metrics = compute_metrics(
        test_y,
        shuffled_pred,
        labels=labels,
        positive_scores=shuffled_scores,
    )

    return {
        "task": aligned["task"],
        "class_labels": list(labels),
        "seed": seed,
        "selection_metric": selection_metric,
        "c_grid": sorted({float(value) for value in c_grid}),
        "selected_C": selected_c,
        "selection_trace": selection_trace,
        "validation_metrics_at_selection": validation_metrics,
        "test_metrics": test_metrics,
        "bootstrap_95": bootstrap,
        "test_predictions": {
            "ids": test_ids.tolist(),
            "true": test_y.tolist(),
            "predicted": test_pred.tolist(),
            "positive_scores": (
                None if test_scores is None else test_scores.astype(float).tolist()
            ),
        },
        "shuffled_label_sanity": {
            "seed": seed + 1,
            "selected_C": shuffled_c,
            "selection_trace": shuffled_trace,
            "validation_metrics_at_selection": shuffled_validation,
            "test_metrics": shuffled_test_metrics,
        },
        "counts": {
            split: {
                "rows": len(values[1]),
                "functions": len(set(values[2].tolist())),
            }
            for split, values in split_arrays.items()
        },
    }


def run_probe_files(
    manifest_path: str | Path,
    embeddings_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load manifest/cache files and run the reusable probe."""
    return run_probe(
        read_jsonl(manifest_path),
        load_embedding_cache(embeddings_path),
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=sorted(TASK_LABELS), default="4way")
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument(
        "--selection-metric",
        choices=sorted(SELECTION_METRICS),
        default="macro_f1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    result = run_probe_files(
        args.manifest,
        args.embeddings,
        task=args.task,
        c_grid=args.c_grid,
        selection_metric=args.selection_metric,
        seed=args.seed,
        max_iter=args.max_iter,
        bootstrap_samples=args.bootstrap_samples,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
