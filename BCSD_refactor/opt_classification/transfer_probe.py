from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .build_manifest import read_jsonl
from .probe import (
    TASK_LABELS,
    _fit_and_select,
    _normalize_task,
    _positive_scores,
    align_embeddings_to_manifest,
    compute_metrics,
    function_clustered_bootstrap,
    load_embedding_cache,
)

DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_SELECTION_METRIC = "macro_f1"


def fit_bench_probe(
    bench_manifest_rows: Sequence[Mapping[str, Any]],
    bench_cache: Mapping[str, Any],
    *,
    task: str = "4way",
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    seed: int = 42,
    max_iter: int = 1000,
):
    task = _normalize_task(task)
    labels = TASK_LABELS[task]
    aligned = align_embeddings_to_manifest(bench_manifest_rows, bench_cache, task=task)
    train_mask = aligned["splits"] == "train"
    val_mask = aligned["splits"] == "val"
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Bench manifest lacks train and/or val splits")
    train_x = aligned["embeddings"][train_mask]
    train_y = aligned["labels"][train_mask]
    val_x = aligned["embeddings"][val_mask]
    val_y = aligned["labels"][val_mask]
    model, selected_c, selection_trace, val_metrics = _fit_and_select(
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
    return {
        "model": model,
        "labels": labels,
        "selected_c": selected_c,
        "selection_trace": selection_trace,
        "val_metrics": val_metrics,
        "selection_metric": selection_metric,
    }


def apply_probe(
    fitted: Mapping[str, Any],
    unseen_manifest_rows: Sequence[Mapping[str, Any]],
    unseen_cache: Mapping[str, Any],
    *,
    task: str,
    seed: int = 42,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    task = _normalize_task(task)
    labels = fitted["labels"]
    model = fitted["model"]
    aligned = align_embeddings_to_manifest(unseen_manifest_rows, unseen_cache, task=task)
    x = aligned["embeddings"]
    y = aligned["labels"]
    ids = aligned["ids"]
    pred = model.predict(x)
    scores = _positive_scores(model, x, labels)
    test_metrics = compute_metrics(y, pred, labels=labels, positive_scores=scores)
    bootstrap = function_clustered_bootstrap(
        y,
        pred,
        ids,
        labels=labels,
        positive_scores=scores,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "task": task,
        "class_labels": list(labels),
        "seed": seed,
        "selection_metric": fitted["selection_metric"],
        "selected_C": fitted["selected_c"],
        "selection_trace": fitted["selection_trace"],
        "bench_validation_metrics_at_selection": fitted["val_metrics"],
        "unseen_test_metrics": test_metrics,
        "bootstrap_95": bootstrap,
        "unseen_counts": {
            "rows": int(len(ids)),
            "functions": int(len(set(ids.tolist()))),
        },
        "unseen_predictions": {
            "ids": ids.tolist(),
            "true": y.tolist(),
            "predicted": pred.tolist(),
            "positive_scores": (
                None if scores is None else scores.astype(float).tolist()
            ),
        },
    }


def run_transfer_probe(
    bench_manifest_rows: Sequence[Mapping[str, Any]],
    bench_cache: Mapping[str, Any],
    unseen_manifest_rows: Sequence[Mapping[str, Any]],
    unseen_cache: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    fitted = fit_bench_probe(bench_manifest_rows, bench_cache, **kwargs)
    return apply_probe(fitted, unseen_manifest_rows, unseen_cache, **kwargs)


def run_transfer_probe_files(
    bench_manifest_path: str | Path,
    bench_embeddings_path: str | Path,
    unseen_manifest_path: str | Path,
    unseen_embeddings_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return run_transfer_probe(
        read_jsonl(bench_manifest_path),
        load_embedding_cache(bench_embeddings_path),
        read_jsonl(unseen_manifest_path),
        load_embedding_cache(unseen_embeddings_path),
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-manifest", required=True)
    parser.add_argument("--bench-embeddings", required=True)
    parser.add_argument("--unseen-manifest", required=True)
    parser.add_argument("--unseen-embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=sorted(TASK_LABELS), default="4way")
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    parser.add_argument("--selection-metric", default=DEFAULT_SELECTION_METRIC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    result = run_transfer_probe_files(
        args.bench_manifest,
        args.bench_embeddings,
        args.unseen_manifest,
        args.unseen_embeddings,
        task=args.task,
        c_grid=args.c_grid,
        selection_metric=args.selection_metric,
        seed=args.seed,
        max_iter=args.max_iter,
        bootstrap_samples=args.bootstrap_samples,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
