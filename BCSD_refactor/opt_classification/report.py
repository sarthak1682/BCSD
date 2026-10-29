#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PAIRED_COMPARISONS = (
    ("clap-raw", "clap", "BCSD specialization"),
    ("student-pre", "student", "BCSD refinement"),
    ("jtrans", "jtrans-bench", "BCSD specialization"),
    (
        "nova-raw-bidirectional-mean",
        "ebm-no-causal-bidirectional-mean",
        "BCSD specialization",
    ),
    (
        "ebm-s1-bidirectional-mean",
        "ebm-full-bidirectional-mean",
        "BCSD specialization",
    ),
    ("nova-raw-native-mean", "nova-raw-bidirectional-mean", "mask conversion"),
    ("ebm-s1-native-mean", "ebm-s1-bidirectional-mean", "mask conversion"),
    ("ebm-full-native-mean", "ebm-full-bidirectional-mean", "mask conversion"),
    (
        "ebm-no-causal-native-mean",
        "ebm-no-causal-bidirectional-mean",
        "mask conversion",
    ),
)


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "test_metrics" not in payload or "task" not in payload:
            continue
        metrics = payload["test_metrics"]
        sanity = payload.get("shuffled_label_sanity", {}).get("test_metrics", {})
        ci = payload.get("bootstrap_95", {}).get("metrics", {})
        rows.append(
            {
                "model": path.parent.name,
                "task": payload["task"],
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "roc_auc": metrics.get("roc_auc"),
                "selected_C": payload.get("selected_C"),
                "shuffled_accuracy": sanity.get("accuracy"),
                "shuffled_macro_f1": sanity.get("macro_f1"),
                "accuracy_ci_lower": ci.get("accuracy", {}).get("lower"),
                "accuracy_ci_upper": ci.get("accuracy", {}).get("upper"),
                "macro_f1_ci_lower": ci.get("macro_f1", {}).get("lower"),
                "macro_f1_ci_upper": ci.get("macro_f1", {}).get("upper"),
                "roc_auc_ci_lower": ci.get("roc_auc", {}).get("lower"),
                "roc_auc_ci_upper": ci.get("roc_auc", {}).get("upper"),
                "source": str(path.resolve()),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No probe result JSON files found under {results_dir}")
    return rows


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _paired_metric(
    true: np.ndarray,
    predicted: np.ndarray,
    labels: list[str],
    metric: str,
    scores: np.ndarray | None,
) -> float:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    if metric == "accuracy":
        return float(accuracy_score(true, predicted))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(true, predicted))
    if metric == "macro_f1":
        return float(f1_score(true, predicted, labels=labels, average="macro"))
    if metric == "roc_auc":
        if scores is None:
            raise ValueError("ROC-AUC comparison requires positive scores")
        return float(roc_auc_score((true == labels[1]).astype(int), scores))
    raise ValueError(f"Unknown paired metric: {metric}")


def _paired_deltas(results_dir: Path, seed: int = 42) -> list[dict[str, Any]]:
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "test_predictions" in payload and "task" in payload:
            payloads[(path.parent.name, payload["task"])] = payload

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for before_name, after_name, comparison_type in PAIRED_COMPARISONS:
        for task in ("4way", "o0-vs-o3"):
            before = payloads.get((before_name, task))
            after = payloads.get((after_name, task))
            if before is None or after is None:
                continue
            before_predictions = before["test_predictions"]
            after_predictions = after["test_predictions"]
            if (
                before_predictions["ids"] != after_predictions["ids"]
                or before_predictions["true"] != after_predictions["true"]
            ):
                raise ValueError(
                    f"Paired results are not row-aligned: {before_name} vs {after_name}"
                )

            ids = np.asarray(before_predictions["ids"], dtype=object)
            true = np.asarray(before_predictions["true"], dtype=object)
            before_pred = np.asarray(before_predictions["predicted"], dtype=object)
            after_pred = np.asarray(after_predictions["predicted"], dtype=object)
            before_scores = before_predictions.get("positive_scores")
            after_scores = after_predictions.get("positive_scores")
            before_scores_array = (
                None if before_scores is None else np.asarray(before_scores, dtype=float)
            )
            after_scores_array = (
                None if after_scores is None else np.asarray(after_scores, dtype=float)
            )
            labels = list(before["class_labels"])
            metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
            if task == "o0-vs-o3":
                metrics.append("roc_auc")

            clusters = np.asarray(sorted(set(ids.tolist())), dtype=object)
            cluster_indices = {
                cluster: np.flatnonzero(ids == cluster) for cluster in clusters
            }
            samples = int(before.get("bootstrap_95", {}).get("samples", 1000))
            for metric in metrics:
                before_value = _paired_metric(
                    true, before_pred, labels, metric, before_scores_array
                )
                after_value = _paired_metric(
                    true, after_pred, labels, metric, after_scores_array
                )
                bootstrap_deltas: list[float] = []
                for _ in range(samples):
                    sampled = rng.choice(clusters, size=len(clusters), replace=True)
                    indices = np.concatenate(
                        [cluster_indices[cluster] for cluster in sampled]
                    )
                    before_sample = _paired_metric(
                        true[indices],
                        before_pred[indices],
                        labels,
                        metric,
                        None
                        if before_scores_array is None
                        else before_scores_array[indices],
                    )
                    after_sample = _paired_metric(
                        true[indices],
                        after_pred[indices],
                        labels,
                        metric,
                        None
                        if after_scores_array is None
                        else after_scores_array[indices],
                    )
                    bootstrap_deltas.append(after_sample - before_sample)
                lower, upper = np.quantile(bootstrap_deltas, [0.025, 0.975])
                rows.append(
                    {
                        "comparison_type": comparison_type,
                        "before": before_name,
                        "after": after_name,
                        "task": task,
                        "metric": metric,
                        "before_value": before_value,
                        "after_value": after_value,
                        "delta_after_minus_before": after_value - before_value,
                        "delta_ci_lower": float(lower),
                        "delta_ci_upper": float(upper),
                        "function_clusters": len(clusters),
                        "bootstrap_samples": samples,
                    }
                )
    return rows


def _write_report(
    rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], output: Path
) -> None:
    lines = [
        "# Optimization-classification probe results",
        "",
        "Frozen linear probes; hyperparameters were selected on the validation split only.",
        "",
        "| Model | Task | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC | Shuffled accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["task"], item["model"])):
        lines.append(
            "| {model} | {task} | {accuracy} | {balanced_accuracy} | "
            "{macro_f1} | {roc_auc} | {shuffled_accuracy} |".format(
                model=row["model"],
                task=row["task"],
                accuracy=_fmt(row["accuracy"]),
                balanced_accuracy=_fmt(row["balanced_accuracy"]),
                macro_f1=_fmt(row["macro_f1"]),
                roc_auc=_fmt(row["roc_auc"]),
                shuffled_accuracy=_fmt(row["shuffled_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## Paired deltas",
            "",
            "| Type | Before → after | Task | Metric | Before | After | Δ | 95% CI |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            "| {comparison_type} | {before} → {after} | {task} | {metric} | "
            "{before_value:.3f} | {after_value:.3f} | "
            "{delta_after_minus_before:+.3f} | [{delta_ci_lower:+.3f}, "
            "{delta_ci_upper:+.3f}] |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpret paired pre/post checkpoint deltas as the primary evidence. "
            "Cross-architecture rankings are descriptive because tokenization, pooling, "
            "capacity, and pretraining differ.",
            "",
            "The relevant BCSD conflict is induced by positive pairs: pulling the same "
            "function across optimization levels together can suppress optimization "
            "features even when such examples are never used as negatives.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(rows: list[dict[str, Any]], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    tasks = ("4way", "o0-vs-o3")
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for axis, task in zip(axes, tasks):
        selected = sorted(
            (row for row in rows if row["task"] == task),
            key=lambda item: item["model"],
        )
        if not selected:
            axis.set_visible(False)
            continue
        metric = "macro_f1" if task == "4way" else "roc_auc"
        names = [row["model"] for row in selected]
        values = [row[metric] for row in selected]
        axis.bar(names, values)
        axis.set_ylim(0, 1)
        axis.set_ylabel(metric.replace("_", " ").upper())
        axis.set_title("O0–O3 four-way" if task == "4way" else "O0 vs O3")
        axis.tick_params(axis="x", labelrotation=45)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def generate_report(results_dir: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_results(results_dir)
    paired_rows = _paired_deltas(results_dir)
    _write_csv(rows, output_dir / "metrics.csv")
    if paired_rows:
        _write_csv(paired_rows, output_dir / "paired_deltas.csv")
    _write_report(rows, paired_rows, output_dir / "report.md")
    _write_plot(rows, output_dir / "probe_summary.png")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = generate_report(args.results_dir, args.output_dir)
    print(f"Wrote aggregate artifacts for {len(rows)} probe results to {args.output_dir}")


if __name__ == "__main__":
    main()
