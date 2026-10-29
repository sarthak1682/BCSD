#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from shared.data_utils import parse_bench_opt


OPTIMIZATIONS = ("O0", "O1", "O2", "O3")
QUARTET_FIELDS = ("id", "compiler", "architecture", "variant")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            missing = {"id", "opt"}.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_number} misses {sorted(missing)}")
            rows.append(row)
    return rows


def iter_id_opt_rows(path: str | Path) -> Iterable[dict[str, str]]:
    """Stream only manifest-relevant fields, never multi-GB assembly payloads."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"id", "opt"}.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_number} misses {sorted(missing)}")
            yield {"id": str(row["id"]), "opt": str(row["opt"])}


def read_key_set(path: str | Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"id", "opt"}.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_number} misses {sorted(missing)}")
            keys.add((str(row["id"]), str(row["opt"])))
    return keys


def binary_group(function_id: str) -> str:
    return function_id.rsplit("__", 1)[0]


def _manifest_quartets(
    source_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    grouped: dict[
        tuple[str, str, str, str], list[tuple[str, str]]
    ] = defaultdict(list)
    total_rows = 0

    for source_row in source_rows:
        total_rows += 1
        function_id = str(source_row["id"])
        opt = str(source_row["opt"])
        parsed = parse_bench_opt(opt)
        key = (
            function_id,
            parsed["compiler"],
            parsed["architecture"],
            parsed["variant"],
        )
        grouped[key].append((parsed["optimization"], opt))

    manifest_rows: list[dict[str, str]] = []
    rejection_counts: Counter[str] = Counter()
    for key in sorted(grouped):
        entries = grouped[key]
        counts = Counter(optimization for optimization, _ in entries)
        observed = set(counts)
        expected = set(OPTIMIZATIONS)
        if observed - expected:
            rejection_counts["extra_optimization"] += 1
            continue
        if observed != expected:
            rejection_counts["incomplete"] += 1
            continue
        if any(counts[optimization] != 1 for optimization in OPTIMIZATIONS):
            rejection_counts["duplicate"] += 1
            continue

        function_id, compiler, architecture, variant = key
        by_optimization = {optimization: opt for optimization, opt in entries}
        for optimization in OPTIMIZATIONS:
            manifest_rows.append(
                {
                    "id": function_id,
                    "opt": by_optimization[optimization],
                    "compiler": compiler,
                    "optimization": optimization,
                    "architecture": architecture,
                    "variant": variant,
                    "binary_group": binary_group(function_id),
                }
            )

    stats = {
        "input_rows": total_rows,
        "candidate_groups": len(grouped),
        "retained_quartets": len(manifest_rows) // len(OPTIMIZATIONS),
        "retained_rows": len(manifest_rows),
        "rejected_incomplete": rejection_counts["incomplete"],
        "rejected_duplicate": rejection_counts["duplicate"],
        "rejected_extra_optimization": rejection_counts["extra_optimization"],
    }
    return manifest_rows, stats


def _align_complete_quartets(
    rows: Sequence[dict[str, str]],
    alignment_key_sets: Sequence[set[tuple[str, str]]],
) -> tuple[list[dict[str, str]], int]:
    if not alignment_key_sets:
        return list(rows), 0

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in QUARTET_FIELDS)
        grouped[key].append(row)

    retained: list[dict[str, str]] = []
    rejected = 0
    for group_rows in grouped.values():
        row_keys = {(row["id"], row["opt"]) for row in group_rows}
        if all(row_keys <= available for available in alignment_key_sets):
            retained.extend(group_rows)
        else:
            rejected += 1
    return retained, rejected


def _limit_binary_groups(
    rows: Sequence[dict[str, str]],
    max_groups: int | None,
    *,
    seed: int,
) -> tuple[list[dict[str, str]], int]:
    if max_groups is not None and max_groups < 1:
        raise ValueError("max_groups must be positive")
    groups = sorted({row["binary_group"] for row in rows})
    if max_groups is None or len(groups) <= max_groups:
        return list(rows), len(groups)

    rng = random.Random(seed)
    rng.shuffle(groups)
    keep = set(groups[:max_groups])
    return [row for row in rows if row["binary_group"] in keep], len(keep)


def _train_val_groups(
    groups: Sequence[str],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")

    ordered = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    if not ordered or val_fraction == 0:
        return set(ordered), set()

    number_val = round(len(ordered) * val_fraction)
    if len(ordered) >= 2:
        number_val = min(max(number_val, 1), len(ordered) - 1)
    else:
        number_val = 0
    val_groups = set(ordered[:number_val])
    return set(ordered[number_val:]), val_groups


def assert_split_disjointness(rows: Sequence[Mapping[str, Any]]) -> None:
    functions: dict[str, set[str]] = defaultdict(set)
    binaries: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        functions[split].add(str(row["id"]))
        binaries[split].add(str(row["binary_group"]))

    for left, right in combinations(sorted(functions), 2):
        function_overlap = functions[left] & functions[right]
        if function_overlap:
            example = sorted(function_overlap)[0]
            raise AssertionError(
                f"Function leakage between {left} and {right}: {example}"
            )
        binary_overlap = binaries[left] & binaries[right]
        if binary_overlap:
            example = sorted(binary_overlap)[0]
            raise AssertionError(
                f"Binary leakage between {left} and {right}: {example}"
            )


def _split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        result[split] = {
            "rows": len(selected),
            "functions": len({row["id"] for row in selected}),
            "binary_groups": len({row["binary_group"] for row in selected}),
            "optimization_counts": dict(
                sorted(Counter(row["optimization"] for row in selected).items())
            ),
        }
    return result


def build_manifest_rows(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
    max_groups: int | None = None,
    alignment_key_sets: Sequence[set[tuple[str, str]]] = (),
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Construct manifest rows and a summary.

    ``max_groups`` is a smoke-test cap applied independently to the official
    train and test sources before splitting the retained train binary groups.
    The official test rows are never reassigned to train or validation.
    """
    train_manifest, train_stats = _manifest_quartets(train_rows)
    test_manifest, test_stats = _manifest_quartets(test_rows)
    train_manifest, train_alignment_rejected = _align_complete_quartets(
        train_manifest, alignment_key_sets
    )
    test_manifest, test_alignment_rejected = _align_complete_quartets(
        test_manifest, alignment_key_sets
    )
    train_manifest, train_group_count = _limit_binary_groups(
        train_manifest, max_groups, seed=seed
    )
    test_manifest, test_group_count = _limit_binary_groups(
        test_manifest, max_groups, seed=seed + 1
    )

    all_train_groups = sorted({row["binary_group"] for row in train_manifest})
    train_groups, val_groups = _train_val_groups(
        all_train_groups, val_fraction=val_fraction, seed=seed
    )

    output: list[dict[str, str]] = []
    for row in train_manifest:
        output.append(
            {
                "split": "val" if row["binary_group"] in val_groups else "train",
                **row,
            }
        )
    output.extend({"split": "test", **row} for row in test_manifest)
    split_rank = {"train": 0, "val": 1, "test": 2}
    output.sort(
        key=lambda row: (
            split_rank[row["split"]],
            row["binary_group"],
            row["id"],
            row["compiler"],
            row["architecture"],
            row["variant"],
            OPTIMIZATIONS.index(row["optimization"]),
        )
    )

    assert_split_disjointness(output)
    summary: dict[str, Any] = {
        "seed": seed,
        "val_fraction": val_fraction,
        "max_groups_per_source": max_groups,
        "sources": {
            "train": {
                **train_stats,
                "rejected_cross_format_alignment": train_alignment_rejected,
                "selected_binary_groups": train_group_count,
            },
            "test": {
                **test_stats,
                "rejected_cross_format_alignment": test_alignment_rejected,
                "selected_binary_groups": test_group_count,
            },
        },
        "splits": _split_summary(output),
        "disjointness": {
            "function_ids": True,
            "binary_groups": True,
        },
    }
    return output, summary


def build_manifest(
    train_path: str | Path,
    test_path: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
    max_groups: int | None = None,
    alignment_paths: Sequence[str | Path] = (),
    alignment_path_groups: Sequence[Sequence[str | Path]] = (),
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    alignment_key_sets = [read_key_set(path) for path in alignment_paths]
    for group in alignment_path_groups:
        combined_keys: set[tuple[str, str]] = set()
        for path in group:
            combined_keys.update(read_key_set(path))
        alignment_key_sets.append(combined_keys)
    rows, summary = build_manifest_rows(
        iter_id_opt_rows(train_path),
        iter_id_opt_rows(test_path),
        val_fraction=val_fraction,
        seed=seed,
        max_groups=max_groups,
        alignment_key_sets=alignment_key_sets,
    )
    summary["input_paths"] = {
        "train": str(Path(train_path).resolve()),
        "test": str(Path(test_path).resolve()),
    }
    summary["alignment_paths"] = [
        str(Path(path).resolve()) for path in alignment_paths
    ]
    summary["alignment_path_groups"] = [
        [str(Path(path).resolve()) for path in group]
        for group in alignment_path_groups
    ]

    manifest_path = Path(manifest_path)
    summary_path = Path(summary_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Nova bench train JSONL")
    parser.add_argument("--test", required=True, help="Nova bench official test JSONL")
    parser.add_argument("--manifest", required=True, help="Output manifest JSONL")
    parser.add_argument("--summary", required=True, help="Output summary JSON")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--align-jsonl",
        action="append",
        default=[],
        help="Additional model-format JSONL whose (id,opt) keys must cover each quartet.",
    )
    parser.add_argument(
        "--align-pair",
        action="append",
        nargs=2,
        default=[],
        metavar=("TRAIN_JSONL", "TEST_JSONL"),
        help="Train/test files for one model format; their key sets are unioned.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Smoke cap per source on retained binary groups",
    )
    args = parser.parse_args()
    _, summary = build_manifest(
        args.train,
        args.test,
        args.manifest,
        args.summary,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_groups=args.max_groups,
        alignment_paths=args.align_jsonl,
        alignment_path_groups=args.align_pair,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
