import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

sys.path.insert(0, os.path.abspath("./PalmTree"))
from palmtree_binary_utils import (  # type: ignore
    DEFAULT_EXCLUDED_FUNCTIONS,
    _build_symbol_map,
    _extract_function_record,
    _load_angr,
    _should_keep_function,
)

from function_corpus_adapters import VariantBinary, create_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare PalmTree JSONL data with stable cross-variant IDs"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["binarycorp-bin3m", "benchset"],
        required=True,
        help="Which raw binary layout to read",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-instructions", type=int, default=1)
    parser.add_argument("--limit-binaries", type=int, default=None)
    parser.add_argument(
        "--include-runtime-functions",
        action="store_true",
        help="Keep runtime/helper functions instead of filtering them out",
    )
    parser.add_argument(
        "--extra-excluded-function",
        action="append",
        default=[],
        help="Function name to exclude. Can be provided multiple times.",
    )
    return parser.parse_args()


def should_keep_record(
    record: Dict[str, object],
    min_instructions: int,
) -> bool:
    num_instructions = int(record.get("num_instructions", 0))
    return num_instructions >= min_instructions


def normalize_record(
    record: Dict[str, object],
    sample: VariantBinary,
) -> Dict[str, object]:
    function_name = str(record["function"])
    record["id"] = f"{sample.binary_key}::{function_name}"
    record["opt"] = sample.variant_label
    record["opt_level"] = sample.opt_level
    record["variant"] = sample.variant_label
    record["source"] = sample.relative_path
    record["binary"] = sample.relative_path
    record["binary_key"] = sample.binary_key
    record["binary_family"] = sample.binary_family
    record["project"] = sample.project
    record["split"] = sample.split
    return record


def build_excluded_functions(
    include_runtime_functions: bool,
    extra_excluded_functions: Iterable[str],
) -> Set[str]:
    excluded = set(extra_excluded_functions)
    if not include_runtime_functions:
        excluded |= DEFAULT_EXCLUDED_FUNCTIONS
    return excluded


def main() -> None:
    args = parse_args()
    angr = _load_angr()
    adapter = create_adapter(args.dataset_type, args.dataset_root)
    excluded_functions = build_excluded_functions(
        include_runtime_functions=args.include_runtime_functions,
        extra_excluded_functions=args.extra_excluded_function,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "scanned_binaries": 0,
        "valid_binaries": 0,
        "emitted_functions": 0,
    }

    with args.output.open("w", encoding="utf-8") as fh:
        for sample_index, sample in enumerate(adapter.iter_samples(args.split), start=1):
            if args.limit_binaries is not None and sample_index > args.limit_binaries:
                break

            stats["scanned_binaries"] += 1
            try:
                project = angr.Project(
                    str(sample.binary_path),
                    load_options={"auto_load_libs": False},
                )
                cfg = project.analyses.CFGFast(normalize=True, data_references=True)
            except Exception as exc:
                print(f"Skipping {sample.binary_path}: {exc}")
                continue

            stats["valid_binaries"] += 1
            symbol_map = _build_symbol_map(project)

            for func in cfg.kb.functions.values():
                if not _should_keep_function(func, excluded_functions):
                    continue

                record = _extract_function_record(
                    project=project,
                    binary_path=sample.binary_path,
                    input_root=args.dataset_root,
                    func=func,
                    symbol_map=symbol_map,
                )
                if record is None or not should_keep_record(record, args.min_instructions):
                    continue

                record = normalize_record(record, sample)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["emitted_functions"] += 1

    print(
        f"Wrote {stats['emitted_functions']} functions from "
        f"{stats['valid_binaries']}/{stats['scanned_binaries']} binaries to {args.output}"
    )


if __name__ == "__main__":
    main()
