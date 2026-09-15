#!/usr/bin/env python3
"""Audit structural suitability of BinaryCorp data for scratch joint training."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

LABEL_TOKEN_RE = re.compile(r"<label-\d+>")


def audit(path: Path) -> tuple[dict, set[str], set[str]]:
    function_opts: dict[str, set[str]] = collections.defaultdict(set)
    function_counts: collections.Counter[str] = collections.Counter()
    opt_counts: collections.Counter[str] = collections.Counter()
    character_lengths = []
    instruction_lengths = []
    asm_field_types: collections.Counter[str] = collections.Counter()
    label_token_rows = 0
    label_token_count = 0
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            missing = {"id", "opt", "asm"}.difference(record)
            if missing:
                raise ValueError(f"{path}:{line_number} misses {sorted(missing)}")
            function_id = str(record["id"])
            opt = str(record["opt"])
            assembly = record["asm"]
            if isinstance(assembly, list):
                asm_field_types["list"] += 1
                assembly_text = "\n".join(str(item) for item in assembly)
                instruction_lengths.append(len(assembly))
            else:
                asm_field_types[type(assembly).__name__] += 1
                assembly_text = str(assembly)
                instruction_lengths.append(len(assembly_text.splitlines()))
            label_hits = len(LABEL_TOKEN_RE.findall(assembly_text))
            if label_hits:
                label_token_rows += 1
                label_token_count += label_hits
            function_opts[function_id].add(opt)
            function_counts[function_id] += 1
            opt_counts[opt] += 1
            character_lengths.append(len(assembly_text))
            rows += 1

    variants = np.asarray(list(function_counts.values()))
    lengths = np.asarray(character_lengths)
    instructions = np.asarray(instruction_lengths)
    usable_functions = sum(count >= 2 for count in function_counts.values())
    usable_pairs = sum(count for count in function_counts.values() if count >= 2)
    binaries = {function_id.rsplit("__", 1)[0] for function_id in function_opts}
    report = {
        "path": str(path.resolve()),
        "rows": rows,
        "functions": len(function_opts),
        "binary_groups": len(binaries),
        "usable_functions": int(usable_functions),
        "usable_variant_pairs": int(usable_pairs),
        "asm_field_types": dict(sorted(asm_field_types.items())),
        "nova_label_token_rows": int(label_token_rows),
        "nova_label_token_count": int(label_token_count),
        "opt_counts": dict(sorted(opt_counts.items())),
        "variants_per_function": {
            "min": int(variants.min()) if len(variants) else 0,
            "median": float(np.median(variants)) if len(variants) else 0.0,
            "max": int(variants.max()) if len(variants) else 0,
        },
        "assembly_characters": {
            "p50": int(np.percentile(lengths, 50)) if len(lengths) else 0,
            "p95": int(np.percentile(lengths, 95)) if len(lengths) else 0,
            "p99": int(np.percentile(lengths, 99)) if len(lengths) else 0,
        },
        "assembly_instructions": {
            "p50": int(np.percentile(instructions, 50)) if len(instructions) else 0,
            "p95": int(np.percentile(instructions, 95)) if len(instructions) else 0,
            "p99": int(np.percentile(instructions, 99)) if len(instructions) else 0,
        },
    }
    return report, set(function_opts), binaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train")
    parser.add_argument("--test", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    train_report, train_ids, train_binaries = audit(Path(args.train))
    report = {"train": train_report}
    if args.test:
        test_report, test_ids, test_binaries = audit(Path(args.test))
        report["test"] = test_report
        report["train_test_function_overlap"] = len(train_ids.intersection(test_ids))
        report["train_test_binary_overlap"] = len(
            train_binaries.intersection(test_binaries)
        )

    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
