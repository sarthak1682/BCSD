"""Generic data utilities shared across all scripts.

Deduplicated from: run_nova_ebm_stages.py, run_nova_ebm_stages_bench.py,
train_c_bidir_no_mntp.py, eval_c_bidir_no_mntp.py, train_teacher_bench.py,
distill_nova.py, distill_student_bench.py, eval_bench.py, eval_student.py, etc.
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed=42):
    """Full reproducibility seeding (deterministic + benchmark flags)."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}.")


def load_jsonl(path: str) -> List[Dict]:
    """Load .jsonl with columns: id, opt, asm (handles blank lines)."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return data


# Alias for backward compatibility with scripts that used this name
load_binarycorp_jsonl = load_jsonl


def parse_bench_opt(opt: str) -> Dict[str, str]:
    """Parse bench-format optimization label into components.

    Example: 'gcc_O3_x86_64_none' -> {compiler, optimization, architecture, variant}
    """
    parts = opt.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected bench opt label: {opt}")
    return {
        "compiler": parts[0],
        "optimization": parts[1],
        "architecture": "_".join(parts[2:-1]),
        "variant": parts[-1],
    }


def asm_to_text(asm) -> str:
    """Normalize asm field to a single string."""
    return "\n".join(asm) if isinstance(asm, list) else asm


def group_samples_by_id(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Group samples by function id, validating bench opt labels."""
    grouped: Dict[str, List[Dict]] = {}
    for sample in samples:
        parse_bench_opt(sample["opt"])
        grouped.setdefault(sample["id"], []).append(sample)
    return grouped


def build_eval_pairs(samples: List[Dict], seed: int = 42) -> List[Dict]:
    """Pair up variants of each function for retrieval evaluation."""
    grouped = defaultdict(list)
    for sample in samples:
        parse_bench_opt(sample["opt"])
        grouped[sample["id"]].append(sample)

    rng = random.Random(seed)
    pairs = []
    skipped = 0
    for fid, variants in grouped.items():
        if len(variants) < 2:
            skipped += 1
            continue
        ordered = variants[:]
        rng.shuffle(ordered)
        for idx, query in enumerate(ordered):
            target = ordered[(idx + 1) % len(ordered)]
            pairs.append({"id": fid, "query": query, "target": target})

    print(f"Built {len(pairs)} eval pairs from {len(grouped)} functions ({skipped} skipped with <2 variants).")
    return pairs

