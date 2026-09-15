"""Generic data utilities shared across all scripts.

Deduplicated from: run_nova_ebm_stages.py, run_nova_ebm_stages_bench.py,
train_c_bidir_no_mntp.py, eval_c_bidir_no_mntp.py, train_teacher_bench.py,
distill_nova.py, distill_student_bench.py, eval_bench.py, eval_student.py, etc.
"""

import json
import os
import random
import re
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


EBM_BASE_SPECIAL_TOKENS = [
    "<addr>", "<byte>", "<str>", "<BLK>", "<SEP>", "<EOS>", "<PAD>", "[MASK]"
]

_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_LEADING_ADDR_RE = re.compile(r"^\s*(?:0x)?[0-9a-fA-F]{3,}:?\s+")
_LEADING_BYTES_RE = re.compile(r"^\s*(?:(?:[0-9a-fA-F]{2})\s+){2,}")
_ADDR_SYMBOL_RE = re.compile(
    r"\b(?:(?:loc|sub|off|unk|seg|asc)_[A-Za-z0-9_.$@]+|a[A-Z][A-Za-z0-9_]*)\b"
)
_HEX_ADDR_RE = re.compile(r"(?<![A-Za-z0-9_$])(?:0x[0-9a-fA-F]{3,}|[0-9a-fA-F]{3,}h)(?![A-Za-z0-9_$])")
_HEX_BYTE_RE = re.compile(r"(?<![A-Za-z0-9_$])(?:0x[0-9a-fA-F]{1,2}|[0-9a-fA-F]{1,2}h)(?![A-Za-z0-9_$])")
_DECIMAL_RE = re.compile(r"(?<![A-Za-z0-9_$])[-+]?\d+(?![A-Za-z0-9_$])")
_BLOCK_LABEL_RE = re.compile(
    r"^\s*(?:<label-\d+>|(?:loc|sub|off|unk|INSTR)[A-Za-z0-9_.$@]*:?)\s*$"
)


def bench_setting_block(opt: str) -> str:
    """Return the binary_sim-style setting block for a bench opt label.

    The upstream data generator appends a final block like
    ``o3 gcc x86 bcf`` to each function. Bench labels carry the same pieces in
    ``compiler_OPT_arch_variant`` form.
    """
    parsed = parse_bench_opt(opt)
    return " ".join(
        part.lower()
        for part in (
            parsed["optimization"],
            parsed["compiler"],
            parsed["architecture"],
            parsed["variant"],
        )
        if part
    )


def ebm_setting_block(opt: str) -> str:
    """Return the original EBM language-token block for a setting label.

    Benchmark labels encode compiler, optimization, architecture, and variant
    fields (for example, ``gcc_O3_x86_64_none``). BinaryCorp3M labels contain
    only an optimization level (for example, ``O3``).
    """
    normalized = opt.strip()
    if not normalized:
        raise ValueError("EBM setting label must not be empty")
    if "_" in normalized:
        return bench_setting_block(normalized)
    if normalized.lower() not in {"o0", "o1", "o2", "o3", "os"}:
        raise ValueError(f"Unexpected BinaryCorp3M optimization label: {opt}")
    return normalized.lower()


def ebm_preprocess_asm(asm, opt: str = "", include_settings: bool = False) -> str:
    """Apply binary_sim/EBM-style assembly normalization.

    The upstream generator replaces addresses, strings, and byte/immediate
    values with angle-bracket special tokens and inserts ``<BLK>`` between basic
    blocks. When ``include_settings`` is true, append a final settings block.
    """
    text = asm_to_text(asm)
    out_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if _BLOCK_LABEL_RE.match(line):
            if out_lines and out_lines[-1] != "<BLK>":
                out_lines.append("<BLK>")
            continue

        line = _STRING_RE.sub(" <str> ", line)
        line = _LEADING_ADDR_RE.sub("<addr> ", line)
        line = re.sub(r"^<addr> (?:(?:[0-9a-fA-F]{2})\s+){2,}", "<addr> <byte> ", line)
        line = _LEADING_BYTES_RE.sub("<byte> ", line)
        line = _ADDR_SYMBOL_RE.sub("<addr>", line)
        line = _HEX_ADDR_RE.sub("<addr>", line)
        line = _HEX_BYTE_RE.sub("<byte>", line)
        line = _DECIMAL_RE.sub("<byte>", line)
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            out_lines.append(line)

    while out_lines and out_lines[0] == "<BLK>":
        out_lines.pop(0)
    while out_lines and out_lines[-1] == "<BLK>":
        out_lines.pop()

    if include_settings and opt:
        if out_lines:
            out_lines.append("<BLK>")
        out_lines.append(ebm_setting_block(opt))

    return "\n".join(out_lines)


def build_ebm_translation_item(
    source_asm,
    source_opt: str,
    target_asm,
    target_opt: str,
    *,
    preprocess: bool = False,
):
    """Build ``(source, target, target_settings)`` in original EBM format.

    The source string ends with its language-token block. Translation
    collators insert ``<SEP>`` and the returned target language-token block
    before the target body, then terminate the target with ``<EOS>``.
    """
    if preprocess:
        source = ebm_preprocess_asm(
            source_asm, source_opt, include_settings=True
        )
        target = ebm_preprocess_asm(target_asm)
    else:
        source_body = asm_to_text(source_asm).rstrip()
        source = f"{source_body}\n{ebm_setting_block(source_opt)}"
        target = asm_to_text(target_asm).strip()
    return source, target, ebm_setting_block(target_opt)


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


def get_embeddings_dir(dataset: str, model: str) -> str:
    """Return BCSD_refactor/embeddings/{dataset}/{model}/, creating it if needed.

    Uses this file's location to find the repo root — no hardcoded absolute paths.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    emb_dir = os.path.join(repo_root, "embeddings", dataset, model)
    os.makedirs(emb_dir, exist_ok=True)
    return emb_dir
