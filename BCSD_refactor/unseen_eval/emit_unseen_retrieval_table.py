#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "unseen_eval" / "summary.csv"
PROJECTS = ("moonshine", "mquickjs", "zvec")

# (csv prefix, paper name, family group id)
ROWS = [
    ("clap_final", "CLAP", 0),
    ("clap_30ep", "CLAP 30ep", 0),
    ("qwen_ebm", "Qwen-EBM", 0),
    ("nova_dora", "Nova-BCSD DoRA", 0),
    ("jtrans", "jTrans", 0),
    ("jtrans_30ep", "jTrans 30ep", 0),
    ("asm2vec", "asm2vec", 0),
    ("ebm_full", "EBM-full", 1),
    ("ebm_nocausal", "EBM-no-causal", 1),
    ("student", "Student", 2),
    ("student_scratch", "Student (no distill)", 2),
    ("student_ebmfull", "Student EBM-full", 2),
    ("js_23m", "Joint scratch 23.5M", 3),
    ("js_83m_precl", "Joint scratch 83M pre-CL", 3),
    ("js_83m_frozenhead", "Joint scratch 83M frozen-head", 3),
    ("js_83m", "Joint scratch 83M", 3),
    ("mlm_83m_precl", "MLM-only 83M pre-CL", 4),
    ("mlm_83m_postcl", "MLM-only 83M post-CL", 4),
    ("joint_repkd", "Joint repKD 97M", 5),
    ("l4d_repkd_precl", "L4D repKD 97M pre-CL", 5),
    ("l4d_repkd_postcl", "L4D repKD 97M post-CL", 5),
    ("l4d_mean", "L4D (mean pool)", 6),
    ("l4d_linear", "L4D (linear head)", 6),
    ("l4d_attention", "L4D (attention head)", 6),
]


def load() -> dict:
    data: dict[str, dict[str, dict[str, float]]] = {}
    with SUMMARY.open() as h:
        for row in csv.DictReader(h):
            model = row["model"]
            prefix, _, project = model.rpartition("_")
            data.setdefault(prefix, {})[project] = {
                "pool50": float(row["Pool50"]),
                "fullr1": float(row["FullR1"]),
            }
    return data


def f(v: float) -> str:
    return f"{v:.3f}"


def main() -> None:
    data = load()
    lines: list[str] = []
    prev_group = None
    for prefix, name, group in ROWS:
        d = data[prefix]
        cells = []
        for p in PROJECTS:
            cells += [f(d[p]["pool50"]), f(d[p]["fullr1"])]
        p50 = sum(d[p]["pool50"] for p in PROJECTS) / len(PROJECTS)
        r1 = sum(d[p]["fullr1"] for p in PROJECTS) / len(PROJECTS)
        cells += [f(p50), f(r1)]
        if prev_group is not None and group != prev_group:
            lines.append("\\midrule")
        prev_group = group
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    print("%% rows for tab:unseen-retrieval")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
