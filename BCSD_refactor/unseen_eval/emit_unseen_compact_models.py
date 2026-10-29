#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UE = ROOT / "unseen_eval"
PROJ = ("moonshine", "mquickjs", "zvec")

# display name -> (opt_class result dir, retrieval report template per project)
MODELS = [
    ("CLAP", "clap", "clap_final_{p}.json"),
    ("jTrans", "jtrans", "jtrans_{p}.json"),
    ("EBM-full", "ebm-full", "ebm_full/{p}/eval_bench_report_{p}.json"),
    ("EBM-no-causal", "ebm-no-causal", "ebm_nocausal/{p}/eval_bench_report_{p}.json"),
    ("Student", "student", "student/{p}/eval_student_bench_metrics_{p}.json"),
    ("Joint scratch 23.5M", "joint-scratch", "js_23m_{p}.json"),
    ("asm2vec", "asm2vec", "asm2vec/{p}/eval_asm2vec_bench_metrics_{p}.json"),
]


def _report(path: Path) -> dict:
    d = json.loads(path.read_text())
    return d.get("report", d)


def opt(dirname: str, p: str) -> dict:
    a = json.loads((UE / "opt_class" / "results" / dirname / p / "4way.json").read_text())
    b = json.loads((UE / "opt_class" / "results" / dirname / p / "o0-vs-o3.json").read_text())
    return {
        "acc4": a["unseen_test_metrics"]["accuracy"],
        "f1": a["unseen_test_metrics"]["macro_f1"],
        "bacc": b["unseen_test_metrics"]["accuracy"],
        "auc": b["unseen_test_metrics"]["roc_auc"],
    }


def ret(tmpl: str, p: str) -> dict:
    r = _report(UE / tmpl.format(p=p))
    pr, fr = r["pooled_recall"], r["full_recall_one_target_per_function"]
    return {
        "p50": pr["Recall@1_Pool50"],
        "p500": pr["Recall@1_Pool500"],
        "r1": fr["Recall@1"]["recall"],
        "r5": fr["Recall@5"]["recall"],
    }


def main() -> None:
    def avg(vals):
        return sum(vals) / len(vals)

    print(r"%% compact unseen rows for non-repKD models (30ep excluded)")
    print(r"%% Model & 4wAcc & MacroF1 & BinAcc & AUROC & P@50 & P@500 & R@1 & R@5 \\")
    rows = []
    for name, d, tmpl in MODELS:
        o = {k: avg([opt(d, p)[k] for p in PROJ]) for k in ("acc4", "f1", "bacc", "auc")}
        r = {k: avg([ret(tmpl, p)[k] for p in PROJ]) for k in ("p50", "p500", "r1", "r5")}
        rows.append((name, o, r))
        print(f"{name} & {o['acc4']:.3f} & {o['f1']:.3f} & {o['bacc']:.3f} & "
              f"{o['auc']:.3f} & {r['p50']:.3f} & {r['p500']:.3f} & {r['r1']:.3f} & {r['r5']:.3f} \\\\")

    # column maxima (for bolding), across these models only
    print()
    best = {
        "acc4": max(x[1]["acc4"] for x in rows), "f1": max(x[1]["f1"] for x in rows),
        "bacc": max(x[1]["bacc"] for x in rows), "auc": max(x[1]["auc"] for x in rows),
        "p50": max(x[2]["p50"] for x in rows), "p500": max(x[2]["p500"] for x in rows),
        "r1": max(x[2]["r1"] for x in rows), "r5": max(x[2]["r5"] for x in rows),
    }
    print("%% best per column (within non-repKD models):",
          {k: round(v, 3) for k, v in best.items()})


if __name__ == "__main__":
    main()
