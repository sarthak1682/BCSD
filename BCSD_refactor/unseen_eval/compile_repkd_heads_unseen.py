#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UE = ROOT / "unseen_eval"
PROJECTS = ("moonshine", "mquickjs", "zvec")

# opt-head results (produced by run_repkd_heads.sh)
OPT = {
    "nova_repkd": UE / "repkd_heads" / "opt" / "nova_repkd",
    "l4d_repkd": UE / "repkd_heads" / "opt" / "l4d_repkd",
}
# BCSD-head results
BCSD = {
    "nova_repkd_cl20_lossselect": UE / "joint_repkd_{p}.json",
    "l4d_repkd_cl20_novamatched": UE / "repkd_heads" / "bcsd" / "l4d_repkd_cl20" / "{p}.json",
    "l4d_repkd_cl1_fullnet": UE / "l4d_repkd_postcl_{p}.json",
    "l4d_repkd_precl_nohead": UE / "l4d_repkd_precl_{p}.json",
}


def _report(path: Path) -> dict:
    d = json.loads(path.read_text())
    return d.get("report", d)


def bcsd_metrics(path: Path) -> dict:
    r = _report(path)
    pr, fr = r["pooled_recall"], r["full_recall_one_target_per_function"]
    return {
        "pool50": pr["Recall@1_Pool50"],
        "pool500": pr["Recall@1_Pool500"],
        "full_r1": fr["Recall@1"]["recall"],
        "full_r5": fr["Recall@5"]["recall"],
        "full_r10": fr["Recall@10"]["recall"],
    }


def opt_metrics(dir_: Path, project: str) -> dict:
    d4 = json.loads((dir_ / project / "test_4way.json").read_text())
    d2 = json.loads((dir_ / project / "test_o0_vs_o3_from_4way.json").read_text())
    return {
        "acc4": d4["test_metrics"]["accuracy"],
        "f1_4": d4["test_metrics"]["macro_f1"],
        "acc_o0o3": d2["test_metrics"]["accuracy"],
        "auc_o0o3": d2["test_metrics"]["roc_auc"],
    }


def main() -> None:
    rows: list[dict] = []

    for name, dir_ in OPT.items():
        for p in PROJECTS:
            m = opt_metrics(dir_, p)
            rows.append({"task": "opt", "model": name, "project": p, **m})
    for name, tmpl in BCSD.items():
        for p in PROJECTS:
            path = Path(str(tmpl).format(p=p))
            if not path.is_file():
                continue
            rows.append({"task": "bcsd", "model": name, "project": p,
                         **bcsd_metrics(path)})

    out_json = UE / "repkd_heads_unseen.json"
    out_csv = UE / "repkd_heads_unseen.csv"
    keys = sorted({k for r in rows for k in r})
    out_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    with out_csv.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}\n")

    def avg(name, task, field):
        vals = [r[field] for r in rows
                if r["task"] == task and r["model"] == name and field in r]
        return sum(vals) / len(vals) if vals else float("nan")

    # which BCSD-head variant pairs with each backbone (Nova-matched cl20)
    bcsd_head = {
        "nova_repkd": "nova_repkd_cl20_lossselect",
        "l4d_repkd": "l4d_repkd_cl20_novamatched",
    }
    print("=" * 100)
    print("repKD task heads on unseen projects — macro-average over moonshine/mquickjs/zvec")
    print("=" * 100)
    print(f"{'backbone':14s} | {'--- opt head ---':^38s} | {'--- BCSD head ---':^30s}")
    print(f"{'':14s} | {'4-wayAcc':>10s} {'4-wayF1':>9s} {'O0/O3':>8s} {'AUROC':>8s}"
          f" | {'Pool50':>8s} {'Pool500':>8s} {'FullR1':>8s} ")
    for name in ("nova_repkd", "l4d_repkd"):
        b = bcsd_head[name]
        print(f"{name:14s} | {avg(name,'opt','acc4'):>10.3f} {avg(name,'opt','f1_4'):>9.3f} "
              f"{avg(name,'opt','acc_o0o3'):>8.3f} {avg(name,'opt','auc_o0o3'):>8.4f}"
              f" | {avg(b,'bcsd','pool50'):>8.3f} {avg(b,'bcsd','pool500'):>8.3f} "
              f"{avg(b,'bcsd','full_r1'):>8.3f}")

    print()
    print("BCSD-head variants (unseen, macro-avg):")
    for name in BCSD:
        if not any(r["task"] == "bcsd" and r["model"] == name for r in rows):
            continue
        print(f"  {name:34s} Pool50={avg(name,'bcsd','pool50'):.3f} "
              f"Pool500={avg(name,'bcsd','pool500'):.3f} "
              f"FullR1={avg(name,'bcsd','full_r1'):.3f} "
              f"FullR5={avg(name,'bcsd','full_r5'):.3f}")

    print()
    print("Per-project detail:")
    for p in PROJECTS:
        print(f"  [{p}]")
        for name, dir_ in OPT.items():
            m = opt_metrics(dir_, p)
            print(f"    opt  {name:12s} 4wAcc={m['acc4']:.3f} F1={m['f1_4']:.3f} "
                  f"O0/O3={m['acc_o0o3']:.3f} AUC={m['auc_o0o3']:.4f}")
        for name, tmpl in BCSD.items():
            path = Path(str(tmpl).format(p=p))
            if not path.is_file():
                continue
            m = bcsd_metrics(path)
            print(f"    bcsd {name:28s} Pool50={m['pool50']:.3f} Pool500={m['pool500']:.3f} "
                  f"FullR1={m['full_r1']:.3f} R5={m['full_r5']:.3f}")


if __name__ == "__main__":
    main()
