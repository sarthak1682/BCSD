"""Multi-ground-truth (multi-GT) retrieval evaluation for the Bench dataset.

WHY THIS EXISTS
---------------
`eval_bench.py` scores retrieval with a *single* ground truth per query: it
builds a canonical target bank with exactly one target instance per function
(`build_canonical_target_bank` dedupes by id) and a query is correct if that one
target is retrieved. Yunru's request #1 is the opposite, more realistic setting:
the function pool contains *multiple* ground truths (several compiled variants of
the same function), and retrieving *any* of them counts as a hit.

This module implements that without modifying anything else. It is:
  * SEPARATE  — a standalone script, `eval_bench.py` is untouched.
  * NON-DESTRUCTIVE — it consumes a *saved* bench embeddings `.pt` (produced by
    each model's existing eval.py) and writes to a NEW filename
    (`<embeddings>.multigt_report.json` by default). It never overwrites the
    existing `*_metrics.json` / `*.full_report.json` reports or the embeddings.

WHAT IT COMPUTES
----------------
Using the asymmetric query/target encodings already saved in the `.pt`:
  * Full multi-GT retrieval over the entire target bank.
  * Pooled multi-GT retrieval at several pool sizes (function-pool stability),
    averaged over many random trials.
Metrics per setting: Recall@k (hit rate), NDCG@k (graded over the multiple
relevant variants), MRR, plus the average number of ground truths per query.

By default the query's own identical instance (same id AND same opt) is removed
from the candidate set so a trivial self-match cannot inflate scores; every
*other* variant of the same function is treated as a valid ground truth. Pass
`--no_exclude_self_opt` to keep it.

USAGE
-----
    python eval_bench_multigt.py \
        embeddings/binarycorp_bench/nova_teacher/eval_bench_embeddings.pt
    python eval_bench_multigt.py path/to/eval_bench_embeddings.pt \
        --device cuda --pool_sizes 50 100 200 500 --num_trials 100
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Allow running from anywhere; shared/ lives one level up.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def normalize_embeddings(tensor):
    return torch.nan_to_num(F.normalize(tensor.float(), p=2, dim=1))


def _eval_multigt(query_embs, target_embs, ids, query_opts, target_opts,
                  query_idx, bank_idx, ks, exclude_self_opt, device,
                  chunk_size=256):
    """Score a multi-GT retrieval task over an arbitrary query/bank selection.

    Args:
        query_embs, target_embs: normalized [N, D] tensors (query & target sides).
        ids: per-pair function id (shared by query and target of the pair).
        query_opts, target_opts: per-pair opt label for each side.
        query_idx: indices (into query_embs) to use as queries.
        bank_idx: indices (into target_embs) forming the retrieval pool/bank.
        ks: iterable of cutoffs for Recall@k / NDCG@k.
        exclude_self_opt: drop the query's own (id, opt) instance from candidates.
        device: torch device for the similarity matmul.
    """
    ks = sorted(set(ks))
    max_k = max(ks)

    bank = target_embs[bank_idx].to(device)            # [B, D]
    bank_ids = [ids[j] for j in bank_idx]
    bank_opts = [target_opts[j] for j in bank_idx]

    # (id, opt) -> position in bank (for self removal); id -> bank positions.
    pos_of = {}
    id_to_bankpos = defaultdict(list)
    for p, (fid, op) in enumerate(zip(bank_ids, bank_opts)):
        pos_of[(fid, op)] = p
        id_to_bankpos[fid].append(p)

    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    mrr_sum = 0.0
    gt_sum = 0
    scored = 0

    idcg_cache = {}

    def idcg(n):
        if n not in idcg_cache:
            idcg_cache[n] = sum(1.0 / math.log2(r + 2) for r in range(n))
        return idcg_cache[n]

    query_idx = list(query_idx)
    for start in range(0, len(query_idx), chunk_size):
        chunk = query_idx[start:start + chunk_size]
        q = query_embs[chunk].to(device)
        sims = q @ bank.T                              # [c, B]

        for row, qi in enumerate(chunk):
            fid = ids[qi]
            oq = query_opts[qi]
            rel = set(id_to_bankpos.get(fid, ()))
            sim_row = sims[row]

            if exclude_self_opt:
                self_p = pos_of.get((fid, oq))
                if self_p is not None and self_p in rel:
                    rel.discard(self_p)
                    sim_row = sim_row.clone()
                    sim_row[self_p] = float("-inf")

            if not rel:
                continue

            scored += 1
            R = len(rel)
            gt_sum += R

            kk = min(max_k, sim_row.numel())
            top_idx = torch.topk(sim_row, kk).indices.tolist()

            for k in ks:
                topk = top_idx[:k]
                if any(t in rel for t in topk):
                    sums[f"Recall@{k}"] += 1.0
                dcg = sum(
                    1.0 / math.log2(rank + 2)
                    for rank, t in enumerate(topk)
                    if t in rel
                )
                denom = idcg(min(R, k))
                if denom > 0:
                    sums[f"NDCG@{k}"] += dcg / denom

            best_rel_sim = max(sim_row[p].item() for p in rel)
            rank = int((sim_row > best_rel_sim).sum().item()) + 1
            mrr_sum += 1.0 / rank

    metrics = {key: (val / scored if scored else 0.0) for key, val in sums.items()}
    metrics["MRR"] = mrr_sum / scored if scored else 0.0
    metrics["queries_scored"] = scored
    metrics["avg_ground_truths"] = gt_sum / scored if scored else 0.0
    return metrics


def compute_full_multigt(result, ks, exclude_self_opt, device, max_queries=0):
    ids = result["ids"]
    query_opts = result["query_opts"]
    target_opts = result["target_opts"]
    query_embs = normalize_embeddings(result["query_embeddings"])
    target_embs = normalize_embeddings(result["target_embeddings"])

    query_idx = list(range(len(ids)))
    if max_queries and max_queries < len(query_idx):
        rng = np.random.default_rng(42)
        query_idx = sorted(rng.choice(len(ids), max_queries, replace=False).tolist())

    bank_idx = list(range(len(ids)))
    return _eval_multigt(
        query_embs, target_embs, ids, query_opts, target_opts,
        query_idx, bank_idx, ks, exclude_self_opt, device,
    )


def compute_pooled_multigt(result, pool_size, num_trials, ks, exclude_self_opt,
                           device, seed=42):
    """Sample `pool_size` functions per trial; pool = all their variants (multi-GT)."""
    ids = result["ids"]
    query_opts = result["query_opts"]
    target_opts = result["target_opts"]
    query_embs = normalize_embeddings(result["query_embeddings"])
    target_embs = normalize_embeddings(result["target_embeddings"])

    id_to_query = defaultdict(list)
    id_to_target = defaultdict(list)
    for i, fid in enumerate(ids):
        id_to_query[fid].append(i)
        id_to_target[fid].append(i)

    func_ids = list(id_to_target)
    rng = np.random.default_rng(seed)

    agg = defaultdict(float)
    trials_run = 0
    for _ in range(num_trials):
        if len(func_ids) <= pool_size:
            sampled = func_ids
        else:
            sampled = rng.choice(func_ids, pool_size, replace=False).tolist()

        bank_idx, query_idx = [], []
        for fid in sampled:
            bank_idx.extend(id_to_target[fid])
            query_idx.extend(id_to_query[fid])

        m = _eval_multigt(
            query_embs, target_embs, ids, query_opts, target_opts,
            query_idx, bank_idx, ks, exclude_self_opt, device,
        )
        if m["queries_scored"] == 0:
            continue
        trials_run += 1
        for k in ks:
            agg[f"Recall@{k}"] += m[f"Recall@{k}"]
            agg[f"NDCG@{k}"] += m[f"NDCG@{k}"]
        agg["MRR"] += m["MRR"]
        agg["avg_ground_truths"] += m["avg_ground_truths"]

    out = {key: (val / trials_run if trials_run else 0.0) for key, val in agg.items()}
    out["pool_size"] = pool_size
    out["trials"] = trials_run
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Multi-ground-truth retrieval eval over saved bench embeddings."
    )
    parser.add_argument(
        "embeddings_path",
        help="Path to a saved bench eval embeddings .pt "
             "(query/target embeddings + ids + opts).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <embeddings_path>.multigt_report.json "
             "(does NOT overwrite existing single-GT reports).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pool_sizes", type=int, nargs="+", default=[50, 100, 200, 500])
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--full_max_queries",
        type=int,
        default=0,
        help="Cap queries in the full-bank eval (0 = use all). "
             "Helpful when the bank is very large.",
    )
    parser.add_argument(
        "--skip_full",
        action="store_true",
        help="Skip the full-bank multi-GT eval (run only pooled).",
    )
    parser.add_argument(
        "--no_exclude_self_opt",
        action="store_true",
        help="Keep the query's own (id, opt) instance as a valid ground truth.",
    )
    args = parser.parse_args()

    embeddings_path = os.path.abspath(args.embeddings_path)
    output_path = args.output or (embeddings_path + ".multigt_report.json")
    exclude_self_opt = not args.no_exclude_self_opt

    print(f"Loading embeddings from {embeddings_path}")
    result = torch.load(embeddings_path, map_location="cpu")

    required = {"ids", "query_opts", "target_opts", "query_embeddings", "target_embeddings"}
    missing = required - set(result)
    if missing:
        raise KeyError(
            f"Embeddings file is missing keys {missing}. "
            "This script expects a *bench* eval .pt (query/target split), "
            "not a 3M-style single-embedding file."
        )

    report = {
        "source": embeddings_path,
        "setting": "multi_ground_truth",
        "exclude_self_opt": exclude_self_opt,
        "ks": sorted(set(args.ks)),
        "pooled_multigt": {},
    }

    print(f"Device: {args.device} | exclude_self_opt={exclude_self_opt}")

    for pool_size in args.pool_sizes:
        print(f"  pooled multi-GT @ pool_size={pool_size} "
              f"({args.num_trials} trials)...")
        report["pooled_multigt"][f"pool{pool_size}"] = compute_pooled_multigt(
            result, pool_size, args.num_trials, args.ks, exclude_self_opt, args.device
        )

    if not args.skip_full:
        print("  full-bank multi-GT...")
        report["full_multigt"] = compute_full_multigt(
            result, args.ks, exclude_self_opt, args.device, args.full_max_queries
        )

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Multi-GT report ===")
    for pool_size in args.pool_sizes:
        m = report["pooled_multigt"][f"pool{pool_size}"]
        ks = sorted(set(args.ks))
        recalls = " ".join(f"R@{k}={m[f'Recall@{k}']:.4f}" for k in ks)
        ndcgs = " ".join(f"NDCG@{k}={m[f'NDCG@{k}']:.4f}" for k in ks)
        print(f"[pool{pool_size}] {recalls} | {ndcgs} | MRR={m['MRR']:.4f} "
              f"| avgGT={m['avg_ground_truths']:.2f}")
    if not args.skip_full:
        m = report["full_multigt"]
        ks = sorted(set(args.ks))
        recalls = " ".join(f"R@{k}={m[f'Recall@{k}']:.4f}" for k in ks)
        ndcgs = " ".join(f"NDCG@{k}={m[f'NDCG@{k}']:.4f}" for k in ks)
        print(f"[full]   {recalls} | {ndcgs} | MRR={m['MRR']:.4f} "
              f"| avgGT={m['avg_ground_truths']:.2f} | queries={m['queries_scored']}")

    print(f"\nSaved multi-GT report to {output_path}")


if __name__ == "__main__":
    main()
