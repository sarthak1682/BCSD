import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


OPT_LEVELS = ("O0", "O1", "O2", "O3")
COMPILERS = ("clang", "gcc")
ARCHITECTURES = ("x86_64", "arm64", "mips64", "powerpc64")
VARIANTS = ("none", "all", "bogus", "flattening", "substitution")


def parse_bench_opt(opt):
    parts = opt.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected bench opt label: {opt}")
    return {
        "compiler": parts[0],
        "optimization": parts[1],
        "architecture": "_".join(parts[2:-1]),
        "variant": parts[-1],
    }


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}.")


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return data


def build_eval_pairs(samples, seed=42):
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


def normalize_embeddings(tensor):
    return torch.nan_to_num(F.normalize(tensor.float(), p=2, dim=1))


def build_canonical_target_bank(ids, target_opts, target_embs, target_filter=None):
    canonical_indices = []
    target_ids = []
    seen_ids = set()

    for idx, opt in enumerate(target_opts):
        parsed = parse_bench_opt(opt)
        if target_filter is not None and not target_filter(parsed):
            continue
        fid = ids[idx]
        if fid in seen_ids:
            continue
        seen_ids.add(fid)
        canonical_indices.append(idx)
        target_ids.append(fid)

    if not canonical_indices:
        return [], torch.empty((0, target_embs.shape[1]), dtype=target_embs.dtype)
    return target_ids, target_embs[canonical_indices]


def compute_filtered_recall(result, query_filter=None, target_filter=None, k=1, chunk_size=256):
    ids = result["ids"]
    query_opts = result["query_opts"]
    target_opts = result["target_opts"]
    query_embs = normalize_embeddings(result["query_embeddings"])
    target_embs = normalize_embeddings(result["target_embeddings"])

    query_indices = []
    for idx, opt in enumerate(query_opts):
        parsed = parse_bench_opt(opt)
        if query_filter is None or query_filter(parsed):
            query_indices.append(idx)

    target_ids, target_bank = build_canonical_target_bank(ids, target_opts, target_embs, target_filter)

    if not query_indices or not target_ids:
        return {
            "recall": None,
            "queries": len(query_indices),
            "targets": len(target_ids),
            "correct": 0,
        }

    k = min(k, len(target_ids))
    correct = 0

    for start in range(0, len(query_indices), chunk_size):
        batch_indices = query_indices[start:start + chunk_size]
        sim = query_embs[batch_indices] @ target_bank.T
        top_k = sim.topk(k, dim=1).indices

        for row, query_idx in enumerate(batch_indices):
            query_id = ids[query_idx]
            if any(target_ids[target_idx] == query_id for target_idx in top_k[row].tolist()):
                correct += 1

    return {
        "recall": correct / len(query_indices),
        "queries": len(query_indices),
        "targets": len(target_ids),
        "correct": correct,
    }


def compute_pooled_recall(result, pool_size=50, num_trials=100, seed=42):
    ids = result["ids"]
    query_embs = normalize_embeddings(result["query_embeddings"])
    target_embs = normalize_embeddings(result["target_embeddings"])

    id_to_indices = defaultdict(list)
    for idx, fid in enumerate(ids):
        id_to_indices[fid].append(idx)

    paired_ids = list(id_to_indices)
    rng = np.random.default_rng(seed)
    total_correct = 0
    total_queries = 0

    for _ in range(num_trials):
        if len(paired_ids) < pool_size:
            sampled_ids = list(paired_ids)
        else:
            sampled_ids = rng.choice(paired_ids, pool_size, replace=False).tolist()

        sampled_indices = [int(rng.choice(id_to_indices[fid])) for fid in sampled_ids]
        sim = query_embs[sampled_indices] @ target_embs[sampled_indices].T
        preds = sim.argmax(dim=1).tolist()

        total_correct += sum(1 for i, pred in enumerate(preds) if pred == i)
        total_queries += len(sampled_indices)

    return total_correct / max(1, total_queries)


def compute_directed_matrix(result, values, field, ks=(1, 5, 10)):
    output = {}
    for k in ks:
        matrix = {}
        for source_value in values:
            matrix[source_value] = {}
            for target_value in values:
                matrix[source_value][target_value] = compute_filtered_recall(
                    result,
                    query_filter=lambda parsed, v=source_value: parsed[field] == v,
                    target_filter=lambda parsed, v=target_value: parsed[field] == v,
                    k=k,
                )
        output[f"Recall@{k}"] = matrix
    return output


def compute_counts(result):
    query_counts = defaultdict(int)
    target_counts = defaultdict(int)
    function_ids = set(result["ids"])

    for opt in result["query_opts"]:
        parsed = parse_bench_opt(opt)
        query_counts["optimization:" + parsed["optimization"]] += 1
        query_counts["compiler:" + parsed["compiler"]] += 1
        query_counts["architecture:" + parsed["architecture"]] += 1
        query_counts["variant:" + parsed["variant"]] += 1

    for opt in result["target_opts"]:
        parsed = parse_bench_opt(opt)
        target_counts["optimization:" + parsed["optimization"]] += 1
        target_counts["compiler:" + parsed["compiler"]] += 1
        target_counts["architecture:" + parsed["architecture"]] += 1
        target_counts["variant:" + parsed["variant"]] += 1

    return {
        "pairs": len(result["ids"]),
        "unique_function_ids": len(function_ids),
        "query_counts": dict(query_counts),
        "target_counts": dict(target_counts),
    }


def compute_report(result):
    return {
        "counts": compute_counts(result),
        "pooled_recall": {
            "Recall@1_Pool50": compute_pooled_recall(result, pool_size=50),
            "Recall@1_Pool100": compute_pooled_recall(result, pool_size=100),
            "Recall@1_Pool200": compute_pooled_recall(result, pool_size=200),
            "Recall@1_Pool500": compute_pooled_recall(result, pool_size=500),
        },
        "full_recall_one_target_per_function": {
            "Recall@1": compute_filtered_recall(result, k=1),
            "Recall@5": compute_filtered_recall(result, k=5),
            "Recall@10": compute_filtered_recall(result, k=10),
        },
        "optimization_transfer": compute_directed_matrix(result, OPT_LEVELS, "optimization"),
        "compiler_transfer": compute_directed_matrix(result, COMPILERS, "compiler"),
        "architecture_transfer": compute_directed_matrix(result, ARCHITECTURES, "architecture"),
        "variant_transfer": compute_directed_matrix(result, VARIANTS, "variant"),
    }


def print_matrix(metrics, title, recall_key="Recall@1"):
    matrix = metrics[recall_key]
    labels = list(matrix.keys())
    print(f"\n{title} ({recall_key})")
    print("query\\target\t" + "\t".join(labels))
    for source in labels:
        cells = []
        for target in labels:
            value = matrix[source][target]["recall"]
            cells.append("n/a" if value is None else f"{value:.4f}")
        print(source + "\t" + "\t".join(cells))


def print_report_summary(report):
    print("\n--- Pooled Recall ---")
    for key, value in report["pooled_recall"].items():
        print(f"{key}: {value:.4f}")

    print("\n--- Full Recall (One Target Per Function) ---")
    for key, value in report["full_recall_one_target_per_function"].items():
        print(f"{key}: {value['recall']:.4f}")

    print_matrix(report["optimization_transfer"], "Optimization transfer", "Recall@1")
    print_matrix(report["compiler_transfer"], "Compiler transfer", "Recall@1")
    print_matrix(report["architecture_transfer"], "Architecture transfer", "Recall@1")
    print_matrix(report["variant_transfer"], "Variant transfer", "Recall@1")


def main():
    parser = argparse.ArgumentParser(description="Compute a full bench retrieval report from saved embeddings.")
    parser.add_argument(
        "embeddings_path",
        nargs="?",
        default="eval_student_bench_results_20.pt",
        help="Path to saved bench eval embeddings .pt file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <embeddings_path>.full_report.json.",
    )
    args = parser.parse_args()

    embeddings_path = os.path.abspath(args.embeddings_path)
    output_path = args.output or (embeddings_path + ".full_report.json")

    result = torch.load(embeddings_path, map_location="cpu")
    report = compute_report(result)
    report["source"] = embeddings_path

    print(json.dumps(report["pooled_recall"], indent=2))
    print(json.dumps(report["full_recall_one_target_per_function"], indent=2))
    print_matrix(report["optimization_transfer"], "Optimization transfer")
    print_matrix(report["compiler_transfer"], "Compiler transfer")
    print_matrix(report["architecture_transfer"], "Architecture transfer")
    print_matrix(report["variant_transfer"], "Variant transfer")

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full report to {output_path}")


if __name__ == "__main__":
    main()
