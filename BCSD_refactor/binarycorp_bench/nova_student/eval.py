import json
import math
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import sys
import os
import random
from collections import defaultdict

# Setup paths to import shared modules and evaluation script
script_dir_file = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir_file, "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir_file, "../")))

from eval_bench import set_seed, load_jsonl, build_eval_pairs, compute_report, print_report_summary
from shared.student_model import StudentDistillationModule, LatentAttentionLayer, PositionalEncoding
from shared.nova_utils import setup_nova_tokenizer
from shared.data_utils import get_embeddings_dir
from shared.profiling import InferenceProfiler

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except Exception:
    TSNE = None
    HAS_SKLEARN = False

CACHE_DIR = "/home/ra72yeq/.cache/huggingface/hub/models--lt-asset--nova-1.3b/snapshots/4b4805bac4f13ef8bec678072ef60609ea3b0e77"

repo_root = os.path.abspath(os.path.join(script_dir_file, "../../../"))
TEST_PATH = os.path.join(repo_root, "nvemb", "output_benchset_rebalanced_test_nova.jsonl")
if not os.path.exists(TEST_PATH):
    TEST_PATH = os.path.join(repo_root, "output_benchset_rebalanced_test_nova.jsonl")

INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
RUN_ID = 20
STUDENT_DIR = os.path.join(script_dir_file, f"nova_distilled_student_{RUN_ID}_bench")
RESULTS_PATH = os.path.join(script_dir_file, f"eval_student_bench_results_{RUN_ID}.pt")  # legacy fallback
TSNE_PATH = os.path.join(script_dir_file, f"eval_student_bench_tsne_{RUN_ID}.png")
TSNE_NUM_PAIRS = 40

device = "cuda" if torch.cuda.is_available() else "cpu"

base_tokenizer, nova_tokenizer, MASK_ID = setup_nova_tokenizer()
print("Loaded tokenizer")

with open(os.path.join(STUDENT_DIR, "student_config.json"), "r") as f:
    cfg = json.load(f)
print("Loaded cfg")
pad_id = base_tokenizer.pad_token_id or 0
student = StudentDistillationModule(
    vocab_size=cfg["vocab_size"],
    hidden_dim=cfg["hidden_dim"],
    num_layers=cfg["num_layers"],
    pad_id=pad_id
).to(device).to(torch.bfloat16)

lal_head = LatentAttentionLayer(hidden_dim=cfg["hidden_dim"]).to(device).to(torch.bfloat16)

student.load_state_dict(torch.load(os.path.join(STUDENT_DIR, "student_model.pt"), map_location="cpu"))
lal_head.load_state_dict(torch.load(os.path.join(STUDENT_DIR, "lal_head.pt"), map_location="cpu"))

student.eval()
lal_head.eval()
print("Loaded student and lal head")

test_samples = load_jsonl(TEST_PATH)
print("Loaded test samples")


@torch.no_grad()
def encode_student_texts(texts, batch_size=32, max_len=1024, profiler=None):
    all_embs = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_idx = (i // batch_size) + 1
        batch = texts[i:i+batch_size]
        all_ids_list = []
        for text, char_types in batch:
            result = nova_tokenizer.encode("", text, char_types)
            ids = result["input_ids"][:max_len]
            all_ids_list.append(ids)

        max_len_batch = max(len(x) for x in all_ids_list)
        pad_ids = np.full((len(batch), max_len_batch), pad_id, dtype=np.int64)
        for j, ids in enumerate(all_ids_list):
            pad_ids[j, :len(ids)] = ids

        input_ids = torch.tensor(pad_ids, device=device)
        key_padding_mask = (input_ids == pad_id)

        if profiler is not None:
            with profiler:
                hidden = student(input_ids, key_padding_mask=key_padding_mask)
                emb = lal_head(hidden, key_padding_mask=key_padding_mask)
            profiler.total_samples += len(batch)
        else:
            hidden = student(input_ids, key_padding_mask=key_padding_mask)
            emb = lal_head(hidden, key_padding_mask=key_padding_mask)

        all_embs.append(emb.float().cpu())

        if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches:
            print(f"Processed batch {batch_idx}/{total_batches} ({min(i+batch_size, len(texts))}/{len(texts)} texts)")

    return torch.cat(all_embs, dim=0)


def extract_student_embeddings(pairs, batch_size=32, max_len=1024):
    query_texts = []
    target_texts = []
    all_ids = []
    query_opts = []
    target_opts = []

    for pair in pairs:
        query_asm = pair["query"]["asm"]
        target_asm = pair["target"]["asm"]
        query_text = INSTRUCT_TEMPLATE + query_asm
        query_texts.append((query_text, "0" * len(INSTRUCT_TEMPLATE) + "1" * len(query_asm)))
        target_texts.append((target_asm, "1" * len(target_asm)))
        all_ids.append(pair["id"])
        query_opts.append(pair["query"]["opt"])
        target_opts.append(pair["target"]["opt"])

    profiler = InferenceProfiler(device)

    # Warmup: run first batch outside profiler stats
    print("Warmup pass...")
    _ = encode_student_texts(query_texts[:batch_size], batch_size=batch_size, max_len=max_len)

    print("Encoding query embeddings...")
    query_embeddings = encode_student_texts(query_texts, batch_size=batch_size, max_len=max_len, profiler=profiler)
    print("Encoding target embeddings...")
    target_embeddings = encode_student_texts(target_texts, batch_size=batch_size, max_len=max_len, profiler=profiler)

    return {
        "ids": all_ids,
        "query_opts": query_opts,
        "target_opts": target_opts,
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
        "stats": profiler.get_stats(),
    }


def main():
    set_seed(42)

    eval_pairs = build_eval_pairs(test_samples, seed=42)
    results = extract_student_embeddings(eval_pairs, batch_size=32, max_len=1024)

    emb_dir = get_embeddings_dir("binarycorp_bench", "nova_student")
    RESULTS_PATH = os.path.join(emb_dir, "eval_bench_embeddings.pt")
    torch.save(results, RESULTS_PATH)
    print(f"Saved embeddings to {RESULTS_PATH}")

    print("\n" + "="*50)
    print(f"NOVA STUDENT | Latency: {results['stats']['avg_ms_per_sample']:.2f} ms/function")
    print(f"NOVA STUDENT | Memory:  {results['stats']['peak_memory_mb']:.1f} MB peak")
    print("="*50)

    report = compute_report(results)
    report["source"] = os.path.abspath(RESULTS_PATH)
    report["model"] = f"nova_student_run{RUN_ID}"
    report["data"] = os.path.abspath(TEST_PATH)

    metrics_path = os.path.join(emb_dir, f"eval_student_bench_metrics_{RUN_ID}.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full report to {metrics_path}")

    print_report_summary(report)


if __name__ == "__main__":
    main()
