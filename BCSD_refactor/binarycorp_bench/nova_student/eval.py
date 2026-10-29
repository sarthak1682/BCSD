import argparse
import json
import os
import re
import sys

import numpy as np
import torch

script_dir_file = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir_file, "../../")))

from binarycorp_bench.eval_bench import (
    build_eval_pairs,
    compute_report,
    load_jsonl,
    print_report_summary,
    set_seed,
)
from shared.student_model import LatentAttentionLayer, StudentDistillationModule
from shared.nova_utils import setup_nova_tokenizer
from shared.data_utils import get_embeddings_dir
from shared.profiling import InferenceProfiler


script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
TEST_PATH = os.path.join(script_dir, "./nvemb/output_benchset_rebalanced_test_nova.jsonl")
TEST_PATH = os.environ.get("EVAL_DATA", TEST_PATH)

INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
EVAL_TAG = os.environ.get("EVAL_TAG", "").strip()
STUDENT_DIR = os.environ.get("STUDENT_DIR")


def _slug(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_.-")


def _default_eval_label(student_dir):
    parent = os.path.basename(os.path.dirname(os.path.abspath(student_dir)))
    leaf = os.path.basename(os.path.abspath(student_dir))
    return _slug(f"{parent}_{leaf}") or "nova_student"


# student_best/ saves as student_model_best.pt; student_final/ saves as student_model.pt
def _ckpt(name):
    best = os.path.join(STUDENT_DIR, name.replace(".pt", "_best.pt"))
    plain = os.path.join(STUDENT_DIR, name)
    return best if os.path.exists(best) else plain


@torch.no_grad()
def encode_student_texts(texts, batch_size=32, max_len=1024, profiler=None):
    all_embs = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    instruct_token_len = len(nova_tokenizer.tokenizer.tokenize(INSTRUCT_TEMPLATE))

    for i in range(0, len(texts), batch_size):
        batch_idx = (i // batch_size) + 1
        batch = texts[i:i+batch_size]
        all_ids_list = []
        is_query = []
        for text, char_types in batch:
            result = nova_tokenizer.encode("", text, char_types)
            ids = result["input_ids"][:max_len]
            all_ids_list.append(ids)
            is_query.append(text.startswith(INSTRUCT_TEMPLATE))

        max_len_batch = max(len(x) for x in all_ids_list)
        pad_ids = np.full((len(batch), max_len_batch), pad_id, dtype=np.int64)
        for j, ids in enumerate(all_ids_list):
            pad_ids[j, :len(ids)] = ids

        input_ids = torch.tensor(pad_ids, device=device)
        key_padding_mask = (input_ids == pad_id)

        pool_mask = key_padding_mask.clone()
        for j, query in enumerate(is_query):
            if query:
                excl = min(instruct_token_len, len(all_ids_list[j]))
                pool_mask[j, :excl] = True

        if profiler is not None:
            with profiler:
                hidden = student(input_ids, key_padding_mask=key_padding_mask)
                emb = lal_head(hidden, key_padding_mask=key_padding_mask, pool_mask=pool_mask)
            profiler.total_samples += len(batch)
        else:
            hidden = student(input_ids, key_padding_mask=key_padding_mask)
            emb = lal_head(hidden, key_padding_mask=key_padding_mask, pool_mask=pool_mask)

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


def ensure_outputs_do_not_exist(paths):
    existing = [path for path in paths if os.path.exists(path)]
    if existing:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing eval output(s):\n"
            f"{formatted}\n"
            "Set EVAL_TAG to a new unique label and rerun."
        )


def main():
    global STUDENT_DIR, EVAL_LABEL, EVAL_SUFFIX
    global device, nova_tokenizer, pad_id, student, lal_head

    parser = argparse.ArgumentParser(
        description="Evaluate the Nova student on the bench split.",
        epilog="Set STUDENT_DIR, EVAL_DATA, EVAL_TAG, and EVAL_OUTPUT_DIR as needed.",
    )
    parser.parse_args()

    if STUDENT_DIR is None:
        for _subdir in ("student_best", "student_final"):
            _candidate = f"./model_checkpoints/nova_student_bench/{_subdir}"
            if os.path.exists(_candidate):
                STUDENT_DIR = _candidate
                break
        else:
            raise RuntimeError(
                "No default student checkpoint found. Set STUDENT_DIR to a directory "
                "containing student_config.json, student_model.pt, and lal_head.pt."
            )

    EVAL_LABEL = _slug(EVAL_TAG) if EVAL_TAG else _default_eval_label(STUDENT_DIR)
    EVAL_SUFFIX = f"_{EVAL_LABEL}" if EVAL_LABEL else ""

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

    student.load_state_dict(torch.load(_ckpt("student_model.pt"), map_location="cpu"))
    lal_head.load_state_dict(torch.load(_ckpt("lal_head.pt"), map_location="cpu"))

    student.eval()
    lal_head.eval()
    print("Loaded student and lal head")

    test_samples = load_jsonl(TEST_PATH)
    print("Loaded test samples")

    set_seed(42)

    emb_dir = os.environ.get("EVAL_OUTPUT_DIR") or get_embeddings_dir("binarycorp_bench", "nova_student")
    os.makedirs(emb_dir, exist_ok=True)
    results_path = os.path.join(emb_dir, f"eval_bench_embeddings{EVAL_SUFFIX}.pt")
    metrics_path = os.path.join(emb_dir, f"eval_student_bench_metrics{EVAL_SUFFIX}.json")
    ensure_outputs_do_not_exist([results_path, metrics_path])

    eval_pairs = build_eval_pairs(test_samples, seed=42)
    results = extract_student_embeddings(eval_pairs, batch_size=32, max_len=1024)

    torch.save(results, results_path)
    print(f"Saved embeddings to {results_path}")

    print("\n" + "="*50)
    print(f"NOVA STUDENT | Latency: {results['stats']['avg_ms_per_sample']:.2f} ms/function")
    print(f"NOVA STUDENT | Memory:  {results['stats']['peak_memory_mb']:.1f} MB peak")
    print("="*50)

    report = compute_report(results)
    report["source"] = os.path.abspath(results_path)
    report["model"] = f"nova_student{EVAL_SUFFIX}"
    report["data"] = os.path.abspath(TEST_PATH)

    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full report to {metrics_path}")

    print_report_summary(report)


if __name__ == "__main__":
    main()
