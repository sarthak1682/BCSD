import os
import torch
import numpy as np
import json
import gc
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer
from peft import PeftModel
import sys
import random

# Setup paths to import shared modules and evaluation script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../")))

from eval_bench import set_seed, load_jsonl, build_eval_pairs, compute_report, print_report_summary
from shared.nova_utils import setup_nova_tokenizer, load_nova
from shared.pooling import AttentionPooling
from shared.data_utils import get_embeddings_dir
from shared.profiling import InferenceProfiler

MODEL_ID = "lt-asset/nova-1.3b"
OPT_LEVELS = ("O0", "O1", "O2", "O3")
COMPILERS = ("clang", "gcc")
ARCHITECTURES = ("x86_64", "arm64", "mips64", "powerpc64")
VARIANTS = ("none", "all", "bogus", "flattening", "substitution")

set_seed(42)

cache_dir = "/home/ra72yeq/.cache/huggingface/hub/models--lt-asset--nova-1.3b/snapshots/4b4805bac4f13ef8bec678072ef60609ea3b0e77"

# Load tokenizer and base model using shared utilities
base_tokenizer, nova_tokenizer, MASK_ID = setup_nova_tokenizer()

repo_root = os.path.abspath(os.path.join(script_dir, "../../../"))
BENCH_TEST_PATH = os.path.join(repo_root, "nvemb", "output_benchset_rebalanced_test_nova.jsonl")
if not os.path.exists(BENCH_TEST_PATH):
    BENCH_TEST_PATH = os.path.join(repo_root, "output_benchset_rebalanced_test_nova.jsonl")

eval_samples = load_jsonl(BENCH_TEST_PATH)

# Load base model
base_model = load_nova(device_id=0, base_tokenizer=base_tokenizer)

# Load LoRA adapter
output_dir = os.path.join(script_dir, "nova_contrastive_bidir_noMNTP_bench")
model = PeftModel.from_pretrained(base_model, output_dir)
model.eval()

device = next(model.parameters()).device

# Load Pooling Head
pooling_head = AttentionPooling(model.config.hidden_size).to(device).to(torch.bfloat16)
pooling_head.load_state_dict(torch.load(os.path.join(output_dir, "pooling_head.pt")))
pooling_head.eval()


@torch.no_grad()
def encode_texts(model, tokenizer, pooling_module, texts, batch_size=32, device="cuda", profiler=None):
    model.eval()
    pooling_module.eval()

    label_ids = tokenizer.labels
    base_tokenizer = tokenizer.tokenizer

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        batch_input_ids = []
        batch_masks = []
        batch_label_positions = []

        for text, char_types in batch_texts:
            result = tokenizer.encode("", text, char_types)
            ids = result['input_ids'][:1024]
            raw_mask = result['nova_attention_mask']
            L = len(ids)
            mask = np.maximum(raw_mask[:L, :L], raw_mask[:L, :L].T)
            label_pos = [j for j, tid in enumerate(ids) if tid in label_ids]

            batch_input_ids.append(ids)
            batch_masks.append(mask)
            batch_label_positions.append(label_pos)

        max_len = max(len(x) for x in batch_input_ids)
        pad_id = base_tokenizer.pad_token_id or 0
        padded_ids = np.full((len(batch_texts), max_len), pad_id, dtype=np.int64)
        padded_masks = np.zeros((len(batch_texts), max_len, max_len), dtype=np.float32)

        for j, (ids, mask) in enumerate(zip(batch_input_ids, batch_masks)):
            L = len(ids)
            padded_ids[j, :L] = ids
            padded_masks[j, :L, :L] = mask

        input_ids_t = torch.tensor(padded_ids, dtype=torch.long, device=device)
        nova_mask_t = torch.tensor(padded_masks, dtype=torch.bfloat16, device=device)

        if profiler is not None:
            with profiler:
                outputs = model(input_ids=input_ids_t, nova_attention_mask=nova_mask_t, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                del outputs
                pooled = pooling_module(hidden, batch_label_positions)
            profiler.total_samples += len(batch_texts)
        else:
            outputs = model(input_ids=input_ids_t, nova_attention_mask=nova_mask_t, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            del outputs
            pooled = pooling_module(hidden, batch_label_positions)

        all_embeddings.append(pooled.cpu())

        if i % 500 == 0:
            print(f"Encoded {i}/{len(texts)}")

    return torch.cat(all_embeddings, dim=0)


def extract_bench_eval_embeddings(model, tokenizer, pooling_module, pairs, batch_size=32, device="cuda"):
    instruct_template = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
    query_texts = []
    target_texts = []
    ids = []
    query_opts = []
    target_opts = []

    for pair in pairs:
        query_asm = pair["query"]["asm"]
        target_asm = pair["target"]["asm"]
        query_text = instruct_template + query_asm
        query_texts.append((query_text, "0" * len(instruct_template) + "1" * len(query_asm)))
        target_texts.append((target_asm, "1" * len(target_asm)))
        ids.append(pair["id"])
        query_opts.append(pair["query"]["opt"])
        target_opts.append(pair["target"]["opt"])

    profiler = InferenceProfiler(device)

    # Warmup: run first batch outside profiler stats
    print("Warmup pass...")
    _ = encode_texts(model, tokenizer, pooling_module, query_texts[:batch_size],
                     batch_size=batch_size, device=device)

    print("Encoding query embeddings...")
    query_embeddings = encode_texts(model, tokenizer, pooling_module, query_texts,
                                    batch_size=batch_size, device=device, profiler=profiler)
    print("Encoding target embeddings...")
    target_embeddings = encode_texts(model, tokenizer, pooling_module, target_texts,
                                     batch_size=batch_size, device=device, profiler=profiler)

    return {
        "ids": ids,
        "query_opts": query_opts,
        "target_opts": target_opts,
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
        "stats": profiler.get_stats(),
    }


def main():
    eval_pairs = build_eval_pairs(eval_samples, seed=42)
    eval_result = extract_bench_eval_embeddings(
        model=model,
        tokenizer=nova_tokenizer,
        pooling_module=pooling_head,
        pairs=eval_pairs,
        batch_size=16,
        device=device
    )

    emb_dir = get_embeddings_dir("binarycorp_bench", "nova_teacher")
    save_name = os.path.join(emb_dir, "eval_bench_embeddings.pt")
    torch.save(eval_result, save_name)
    print(f"query embeddings shape: {eval_result['query_embeddings'].shape}")
    print(f"target embeddings shape: {eval_result['target_embeddings'].shape}")
    print(f"Saved to: {save_name}")

    print("\n" + "="*50)
    print(f"NOVA TEACHER | Latency: {eval_result['stats']['avg_ms_per_sample']:.2f} ms/function")
    print(f"NOVA TEACHER | Memory:  {eval_result['stats']['peak_memory_mb']:.1f} MB peak")
    print("="*50)

    report = compute_report(eval_result)
    report["source"] = os.path.abspath(save_name)
    report["model"] = "nova_teacher"
    report["data"] = os.path.abspath(BENCH_TEST_PATH)

    metrics_path = os.path.join(emb_dir, "eval_teacher_bench_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full report to {metrics_path}")

    print_report_summary(report)


if __name__ == "__main__":
    main()
