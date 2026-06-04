import os
import sys

# Setup paths to import shared modules and metrics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from metrics import EvaluationEngine
from shared.data_utils import set_seed, load_jsonl as load_binarycorp_jsonl, get_embeddings_dir
from shared.nova_utils import setup_nova_tokenizer, load_nova
from shared.pooling import AttentionPooling
from shared.profiling import InferenceProfiler

import torch
import numpy as np
import json
import gc
from transformers import AutoTokenizer
from peft import PeftModel
import random

MODEL_ID = "lt-asset/nova-1.3b"

set_seed(42)

# Load tokenizer and base model using shared utilities
base_tokenizer, nova_tokenizer, MASK_ID = setup_nova_tokenizer()

# Load base model
base_model = load_nova(device_id=0, base_tokenizer=base_tokenizer)

script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 3 levels to repository root where the jsonl files are located
script_dir_repo = os.path.abspath(os.path.join(script_dir, "../../../"))

eval_samples = load_binarycorp_jsonl(os.path.join(script_dir_repo, "binarycorp3m_test_nova.jsonl"))

# Load LoRA adapter
output_dir = os.path.join(script_dir, "nova_contrastive_bidir_noMNTP_final")
model = PeftModel.from_pretrained(base_model, output_dir)
model.eval()

device = next(model.parameters()).device

# Load Pooling Head
pooling_head = AttentionPooling(model.config.hidden_size).to(device).to(torch.bfloat16)
pooling_head.load_state_dict(torch.load(os.path.join(output_dir, "pooling_head.pt")))
pooling_head.eval()

@torch.no_grad()
def extract_embeddings_smart(model, tokenizer, pooling_module, samples, batch_size=16, device="cuda"):
    model.eval()
    pooling_module.eval()

    label_ids = tokenizer.labels
    base_tokenizer = tokenizer.tokenizer

    all_embeddings = []
    all_ids = []
    all_opts = []

    profiler = InferenceProfiler(device)
    is_warmup = True

    print("extracting embeddings...")

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]

        batch_input_ids = []
        batch_masks = []
        batch_label_positions = []

        INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "

        for sample in batch_samples:
            if sample["opt"] == "O0":
                text = INSTRUCT_TEMPLATE + sample["asm"]
                char_types = "0" * len(INSTRUCT_TEMPLATE) + "1" * len(sample["asm"])
            else:
                text = sample["asm"]
                char_types = "1" * len(text)

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
        padded_ids = np.full((len(batch_samples), max_len), pad_id, dtype=np.int64)
        padded_masks = np.zeros((len(batch_samples), max_len, max_len), dtype=np.float32)

        for j, (ids, mask) in enumerate(zip(batch_input_ids, batch_masks)):
            L = len(ids)
            padded_ids[j, :L] = ids
            padded_masks[j, :L, :L] = mask

        input_ids_t = torch.tensor(padded_ids, dtype=torch.long, device=device)
        nova_mask_t = torch.tensor(padded_masks, dtype=torch.bfloat16, device=device)

        with profiler:
            outputs = model(input_ids=input_ids_t, nova_attention_mask=nova_mask_t, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            pooled = pooling_module(hidden, batch_label_positions)
        profiler.total_samples += len(batch_samples)

        # Exclude CUDA init from profiler by resetting after first batch
        if is_warmup:
            profiler.total_time_ms = 0.0
            profiler.total_samples = 0
            is_warmup = False

        all_embeddings.append(pooled.cpu())

        all_ids.extend([s["id"] for s in batch_samples])
        all_opts.extend([s["opt"] for s in batch_samples])

        if i % 500 == 0:
            print(f"Extracted {i}/{len(samples)}")

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    return {"ids": all_ids, "opts": all_opts, "embeddings": embeddings_tensor, "stats": profiler.get_stats()}


eval_result = extract_embeddings_smart(
    model=model,
    tokenizer=nova_tokenizer,
    pooling_module=pooling_head,
    samples=eval_samples,
    batch_size=16,
    device=device
)

emb_dir = get_embeddings_dir("binarycorp3m", "nova_teacher")
save_name = os.path.join(emb_dir, "eval_embeddings.pt")
torch.save(eval_result, save_name)
print(f"embeddings shape: {eval_result['embeddings'].shape}")
print(f"Saved to: {save_name}")

print("\n" + "="*50)
print(f"NOVA TEACHER | Latency: {eval_result['stats']['avg_ms_per_sample']:.2f} ms/function")
print(f"NOVA TEACHER | Memory:  {eval_result['stats']['peak_memory_mb']:.1f} MB peak")
print("="*50)

engine = EvaluationEngine(device=device)
report = engine.evaluate(
    results_dict=eval_result,
    pool_sizes=[50, 100, 200, 500, "global"],
    k_list=[1, 5, 10],
    num_trials=100
)

for pool, metrics in report.items():
    print(f"\n[{pool}]")
    print(f"NDCG@10:  {metrics['NDCG@10']:.4f}")
    print(f"Recall@1: {metrics['Recall@1']:.4f}")
    print(f"MRR:      {metrics['MRR']:.4f}")
