import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import math

# Setup paths to import shared modules and metrics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from metrics import EvaluationEngine
from shared.student_model import StudentDistillationModule, LatentAttentionLayer, PositionalEncoding
from shared.nova_utils import setup_nova_tokenizer

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except Exception:
    TSNE = None
    HAS_SKLEARN = False

# ---- 1) Paths / config ----
CACHE_DIR = "/home/ra72yeq/.cache/huggingface/hub/models--lt-asset--nova-1.3b/snapshots/4b4805bac4f13ef8bec678072ef60609ea3b0e77"
script_dir_file = os.path.dirname(os.path.abspath(__file__))
# Go up 3 levels to repository root where the jsonl files are located
script_dir_repo = os.path.abspath(os.path.join(script_dir_file, "../../../"))

TEST_PATH = os.path.join(script_dir_repo, "binarycorp3m_test_nova.jsonl")
INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
RUN_ID = 10
STUDENT_DIR = os.path.join(script_dir_file, "nova_distilled_student_new_10")
if not os.path.exists(STUDENT_DIR):
    STUDENT_DIR = os.path.join(script_dir_file, "nova_distilled_student_20")

RESULTS_PATH = os.path.join(script_dir_file, f"eval_student_results_new_{RUN_ID}.pt")
TSNE_PATH = os.path.join(script_dir_file, f"eval_student_tsne_nvembed_{RUN_ID}.png")
TSNE_NUM_PAIRS = 40

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 2) Load tokenizer ----
base_tokenizer, nova_tokenizer, MASK_ID = setup_nova_tokenizer()
print("Loaded tokenizer")

# ---- 4) Load student weights ----
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

# ---- 5) Load test data ----
def load_binarycorp_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

test_samples = load_binarycorp_jsonl(TEST_PATH)
print("Loaded test samples")

# ---- 6) Extract embeddings ----
@torch.no_grad()
def extract_student_embeddings(samples, batch_size=32, max_len=512):
    all_embs, all_ids, all_opts = [], [], []
    total_batches = (len(samples) + batch_size - 1) // batch_size

    for i in range(0, len(samples), batch_size):
        batch_idx = (i // batch_size) + 1
        batch = samples[i:i+batch_size]
        # tokenize
        all_ids_list = []
        for s in batch:
            if s.get('opt', 'O0') == 'O0':
                text = INSTRUCT_TEMPLATE + s["asm"]
                char_types = "0" * len(INSTRUCT_TEMPLATE) + "1" * len(s["asm"])
            else:
                text = s["asm"]
                char_types = "1" * len(text)
                
            result = nova_tokenizer.encode("", text, char_types)
            ids = result["input_ids"][:max_len]
            all_ids_list.append(ids)

        max_len_batch = max(len(x) for x in all_ids_list)
        pad_ids = np.full((len(batch), max_len_batch), pad_id, dtype=np.int64)
        for j, ids in enumerate(all_ids_list):
            pad_ids[j, :len(ids)] = ids

        input_ids = torch.tensor(pad_ids, device=device)
        key_padding_mask = (input_ids == pad_id)

        hidden = student(input_ids, key_padding_mask=key_padding_mask)
        emb = lal_head(hidden, key_padding_mask=key_padding_mask)
        all_embs.append(emb.float().cpu())

        all_ids.extend([s["id"] for s in batch])
        all_opts.extend([s["opt"] for s in batch])

        if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches:
            print(f"Processed batch {batch_idx}/{total_batches} ({min(i+batch_size, len(samples))}/{len(samples)} samples)")

    return {
        "ids": all_ids,
        "opts": all_opts,
        "embeddings": torch.cat(all_embs, dim=0)
    }

# ---- 8) Run ----
results = extract_student_embeddings(test_samples, batch_size=32, max_len=1024)
torch.save(results, RESULTS_PATH)
print(f"Saved embeddings to {RESULTS_PATH}")

engine = EvaluationEngine(device=device)
report = engine.evaluate(
    results_dict=results,
    pool_sizes=[50, 100, 200, 500, "global"],
    k_list=[1, 5, 10],
    num_trials=100
)

for pool, metrics in report.items():
    print(f"\n[{pool}]")
    print(f"NDCG@10:  {metrics['NDCG@10']:.4f}")
    print(f"Recall@1: {metrics['Recall@1']:.4f}")
    print(f"MRR:      {metrics['MRR']:.4f}")
