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

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except Exception:
    TSNE = None
    HAS_SKLEARN = False

# ---- 1) Paths / config ----
CACHE_DIR = "/home/ra72yeq/.cache/huggingface/hub/models--lt-asset--nova-1.3b/snapshots/4b4805bac4f13ef8bec678072ef60609ea3b0e77"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(PROJECT_DIR, "binarycorp3m_test_nova.jsonl")
INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
RUN_ID = 10
STUDENT_DIR = os.path.join(PROJECT_DIR, "nova_distilled_student_new_10")
RESULTS_PATH = os.path.join(PROJECT_DIR, f"eval_student_results_new_{RUN_ID}.pt")
TSNE_PATH = os.path.join(PROJECT_DIR, f"eval_student_tsne_nvembed_{RUN_ID}.png")
TSNE_NUM_PAIRS = 40

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 2) Load tokenizer ----
sys.path.insert(0, CACHE_DIR)
from modeling_nova import NovaTokenizer  # re-use your local NovaTokenizer

base_tokenizer = AutoTokenizer.from_pretrained("lt-asset/nova-1.3b", cache_dir=CACHE_DIR)
base_tokenizer.add_special_tokens({'additional_special_tokens': ['[MASK]']})
nova_tokenizer = NovaTokenizer(base_tokenizer)

print("Loaded tokenizer")
# ---- 3) Student + LAL definitions (copy from distill_nova.py) ----
class LatentAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_latents=512, num_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, key_padding_mask=None):
        batch_size = hidden_states.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        # Q=sequence, K/V=latents — NV-Embed dictionary style
        attn_output, _ = self.cross_attn(query=hidden_states, key=latents, value=latents)
        output = self.mlp(self.layer_norm(attn_output))
        # Masked mean pool over sequence length (exclude padding)
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).to(output.dtype)
            return (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return output.mean(dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

import math
class StudentDistillationModule(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers=2, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=1024)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids, key_padding_mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        if key_padding_mask is None:
            key_padding_mask = (input_ids == self.pad_id)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.out_proj(x)

print("Loaded StudentDistillationModule")
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
        texts = [INSTRUCT_TEMPLATE + s["asm"] for s in batch]

        # tokenize
        all_ids_list = []
        for text in texts:
            result = nova_tokenizer.encode("", text, "1" * len(text))
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

# ---- 7) Metrics (reuse from train_contrastive.py) ----
def compute_recall_at_k(result, k=1):
    ids = result["ids"]
    opts = result["opts"]
    embs = result["embeddings"].float()
    embs = embs / embs.norm(dim=1, keepdim=True)

    o0_idx = [i for i, o in enumerate(opts) if o == "O0"]
    o3_idx = [i for i, o in enumerate(opts) if o == "O3"]
    o0_ids = [ids[i] for i in o0_idx]
    o3_ids = [ids[i] for i in o3_idx]

    o0_embs = embs[o0_idx]
    o3_embs = embs[o3_idx]

    o3_id_to_idx = {id_: i for i, id_ in enumerate(o3_ids)}
    sim = o0_embs @ o3_embs.T

    correct = 0
    for i, o0_id in enumerate(o0_ids):
        if o0_id not in o3_id_to_idx:
            continue
        correct_o3_idx = o3_id_to_idx[o0_id]
        top_k = sim[i].topk(k).indices.tolist()
        if correct_o3_idx in top_k:
            correct += 1
    return correct / len(o0_ids)

def compute_recall_at_k_pooled(result, pool_size=50, num_trials=100):
    ids = result["ids"]
    opts = result["opts"]
    embs = F.normalize(result["embeddings"].float(), p=2, dim=1)

    o0_idx = [i for i, o in enumerate(opts) if o == "O0"]
    o3_idx = [i for i, o in enumerate(opts) if o == "O3"]
    o0_ids = [ids[i] for i in o0_idx]
    o3_ids = [ids[i] for i in o3_idx]

    paired_ids = list(set(o0_ids) & set(o3_ids))
    total_correct, total_queries = 0, 0

    for _ in range(num_trials):
        if len(paired_ids) < pool_size:
            sampled = paired_ids
        else:
            sampled = np.random.choice(paired_ids, pool_size, replace=False)
        s_o0_idx = [o0_idx[o0_ids.index(fid)] for fid in sampled]
        s_o3_idx = [o3_idx[o3_ids.index(fid)] for fid in sampled]
        emb_o0 = embs[s_o0_idx]
        emb_o3 = embs[s_o3_idx]
        sim = emb_o0 @ emb_o3.T
        preds = sim.argmax(dim=1).tolist()
        correct = [1 if preds[i] == i else 0 for i in range(len(preds))]
        total_correct += sum(correct)
        total_queries += len(correct)
    return total_correct / total_queries

def plot_tsne_pairs(result, num_pairs=20, out_path="tsne.png", seed=42):
    if not HAS_SKLEARN:
        print("scikit-learn not available; skipping t-SNE plot.")
        return

    ids = result["ids"]
    opts = result["opts"]
    embs = result["embeddings"].float()
    embs = F.normalize(embs, p=2, dim=1).cpu().numpy()

    o0_idx = [i for i, o in enumerate(opts) if o == "O0"]
    o3_idx = [i for i, o in enumerate(opts) if o == "O3"]
    o0_ids = [ids[i] for i in o0_idx]
    o3_ids = [ids[i] for i in o3_idx]

    paired_ids = list(set(o0_ids) & set(o3_ids))
    if not paired_ids:
        print("No paired O0/O3 ids found; skipping t-SNE plot.")
        return

    if len(paired_ids) > num_pairs:
        rng = np.random.default_rng(seed)
        paired_ids = rng.choice(paired_ids, num_pairs, replace=False).tolist()

    s_o0_idx = [o0_idx[o0_ids.index(fid)] for fid in paired_ids]
    s_o3_idx = [o3_idx[o3_ids.index(fid)] for fid in paired_ids]
    sel_idx = s_o0_idx + s_o3_idx

    tsne = TSNE(n_components=2, init="random", random_state=seed, perplexity=min(30, len(sel_idx) - 1))
    coords = tsne.fit_transform(embs[sel_idx])

    plt.figure(figsize=(8, 6))
    for i, fid in enumerate(paired_ids):
        o0_point = coords[i]
        o3_point = coords[i + len(paired_ids)]
        plt.scatter(o0_point[0], o0_point[1], marker="o")
        plt.scatter(o3_point[0], o3_point[1], marker="x")
        plt.plot([o0_point[0], o3_point[0]], [o0_point[1], o3_point[1]], linestyle="--", linewidth=0.8)

    plt.title(f"t-SNE of Student Embeddings ({len(paired_ids)} pairs - O0 vs O3)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved t-SNE plot to {out_path}")

# ---- 8) Run ----
results = extract_student_embeddings(test_samples, batch_size=32, max_len=512)
torch.save(results, RESULTS_PATH)
print(f"Saved embeddings to {RESULTS_PATH}")

#plot_tsne_pairs(results, num_pairs=TSNE_NUM_PAIRS, out_path=TSNE_PATH)

print("--- Pooled (Paper Metric) ---")
print(f"Recall@1 (Pool 50):  {compute_recall_at_k_pooled(results, pool_size=50):.4f}")
print(f"Recall@1 (Pool 100): {compute_recall_at_k_pooled(results, pool_size=100):.4f}")
print(f"Recall@1 (Pool 200): {compute_recall_at_k_pooled(results, pool_size=200):.4f}")
print(f"Recall@1 (Pool 500): {compute_recall_at_k_pooled(results, pool_size=500):.4f}")

print("\n--- Global (Full Test Set) ---")
print(f"Recall@1:  {compute_recall_at_k(results, k=1):.4f}")
print(f"Recall@5:  {compute_recall_at_k(results, k=5):.4f}")
print(f"Recall@10: {compute_recall_at_k(results, k=10):.4f}")