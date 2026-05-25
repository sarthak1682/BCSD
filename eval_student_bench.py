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
from eval_bench import set_seed, load_jsonl, build_eval_pairs, compute_report, print_report_summary

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except Exception:
    TSNE = None
    HAS_SKLEARN = False


CACHE_DIR = "/home/ra72yeq/.cache/huggingface/hub/models--lt-asset--nova-1.3b/snapshots/4b4805bac4f13ef8bec678072ef60609ea3b0e77"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(PROJECT_DIR, "nvemb", "output_benchset_rebalanced_test_nova.jsonl")
INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
RUN_ID = 20
STUDENT_DIR = os.path.join(PROJECT_DIR, "nova_distilled_student_20_bench")
RESULTS_PATH = os.path.join(PROJECT_DIR, f"eval_student_bench_results_{RUN_ID}.pt")
TSNE_PATH = os.path.join(PROJECT_DIR, f"eval_student_bench_tsne_{RUN_ID}.png")
TSNE_NUM_PAIRS = 40

device = "cuda" if torch.cuda.is_available() else "cpu"


sys.path.insert(0, CACHE_DIR)
from modeling_nova import NovaTokenizer  # re-use your local NovaTokenizer

base_tokenizer = AutoTokenizer.from_pretrained("lt-asset/nova-1.3b", cache_dir=CACHE_DIR)
base_tokenizer.add_special_tokens({'additional_special_tokens': ['[MASK]']})
nova_tokenizer = NovaTokenizer(base_tokenizer)

print("Loaded tokenizer")

class LatentAttentionLayer(nn.Module):
    """Distills sequence into a single embedding via Perceiver-style cross-attention."""
    def __init__(self, hidden_dim, num_latents=512, num_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))

        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, hidden_states, key_padding_mask=None):
        batch_size = hidden_states.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        normed_latents = self.attn_norm(latents)
        attn_output, _ = self.cross_attn(
            query=normed_latents,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        latents = latents + attn_output
        mlp_out = self.mlp(self.mlp_norm(latents))
        latents = latents + mlp_out

        return latents.mean(dim=1)

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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids, key_padding_mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        if key_padding_mask is None:
            key_padding_mask = (input_ids == self.pad_id)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.out_proj(x)

print("Loaded StudentDistillationModule")

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
def encode_student_texts(texts, batch_size=32, max_len=1024):
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

    print("Encoding query embeddings...")
    query_embeddings = encode_student_texts(query_texts, batch_size=batch_size, max_len=max_len)
    print("Encoding target embeddings...")
    target_embeddings = encode_student_texts(target_texts, batch_size=batch_size, max_len=max_len)

    return {
        "ids": all_ids,
        "query_opts": query_opts,
        "target_opts": target_opts,
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
    }




def main():
    set_seed(42)

    eval_pairs = build_eval_pairs(test_samples, seed=42)
    results = extract_student_embeddings(eval_pairs, batch_size=32, max_len=1024)
    torch.save(results, RESULTS_PATH)
    print(f"Saved embeddings to {RESULTS_PATH}")

    #plot_tsne_pairs(results, num_pairs=TSNE_NUM_PAIRS, out_path=TSNE_PATH)

    report = compute_report(results)
    report["source"] = os.path.abspath(RESULTS_PATH)
    report["model"] = f"nova_student_run{RUN_ID}"
    report["data"] = os.path.abspath(TEST_PATH)

    metrics_path = os.path.join(PROJECT_DIR, f"eval_student_bench_metrics_{RUN_ID}.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full report to {metrics_path}")

    print_report_summary(report)


if __name__ == "__main__":
    main()