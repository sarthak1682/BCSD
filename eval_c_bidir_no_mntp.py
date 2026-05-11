import os
import torch
import numpy as np
import json
import gc
from transformers import AutoTokenizer
from peft import PeftModel
import sys
import random

MODEL_ID = "lt-asset/nova-1.3b"

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}.")

set_seed(42)

cache_dir = "/home/ra72yeq/.cache/huggingface/hub/models--lt-asset--nova-1.3b/snapshots/4b4805bac4f13ef8bec678072ef60609ea3b0e77"
sys.path.insert(0, cache_dir)

from modeling_nova import NovaForCausalLM, NovaTokenizer

base_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_tokenizer.add_special_tokens({'additional_special_tokens': ['[MASK]']})
nova_tokenizer = NovaTokenizer(base_tokenizer)

class AttentionPooling(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, hidden_states, label_positions):
        # hidden_states: [B, L, D]
        B, L, D = hidden_states.shape
        device = hidden_states.device

        # [B, L, 1] -> [B, L]
        attn_scores = self.attention(hidden_states).squeeze(-1)

        mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        for i, pos_list in enumerate(label_positions):
            valid_pos = [p for p in pos_list if p < L]
            if valid_pos:
                mask[i, valid_pos] = True
            else:
                # Fallback: pool over the whole sequence
                mask[i, :] = True

        # Mask out invalid positions
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # Softmax over sequence length
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, L, 1]

        # Weighted sum: [B, L, D] * [B, L, 1] -> [B, D]
        pooled_outputs = torch.sum(hidden_states * attn_weights, dim=1)
        return pooled_outputs

def load_binarycorp_jsonl(path: str):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return data

script_dir = os.path.dirname(os.path.abspath(__file__))
eval_samples = load_binarycorp_jsonl(os.path.join(script_dir, "binarycorp_test_nova.jsonl"))

# Load base model
base_model = NovaForCausalLM.from_pretrained(
    cache_dir,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)
base_model.resize_token_embeddings(len(base_tokenizer))

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

            ids = result['input_ids'][:512]
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

        outputs = model(input_ids=input_ids_t, nova_attention_mask=nova_mask_t, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]

        pooled = pooling_module(hidden, batch_label_positions)
        all_embeddings.append(pooled.cpu())

        all_ids.extend([s["id"] for s in batch_samples])
        all_opts.extend([s["opt"] for s in batch_samples])

        if i % 500 == 0:
            print(f"Extracted {i}/{len(samples)}")

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    return {"ids": all_ids, "opts": all_opts, "embeddings": embeddings_tensor}


def compute_recall_at_k(result, k=1):
    ids = result['ids']
    opts = result['opts']
    embs = result['embeddings'].float()

    embs = embs / embs.norm(dim=1, keepdim=True)

    o0_idx = [i for i, o in enumerate(opts) if o == 'O0']
    o3_idx = [i for i, o in enumerate(opts) if o == 'O3']

    o0_ids = [ids[i] for i in o0_idx]
    o3_ids = [ids[i] for i in o3_idx]

    o0_embs = embs[o0_idx]
    o3_embs = embs[o3_idx]

    o3_id_to_idx = {id_: i for i, id_ in enumerate(o3_ids)}

    sim_matrix = o0_embs @ o3_embs.T

    correct = 0
    for i, o0_id in enumerate(o0_ids):
        if o0_id not in o3_id_to_idx:
            continue
        correct_o3_idx = o3_id_to_idx[o0_id]
        top_k_indices = sim_matrix[i].topk(k).indices.tolist()
        if correct_o3_idx in top_k_indices:
            correct += 1

    return correct / len(o0_ids)


def compute_recall_at_k_pooled(result, k_recall=1, pool_size=50, num_trials=10):
    ids = result['ids']
    opts = result['opts']
    embs = result['embeddings'].float()

    embs = embs / embs.norm(dim=1, keepdim=True)

    o0_idx = [i for i, o in enumerate(opts) if o == 'O0']
    o3_idx = [i for i, o in enumerate(opts) if o == 'O3']

    o0_ids = [ids[i] for i in o0_idx]
    o3_ids = [ids[i] for i in o3_idx]

    paired_ids = list(set(o0_ids) & set(o3_ids))

    total_correct = 0
    total_queries = 0

    for trial in range(num_trials):
        if len(paired_ids) < pool_size:
            sampled_ids = paired_ids
        else:
            sampled_ids = np.random.choice(paired_ids, pool_size, replace=False)

        sampled_o0_idx = [o0_idx[o0_ids.index(fid)] for fid in sampled_ids if fid in o0_ids]
        sampled_o3_idx = [o3_idx[o3_ids.index(fid)] for fid in sampled_ids if fid in o3_ids]

        o0_embs_pool = embs[sampled_o0_idx]
        o3_embs_pool = embs[sampled_o3_idx]

        sim_matrix = o0_embs_pool @ o3_embs_pool.T

        for i in range(len(sampled_ids)):
            top_k_indices = sim_matrix[i].topk(k_recall).indices.tolist()
            if i in top_k_indices:
                total_correct += 1
            total_queries += 1

    return total_correct / total_queries


eval_result = extract_embeddings_smart(
    model=model,
    tokenizer=nova_tokenizer,
    pooling_module=pooling_head,
    samples=eval_samples,
    batch_size=16,
    device=device
)

save_name = os.path.join(script_dir, "embeddings_bidir_contrastive_noMNTP_test.pt")
torch.save(eval_result, save_name)
print(f"embeddings shape: {eval_result['embeddings'].shape}")

print("recall (pooled):")
r1_k50  = compute_recall_at_k_pooled(eval_result, k_recall=1, pool_size=50,  num_trials=100)
r1_k100 = compute_recall_at_k_pooled(eval_result, k_recall=1, pool_size=100, num_trials=100)
r1_k200 = compute_recall_at_k_pooled(eval_result, k_recall=1, pool_size=200, num_trials=100)
r1_k500 = compute_recall_at_k_pooled(eval_result, k_recall=1, pool_size=500, num_trials=100)

print(f"Recall@1 (K=50):  {r1_k50:.4f}")
print(f"Recall@1 (K=100): {r1_k100:.4f}")
print(f"Recall@1 (K=200): {r1_k200:.4f}")
print(f"Recall@1 (K=500): {r1_k500:.4f}")

print("recall (full):")
recall_1  = compute_recall_at_k(eval_result, k=1)
recall_5  = compute_recall_at_k(eval_result, k=5)
recall_10 = compute_recall_at_k(eval_result, k=10)

print(f"Recall@1:  {recall_1:.4f}")
print(f"Recall@5:  {recall_5:.4f}")
print(f"Recall@10: {recall_10:.4f}")
