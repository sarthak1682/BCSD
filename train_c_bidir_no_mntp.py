import os


import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
import sys
import random
import json
from tqdm import tqdm
import gc
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

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
MASK_ID = base_tokenizer.encode('[MASK]')[-1]

nova_tokenizer = NovaTokenizer(base_tokenizer)


def make_bidirectional_nova_mask(nova_mask):
    return np.maximum(nova_mask, nova_mask.T)


model = NovaForCausalLM.from_pretrained(
    cache_dir,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)
model.resize_token_embeddings(len(base_tokenizer))

from peft import LoraConfig, get_peft_model, TaskType

contrastive_lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_dora=True,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, contrastive_lora_config)
model.print_trainable_parameters()

device = next(model.parameters()).device

class AttentionPooling(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, hidden_states, label_positions):
        pooled_outputs = []
        for i, pos_list in enumerate(label_positions):
            valid_pos = [p for p in pos_list if p < hidden_states.shape[1]]

            if not valid_pos:
                pooled_outputs.append(hidden_states[i].mean(dim=0))
                continue

            inst_vectors = hidden_states[i, valid_pos, :]
            attn_scores = self.attention(inst_vectors)
            attn_weights = torch.softmax(attn_scores, dim=0)
            weighted_avg = torch.sum(inst_vectors * attn_weights, dim=0)
            pooled_outputs.append(weighted_avg)

        return torch.stack(pooled_outputs)


def load_binarycorp_jsonl(path: str):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return data

script_dir = os.path.dirname(os.path.abspath(__file__))

train_samples = load_binarycorp_jsonl(os.path.join(script_dir, "binarycorp_train_nova.jsonl"))
eval_samples = load_binarycorp_jsonl(os.path.join(script_dir, "binarycorp_test_nova.jsonl"))

class PairCollator:
    def __init__(self, nova_tokenizer, max_length=1024):
        self.nova_tokenizer = nova_tokenizer
        self.max_length = max_length
        self.label_ids = nova_tokenizer.labels
        self.pad_id = nova_tokenizer.tokenizer.pad_token_id or 0

    def __call__(self, batch):
        flat_texts = []
        for p in batch:
            flat_texts.extend([p[0], p[1]])

        all_ids, all_masks, all_label_positions = [], [], []

        for text in flat_texts:
            result = self.nova_tokenizer.encode("", text, "1" * len(text))

            ids = result['input_ids'][:self.max_length]

            raw_mask = result['nova_attention_mask']
            L = len(ids)
            mask = np.maximum(raw_mask[:L, :L], raw_mask[:L, :L].T)

            label_pos = [i for i, tid in enumerate(ids) if tid in self.label_ids]

            all_ids.append(ids)
            all_masks.append(mask)
            all_label_positions.append(label_pos)

        max_len = max(len(x) for x in all_ids)

        pad_ids = np.full((len(flat_texts), max_len), self.pad_id, dtype=np.int64)
        pad_masks = np.zeros((len(flat_texts), max_len, max_len), dtype=np.float32)

        for i, (ids, mask) in enumerate(zip(all_ids, all_masks)):
            L = len(ids)
            pad_ids[i, :L] = ids
            pad_masks[i, :L, :L] = mask

        return {
            "input_ids": torch.tensor(pad_ids),
            "nova_attention_mask": torch.tensor(pad_masks),
            "label_positions": all_label_positions
        }


def contrastive_loss(embeddings, temperature=0.05, hard_negative_weight=2.0):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    batch_size = embeddings.shape[0]

    labels = torch.arange(batch_size, device=embeddings.device)
    labels[::2] += 1
    labels[1::2] -= 1

    mask_self = torch.eye(batch_size, device=embeddings.device).bool()
    sim_matrix.masked_fill_(mask_self, -1e9)

    mask_pos = torch.zeros_like(sim_matrix).bool()
    mask_pos.scatter_(1, labels.unsqueeze(1), 1)

    with torch.no_grad():
        neg_matrix = sim_matrix.clone()
        neg_matrix.masked_fill_(mask_pos, -1e9)
        neg_matrix.masked_fill_(mask_self, -1e9)

        k = min(3, batch_size - 2)
        topk_values, topk_indices = torch.topk(neg_matrix, k=k, dim=1)

    weights = torch.ones_like(sim_matrix)
    hard_weight_tensor = torch.full_like(topk_values, hard_negative_weight)
    weights.scatter_(1, topk_indices, hard_weight_tensor)

    sim_matrix = sim_matrix * weights

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "


pairs = []
o0_dict = {s['id']: s for s in train_samples if s['opt'] == 'O0'}
o3_dict = {s['id']: s for s in train_samples if s['opt'] == 'O3'}

for func_id in o0_dict:
    if func_id in o3_dict:
        pairs.append((INSTRUCT_TEMPLATE + o0_dict[func_id]['asm'], o3_dict[func_id]['asm']))

print(f"pairs: {len(pairs)}")

batch_size = 32
grad_accum = 4
lr = 3e-5
num_epochs = 2

pair_collator = PairCollator(nova_tokenizer, max_length=512)
pair_loader = DataLoader(pairs, batch_size=batch_size, shuffle=True, collate_fn=pair_collator)

pooling_head = AttentionPooling(model.config.hidden_size).to(device).to(torch.bfloat16)
pooling_head.train()

optimizer = AdamW(list(model.parameters()) + list(pooling_head.parameters()), lr=lr)

total_steps = len(pair_loader) // grad_accum * num_epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.03 * total_steps), total_steps)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False
model.train()
torch.cuda.empty_cache()
gc.collect()

for epoch in range(num_epochs):
    epoch_losses = []
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(pair_loader, desc=f"Epoch {epoch+1}")):
        batch_inputs = {k: v.to(device) for k, v in batch.items() if k != 'label_positions'}
        label_pos = batch['label_positions']

        outputs = model(**batch_inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]

        embeddings = pooling_head(hidden, label_pos)
        loss = contrastive_loss(embeddings)

        loss = loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_losses.append(loss.item() * grad_accum)

        if step % 100 == 0 and step > 0:
            avg = sum(epoch_losses[-50:]) / min(50, len(epoch_losses))
            print(f"Step {step}: avg loss = {avg:.4f}")

    print(f"Epoch {epoch+1}: avg loss = {sum(epoch_losses)/len(epoch_losses):.4f}")


output_dir = os.path.join(script_dir, "nova_contrastive_bidir_noMNTP_final")
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
torch.save(pooling_head.state_dict(), os.path.join(output_dir, "pooling_head.pt"))
print(f"saved to {output_dir}")
