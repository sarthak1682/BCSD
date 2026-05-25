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
from collections import defaultdict
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
        # hidden_states: [B, L, D]
        B, L, D = hidden_states.shape
        device = hidden_states.device
        
        attn_scores = self.attention(hidden_states).squeeze(-1)
        
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        for i, pos_list in enumerate(label_positions):
            valid_pos = [p for p in pos_list if p < L]
            if valid_pos:
                mask[i, valid_pos] = True
            else:
                mask[i, :] = True
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        
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
BENCH_TRAIN_PATH = os.path.join(script_dir, "nvemb", "output_benchset_rebalanced_train_nova.jsonl")
OUTPUT_DIR = os.path.join(script_dir, "nova_contrastive_bidir_noMNTP_bench")

train_samples = load_binarycorp_jsonl(BENCH_TRAIN_PATH)

class PairCollator:
    def __init__(self, nova_tokenizer, max_length=1024):
        self.nova_tokenizer = nova_tokenizer
        self.max_length = max_length
        self.label_ids = nova_tokenizer.labels
        self.pad_id = nova_tokenizer.tokenizer.pad_token_id or 0

    def __call__(self, batch):
        flat_texts = []
        func_ids = []
        instruct_template = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
        for p in batch:
            flat_texts.extend([instruct_template + p[0], p[1]])
            func_ids.extend([p[2], p[2]])

        all_ids, all_masks, all_label_positions = [], [], []

        for text in flat_texts:
            if text.startswith(instruct_template):
                asm_len = len(text) - len(instruct_template)
                char_types = "0" * len(instruct_template) + "1" * asm_len
            else:
                char_types = "1" * len(text)
            result = self.nova_tokenizer.encode("", text, char_types)

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
            "label_positions": all_label_positions,
            "func_ids": torch.tensor(func_ids, dtype=torch.long)
        }


def contrastive_loss_positive_aware(embeddings, func_ids, temperature=0.05):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    batch_size = embeddings.shape[0]

    labels = torch.arange(batch_size, device=embeddings.device)
    labels[::2] += 1
    labels[1::2] -= 1

    id_match_matrix = func_ids.unsqueeze(1) == func_ids.unsqueeze(0)
    mask_ignore = torch.eye(batch_size, device=embeddings.device).bool() | id_match_matrix
    sim_matrix.masked_fill_(mask_ignore, -1e9)

    mask_pos = torch.zeros_like(sim_matrix, dtype=torch.bool)
    mask_pos.scatter_(1, labels.unsqueeze(1), True)
    pos_scores = (embeddings * embeddings[labels]).sum(dim=1) / temperature
    sim_matrix[mask_pos] = pos_scores

    return F.cross_entropy(sim_matrix, labels)


INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "


def parse_bench_opt(opt):
    parts = opt.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected bench opt label: {opt}")
    compiler = parts[0]
    optimization = parts[1]
    variant = parts[-1]
    architecture = "_".join(parts[2:-1])
    return compiler, optimization, architecture, variant


def build_bench_pairs(samples, seed=42):
    grouped = defaultdict(list)
    for sample in samples:
        parse_bench_opt(sample["opt"])
        grouped[sample["id"]].append(sample)

    rng = random.Random(seed)
    func_id_to_int = {fid: i for i, fid in enumerate(sorted(grouped))}
    pairs = []
    skipped = 0

    for fid, variants in grouped.items():
        if len(variants) < 2:
            skipped += 1
            continue

        shuffled = variants[:]
        rng.shuffle(shuffled)
        func_int = func_id_to_int[fid]

        for idx, query in enumerate(shuffled):
            positive = shuffled[(idx + 1) % len(shuffled)]
            pairs.append((query["asm"], positive["asm"], func_int))

    print(
        f"Built {len(pairs)} bench pairs from {len(grouped)} functions "
        f"({skipped} skipped with <2 variants)."
    )
    return pairs


pairs = build_bench_pairs(train_samples, seed=42)
if not pairs:
    raise RuntimeError("No bench pairs were built; check dataset path and opt/id schema.")

batch_size = 32
grad_accum = 4
lr = 3e-5
num_epochs = 1

pair_collator = PairCollator(nova_tokenizer, max_length=1024)
pair_loader = DataLoader(pairs, batch_size=batch_size, shuffle=True, collate_fn=pair_collator)

pooling_head = AttentionPooling(model.config.hidden_size).to(device).to(torch.bfloat16)
pooling_head.train()

optimizer = AdamW(list(model.parameters()) + list(pooling_head.parameters()), lr=lr)

total_steps = max(1, len(pair_loader) // grad_accum * num_epochs)
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
        batch_inputs = {k: v.to(device) for k, v in batch.items() if k not in ('label_positions', 'func_ids')}
        label_pos = batch['label_positions']
        func_ids = batch['func_ids'].to(device)

        outputs = model(**batch_inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]

        embeddings = pooling_head(hidden, label_pos)
        loss = contrastive_loss_positive_aware(embeddings, func_ids)

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


os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained(OUTPUT_DIR)
torch.save(pooling_head.state_dict(), os.path.join(OUTPUT_DIR, "pooling_head.pt"))
print(f"saved to {OUTPUT_DIR}")