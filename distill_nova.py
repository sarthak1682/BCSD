import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import sys
import os
import random
import json
from tqdm import tqdm
import gc
from datetime import datetime
import math

SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

CACHE_DIR = "/home/ra72yeq/.cache/huggingface/hub/models--lt-asset--nova-1.3b/snapshots/4b4805bac4f13ef8bec678072ef60609ea3b0e77"
ADAPTER_PATH = "/home/ra72yeq/projects/NovaXLLM2Vec/nova_bidir_noMNTP_full_contrastive"
DATA_PATH = "binarycorp3m_train_nova.jsonl"
RUN_ID = 10 
OUTPUT_DIR = f"nova_distilled_student_{RUN_ID}"

BATCH_SIZE = 32
GRAD_ACCUM = 4
LR = 1e-4
NUM_EPOCHS = 10
TOTAL_STEPS = None 
MAX_LENGTH = 1024
STUDENT_LAYERS = 2
LAMBDA_START = 1.0
LAMBDA_END = 0.05
VAL_SPLIT = 0.1
EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_DELTA = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, CACHE_DIR)
from modeling_nova import NovaForCausalLM, NovaTokenizer

print(f"Using device: {device}")

def load_binarycorp_jsonl(path: str):
    data =[]
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

print("Loading Data...")
train_samples = load_binarycorp_jsonl(DATA_PATH)

INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "

pairs = []
o0_dict = {s['id']: s for s in train_samples if s['opt'] == 'O0'}
o3_dict = {s['id']: s for s in train_samples if s['opt'] == 'O3'}

for func_id in o0_dict:
    if func_id in o3_dict:
        text_query = INSTRUCT_TEMPLATE + o0_dict[func_id]['asm']
        text_pos = o3_dict[func_id]['asm']
        pairs.append((text_query, text_pos, func_id))

print(f"Loaded {len(pairs)} instructed pairs for distillation.")

random.shuffle(pairs)
val_size = max(1, int(len(pairs) * VAL_SPLIT)) if len(pairs) > 1 else 0
val_pairs = pairs[:val_size]
train_pairs = pairs[val_size:]
print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

print("Loading Teacher (Nova)...")
base_tokenizer = AutoTokenizer.from_pretrained("lt-asset/nova-1.3b", cache_dir=CACHE_DIR)
base_tokenizer.add_special_tokens({'additional_special_tokens': ['[MASK]']})
nova_tokenizer = NovaTokenizer(base_tokenizer)

teacher_model = NovaForCausalLM.from_pretrained(
    CACHE_DIR,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)
teacher_model.resize_token_embeddings(len(base_tokenizer))

if os.path.exists(ADAPTER_PATH):
    print(f"Loading trained adapter from {ADAPTER_PATH}...")
    teacher_model = PeftModel.from_pretrained(teacher_model, ADAPTER_PATH)
    teacher_model = teacher_model.merge_and_unload() 
else:
    print("WARNING: Adapter not found. Using raw Nova as teacher.")

teacher_model.requires_grad_(False)
teacher_model.eval()

HIDDEN_DIM = teacher_model.config.hidden_size
print(f"Teacher Hidden Dim: {HIDDEN_DIM}")


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

        # Cross-attention with Pre-LN and residual
        normed_latents = self.attn_norm(latents)
        attn_output, _ = self.cross_attn(
            query=normed_latents,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        latents = latents + attn_output  # Residual 1

        # MLP with Pre-LN and residual
        mlp_out = self.mlp(self.mlp_norm(latents))
        latents = latents + mlp_out  # Residual 2

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
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
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

print("Initializing Student & Head...")
student_model = StudentDistillationModule(
    vocab_size=len(base_tokenizer),
    hidden_dim=HIDDEN_DIM,
    num_layers=STUDENT_LAYERS,
    pad_id=base_tokenizer.pad_token_id or 0
).to(device).to(torch.bfloat16)

lal_head = LatentAttentionLayer(hidden_dim=HIDDEN_DIM).to(device).to(torch.bfloat16)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Student Module Parameters: {count_params(student_model):,}")
print(f"LAL Head Parameters:     {count_params(lal_head):,}")

def masked_mse_loss(student_hidden, teacher_hidden, key_padding_mask):
    valid = (~key_padding_mask).unsqueeze(-1)
    diff = (student_hidden - teacher_hidden) ** 2
    diff = diff * valid
    denom = valid.sum().clamp_min(1.0)
    return diff.sum() / denom


class DistillCollator:
    def __init__(self, nova_tokenizer, max_length=1024):
        self.nova_tokenizer = nova_tokenizer
        self.max_length = max_length
        self.pad_id = nova_tokenizer.tokenizer.pad_token_id or 0

    def __call__(self, batch):
        flat_texts = []
        func_ids =[]
        for (q, p, fid) in batch:
            flat_texts.extend([q, p])
            func_ids.extend([fid, fid])

        all_ids, all_masks = [],[]

        instruct_template = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
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

            all_ids.append(ids)
            all_masks.append(mask)

        max_len = max(len(x) for x in all_ids)
        pad_ids = np.full((len(flat_texts), max_len), self.pad_id, dtype=np.int64)
        pad_masks = np.zeros((len(flat_texts), max_len, max_len), dtype=np.float32)

        for i, (ids, mask) in enumerate(zip(all_ids, all_masks)):
            L = len(ids)
            pad_ids[i, :L] = ids
            pad_masks[i, :L, :L] = mask

        key_padding_mask = (pad_ids == self.pad_id)

        return {
            "input_ids": torch.tensor(pad_ids),
            "nova_attention_mask": torch.tensor(pad_masks, dtype=torch.bfloat16),
            "key_padding_mask": torch.tensor(key_padding_mask),
            "func_ids": func_ids
        }


def contrastive_loss_positive_aware(embeddings, func_ids, temperature=0.05):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    batch_size = embeddings.shape[0]
    
    labels = torch.arange(batch_size, device=embeddings.device)
    labels[::2] += 1
    labels[1::2] -= 1
    
    mask_ignore = torch.eye(batch_size, device=embeddings.device).bool()
    
    unique_ids = list(set(func_ids))
    id_map = {uid: i for i, uid in enumerate(unique_ids)}
    batch_numeric_ids = [id_map[fid] for fid in func_ids]
    
    ids_tensor = torch.tensor(batch_numeric_ids, device=embeddings.device).unsqueeze(1)
    id_match_matrix = (ids_tensor == ids_tensor.T)
    
    # Mask out self AND same-function pairs (false negatives)
    mask_ignore = mask_ignore | id_match_matrix
    sim_matrix.masked_fill_(mask_ignore, -1e9)
    
    # Restore the true positive scores that were wiped by the id_match_matrix mask
    mask_pos = torch.zeros_like(sim_matrix, dtype=torch.bool)
    mask_pos.scatter_(1, labels.unsqueeze(1), True)
    pos_scores = (embeddings * embeddings[labels]).sum(dim=1) / temperature
    sim_matrix[mask_pos] = pos_scores

    return F.cross_entropy(sim_matrix, labels)

mse_loss_fn = nn.MSELoss()


pair_collator = DistillCollator(nova_tokenizer, max_length=MAX_LENGTH)
pair_loader = DataLoader(train_pairs, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pair_collator)
val_loader = DataLoader(val_pairs, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pair_collator) if val_pairs else None

TOTAL_STEPS = math.ceil(len(pair_loader) / GRAD_ACCUM) * NUM_EPOCHS

optimizer = AdamW(list(student_model.parameters()) + list(lal_head.parameters()), lr=LR)
scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05*TOTAL_STEPS), TOTAL_STEPS)


def evaluate_distill(student_model, lal_head, teacher_model, data_loader):
    if data_loader is None:
        return None
    student_model.eval()
    lal_head.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            nova_mask = batch['nova_attention_mask'].to(device)
            func_ids = batch['func_ids']
            key_padding_mask = batch['key_padding_mask'].to(device)

            teacher_out = teacher_model(
                input_ids=input_ids,
                nova_attention_mask=nova_mask,
                output_hidden_states=True
            )
            h_teacher = teacher_out.hidden_states[-1]
            h_student = student_model(input_ids, key_padding_mask=key_padding_mask)

            loss_distill = masked_mse_loss(h_student, h_teacher.detach(), key_padding_mask)
            embeddings = lal_head(h_student, key_padding_mask=key_padding_mask)
            loss_contrastive = contrastive_loss_positive_aware(embeddings, func_ids)

            loss_total = loss_contrastive + (LAMBDA_END * loss_distill)
            total += loss_total.item()
            steps += 1
    student_model.train()
    lal_head.train()
    return total / max(1, steps)


print(f"Starting Distillation for {TOTAL_STEPS} steps...")

student_model.train()
lal_head.train()
torch.cuda.empty_cache()
gc.collect()

global_step = 0
running_total = 0.0
running_contrastive = 0.0
running_distill = 0.0
best_val_loss = float("inf")
bad_epochs = 0

for epoch in range(NUM_EPOCHS):
    for step, batch in enumerate(tqdm(pair_loader, desc=f"Epoch {epoch+1}")):
        
        input_ids = batch['input_ids'].to(device)
        nova_mask = batch['nova_attention_mask'].to(device)
        func_ids = batch['func_ids']
        
        with torch.no_grad():
            teacher_out = teacher_model(
                input_ids=input_ids,
                nova_attention_mask=nova_mask,
                output_hidden_states=True
            )
            h_teacher = teacher_out.hidden_states[-1]
            
        key_padding_mask = batch['key_padding_mask'].to(device)
        h_student = student_model(input_ids, key_padding_mask=key_padding_mask)

        if TOTAL_STEPS <= 1:
            lambda_mse = LAMBDA_END
        else:
            progress = min(global_step, TOTAL_STEPS - 1) / (TOTAL_STEPS - 1)
            progress = progress ** 2
            lambda_mse = LAMBDA_START + (LAMBDA_END - LAMBDA_START) * progress

        loss_distill = masked_mse_loss(h_student, h_teacher.detach(), key_padding_mask)
        
        if random.random() < 0.5:
            h_selected = h_teacher.detach()
        else:
            h_selected = h_student
            
        embeddings = lal_head(h_selected, key_padding_mask=key_padding_mask)
        loss_contrastive = contrastive_loss_positive_aware(embeddings, func_ids)
        
        loss_total = loss_contrastive + (lambda_mse * loss_distill)
        running_total += loss_total.item()
        running_contrastive += loss_contrastive.item()
        running_distill += loss_distill.item()
        loss_total = loss_total / GRAD_ACCUM
        
        loss_total.backward()
        
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(lal_head.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            if global_step % 20 == 0:
                avg_total = running_total / GRAD_ACCUM
                avg_contrastive = running_contrastive / GRAD_ACCUM
                avg_distill = running_distill / GRAD_ACCUM
                tqdm.write(
                    f" Step {global_step}: "
                    f"Loss={avg_total:.4f} (Cont={avg_contrastive:.3f}, Dist={avg_distill:.3f}, "
                    f"lambda={lambda_mse:.3f})"
                )
                torch.cuda.empty_cache()
            running_total = 0.0
            running_contrastive = 0.0
            running_distill = 0.0

        if global_step >= TOTAL_STEPS:
            break
            
    if global_step >= TOTAL_STEPS:
        break

    if val_loader is not None:
        val_loss = evaluate_distill(student_model, lal_head, teacher_model, val_loader)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

        if val_loss + EARLY_STOP_MIN_DELTA < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            torch.save(student_model.state_dict(), os.path.join(OUTPUT_DIR, "student_model_best.pt"))
            torch.save(lal_head.state_dict(), os.path.join(OUTPUT_DIR, "lal_head_best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

print("Distillation Complete.")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Saving to {OUTPUT_DIR}...")
torch.save(student_model.state_dict(), os.path.join(OUTPUT_DIR, "student_model.pt"))
torch.save(lal_head.state_dict(), os.path.join(OUTPUT_DIR, "lal_head.pt"))

config = {
    "vocab_size": len(base_tokenizer),
    "hidden_dim": HIDDEN_DIM,
    "num_layers": STUDENT_LAYERS
}
with open(os.path.join(OUTPUT_DIR, "student_config.json"), "w") as f:
    json.dump(config, f)

print("Process finished successfully.")