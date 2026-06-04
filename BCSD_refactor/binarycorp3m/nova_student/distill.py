import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig
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

# Setup paths to import shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.data_utils import set_seed, load_jsonl as load_binarycorp_jsonl
from shared.nova_utils import setup_nova_tokenizer, load_nova
from shared.losses import masked_mse_loss, contrastive_loss_positive_aware
from shared.student_model import StudentDistillationModule, LatentAttentionLayer
from shared.collators import DistillCollator

SEED = 42
set_seed(SEED)

CACHE_DIR = "/home/ra72yeq/.cache/huggingface/hub/models--lt-asset--nova-1.3b/snapshots/4b4805bac4f13ef8bec678072ef60609ea3b0e77"

# Check if refactored teacher model exists relative to this file
ADAPTER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../nova_teacher/nova_contrastive_bidir_noMNTP_final"))
if not os.path.exists(ADAPTER_PATH):
    ADAPTER_PATH = "/home/ra72yeq/projects/NovaXLLM2Vec/nova_contrastive_bidir_noMNTP_final"

# Go up 3 levels to repository root where the jsonl files are located
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(script_dir, "binarycorp3m_train_nova.jsonl")

RUN_ID = 20
script_dir_file = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "./model_checkpoints/nova_student"
RESUME_DIR = "/home/ra72yeq/projects/NovaXLLM2Vec/nova_distilled_student_10"

os.makedirs(OUTPUT_DIR, exist_ok=True)
log_path = os.path.join(OUTPUT_DIR, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def log(msg: str, print_to_stdout: bool = True) -> None:
    if print_to_stdout:
        print(msg)
    with open(log_path, "a") as fh:
        fh.write(msg + "\n")

def log_write(msg: str) -> None:
    tqdm.write(msg)
    log(msg, print_to_stdout=False)

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

log(f"Using device: {device}")

log("Loading Data...")
train_samples = load_binarycorp_jsonl(DATA_PATH)

INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "

pairs = []
o0_dict = {s['id']: s for s in train_samples if s['opt'] == 'O0'}
o3_dict = {s['id']: s for s in train_samples if s['opt'] == 'O3'}

str_to_int_map = {fid: i for i, fid in enumerate(o0_dict.keys())}
for func_id in o0_dict:
    if func_id in o3_dict:
        text_query = INSTRUCT_TEMPLATE + o0_dict[func_id]['asm']
        text_pos = o3_dict[func_id]['asm']
        pairs.append((text_query, text_pos, str_to_int_map[func_id]))

log(f"Loaded {len(pairs)} instructed pairs for distillation.")

random.shuffle(pairs)

val_size = max(1, int(len(pairs) * VAL_SPLIT)) if len(pairs) > 1 else 0
val_pairs = pairs[:val_size]
train_pairs = pairs[val_size:]
log(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

log("Loading Teacher (Nova)...")
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
    log(f"Loading trained adapter from {ADAPTER_PATH}...")
    teacher_model = PeftModel.from_pretrained(teacher_model, ADAPTER_PATH)
    teacher_model = teacher_model.merge_and_unload() 
else:
    log("WARNING: Adapter not found. Using raw Nova as teacher.")

teacher_model.requires_grad_(False)
teacher_model.eval()

HIDDEN_DIM = teacher_model.config.hidden_size
log(f"Teacher Hidden Dim: {HIDDEN_DIM}")

student_model = StudentDistillationModule(
    vocab_size=len(base_tokenizer),
    hidden_dim=HIDDEN_DIM,
    num_layers=STUDENT_LAYERS,
    pad_id=base_tokenizer.pad_token_id or 0
).to(device)

lal_head = LatentAttentionLayer(hidden_dim=HIDDEN_DIM, num_latents=512, num_heads=8).to(device)

student_model = student_model.to(torch.bfloat16)
lal_head = lal_head.to(torch.bfloat16)

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
            pool_mask = batch['pool_mask'].to(device)

            teacher_out = teacher_model(
                input_ids=input_ids,
                nova_attention_mask=nova_mask,
                output_hidden_states=True
            )
            h_teacher = teacher_out.hidden_states[-1]
            h_student = student_model(input_ids, key_padding_mask=key_padding_mask)

            loss_distill = masked_mse_loss(h_student, h_teacher.detach(), key_padding_mask)
            embeddings = lal_head(h_student, key_padding_mask=key_padding_mask, pool_mask=pool_mask)
            loss_contrastive = contrastive_loss_positive_aware(embeddings, func_ids)

            loss_total = loss_contrastive + (LAMBDA_END * loss_distill)
            total += loss_total.item()
            steps += 1
    student_model.train()
    lal_head.train()
    return total / max(1, steps)


log(f"Starting Distillation for {TOTAL_STEPS} steps...")

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
        pool_mask = batch['pool_mask'].to(device)
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
            
        embeddings = lal_head(h_selected, key_padding_mask=key_padding_mask, pool_mask=pool_mask)
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
                log_write(
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
        log(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

        if val_loss + EARLY_STOP_MIN_DELTA < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            best_dir = os.path.join(OUTPUT_DIR, "student_best")
            os.makedirs(best_dir, exist_ok=True)
            torch.save(student_model.state_dict(), os.path.join(best_dir, "student_model_best.pt"))
            torch.save(lal_head.state_dict(), os.path.join(best_dir, "lal_head_best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                log("Early stopping triggered.")
                break

log("Distillation Complete.")

final_dir = os.path.join(OUTPUT_DIR, "student_final")
os.makedirs(final_dir, exist_ok=True)

log(f"Saving to {final_dir}...")
torch.save(student_model.state_dict(), os.path.join(final_dir, "student_model.pt"))
torch.save(lal_head.state_dict(), os.path.join(final_dir, "lal_head.pt"))

config = {
    "vocab_size": len(base_tokenizer),
    "hidden_dim": HIDDEN_DIM,
    "num_layers": STUDENT_LAYERS
}
with open(os.path.join(final_dir, "student_config.json"), "w") as f:
    json.dump(config, f)

# Also save config to student_best if it exists
best_dir = os.path.join(OUTPUT_DIR, "student_best")
if os.path.exists(best_dir):
    with open(os.path.join(best_dir, "student_config.json"), "w") as f:
        json.dump(config, f)

log("Process finished successfully.")
