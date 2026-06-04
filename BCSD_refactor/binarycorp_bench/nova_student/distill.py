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
from collections import defaultdict
from tqdm import tqdm
import gc
from datetime import datetime
import math

# Setup paths to import shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.data_utils import set_seed, load_jsonl as load_binarycorp_jsonl, parse_bench_opt
from shared.nova_utils import setup_nova_tokenizer, load_nova
from shared.student_model import StudentDistillationModule, LatentAttentionLayer, PositionalEncoding
from shared.losses import masked_mse_loss, contrastive_loss_positive_aware
from shared.collators import DistillCollator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SEED = 42
set_seed(SEED)

script_dir_file = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = "/home/ra72yeq/.cache/huggingface/hub/models--lt-asset--nova-1.3b/snapshots/4b4805bac4f13ef8bec678072ef60609ea3b0e77"

ADAPTER_PATH = os.path.abspath(os.path.join(script_dir_file, "../nova_teacher/nova_contrastive_bidir_noMNTP_bench"))
if not os.path.exists(ADAPTER_PATH):
    ADAPTER_PATH = "/home/ra72yeq/projects/NovaXLLM2Vec/nova_contrastive_bidir_noMNTP_bench"

repo_root = os.path.abspath(os.path.join(script_dir_file, "../../../"))
DATA_PATH = os.path.join(repo_root, "nvemb", "output_benchset_rebalanced_train_nova.jsonl")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(repo_root, "output_benchset_rebalanced_train_nova.jsonl")

RUN_ID = 20
OUTPUT_DIR = os.path.join(script_dir_file, f"nova_distilled_student_{RUN_ID}_bench")
RESUME_FROM_CHECKPOINT = False
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
GRAD_ACCUM = 8
LR = 1e-4
NUM_EPOCHS = 20
TOTAL_STEPS = None 
MAX_LENGTH = 1024
STUDENT_LAYERS = 2
LAMBDA_START = 0.05 if RESUME_FROM_CHECKPOINT else 1.0
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


def group_bench_samples(samples):
    grouped = defaultdict(list)
    for sample in samples:
        parse_bench_opt(sample["opt"])
        grouped[sample["id"]].append(sample)
    return grouped


def build_bench_pairs(grouped, func_ids, func_id_to_int, seed=42):
    rng = random.Random(seed)
    pairs = []
    skipped = 0

    for fid in func_ids:
        variants = grouped[fid]
        if len(variants) < 2:
            skipped += 1
            continue

        shuffled = variants[:]
        rng.shuffle(shuffled)
        func_int = func_id_to_int[fid]

        # Use each variant once as a query, paired with another same-function variant.
        for idx, query in enumerate(shuffled):
            positive = shuffled[(idx + 1) % len(shuffled)]
            pairs.append((query["asm"], positive["asm"], func_int))

    rng.shuffle(pairs)
    log(f"Built {len(pairs)} pairs from {len(func_ids)} functions ({skipped} skipped with <2 variants).")
    return pairs


grouped_samples = group_bench_samples(train_samples)
valid_func_ids = [fid for fid, variants in grouped_samples.items() if len(variants) >= 2]
random.shuffle(valid_func_ids)

val_func_count = max(1, int(len(valid_func_ids) * VAL_SPLIT)) if len(valid_func_ids) > 1 else 0
val_func_ids = valid_func_ids[:val_func_count]
train_func_ids = valid_func_ids[val_func_count:]
func_id_to_int = {fid: i for i, fid in enumerate(sorted(valid_func_ids))}

train_pairs = build_bench_pairs(grouped_samples, train_func_ids, func_id_to_int, seed=SEED)
val_pairs = build_bench_pairs(grouped_samples, val_func_ids, func_id_to_int, seed=SEED + 1) if val_func_ids else []

if not train_pairs:
    raise RuntimeError("No bench train pairs were built; check dataset path and opt/id schema.")

log(
    f"Train functions: {len(train_func_ids)} | Val functions: {len(val_func_ids)} | "
    f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}"
)

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

log("Initializing Student & Head...")
student_model = StudentDistillationModule(
    vocab_size=len(base_tokenizer),
    hidden_dim=HIDDEN_DIM,
    num_layers=STUDENT_LAYERS,
    pad_id=base_tokenizer.pad_token_id or 0
).to(device).to(torch.bfloat16)

lal_head = LatentAttentionLayer(hidden_dim=HIDDEN_DIM).to(device).to(torch.bfloat16)

if RESUME_FROM_CHECKPOINT:
    log(f"Warm-starting student from {RESUME_DIR}...")
    student_model.load_state_dict(torch.load(os.path.join(RESUME_DIR, "student_model.pt"), map_location="cpu"))
    lal_head.load_state_dict(torch.load(os.path.join(RESUME_DIR, "lal_head.pt"), map_location="cpu"))
else:
    log("Training student from scratch.")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

log(f"Student Module Parameters: {count_params(student_model):,}")
log(f"LAL Head Parameters:     {count_params(lal_head):,}")

mse_loss_fn = nn.MSELoss()

pair_collator = DistillCollator(nova_tokenizer, max_length=MAX_LENGTH)
pair_loader = DataLoader(train_pairs, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pair_collator, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_pairs, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pair_collator, num_workers=4, pin_memory=True) if val_pairs else None

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
            func_ids = batch['func_ids'].to(device)
            key_padding_mask = batch['key_padding_mask'].to(device)
            pool_mask = batch['pool_mask'].to(device)

            teacher_out = teacher_model(
                input_ids=input_ids,
                nova_attention_mask=nova_mask,
                output_hidden_states=True
            )
            h_teacher = teacher_out.hidden_states[-1].detach()
            del teacher_out
            
            h_student = student_model(input_ids, key_padding_mask=key_padding_mask)

            loss_distill = masked_mse_loss(h_student, h_teacher.detach(), key_padding_mask)
            embeddings = lal_head(h_student, key_padding_mask=key_padding_mask, pool_mask=pool_mask)
            loss_contrastive = contrastive_loss_positive_aware(embeddings, func_ids)

            loss_total = loss_contrastive + (LAMBDA_END * loss_distill)
            total += loss_total.item()
            steps += 1

            del h_teacher, h_student, embeddings, loss_contrastive, loss_distill, loss_total
            del input_ids, nova_mask, key_padding_mask, pool_mask
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
        func_ids = batch['func_ids'].to(device)
        
        with torch.no_grad():
            teacher_out = teacher_model(
                input_ids=input_ids,
                nova_attention_mask=nova_mask,
                output_hidden_states=True
            )
            h_teacher = teacher_out.hidden_states[-1].detach()
            del teacher_out
            
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
            running_total = 0.0
            running_contrastive = 0.0
            running_distill = 0.0

        del h_selected, embeddings, loss_contrastive, loss_distill, loss_total
        del h_teacher, h_student, input_ids, nova_mask, key_padding_mask, pool_mask

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
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            torch.save(student_model.state_dict(), os.path.join(OUTPUT_DIR, "student_model_best.pt"))
            torch.save(lal_head.state_dict(), os.path.join(OUTPUT_DIR, "lal_head_best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                log("Early stopping triggered.")
                break

log("Distillation Complete.")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

log(f"Saving to {OUTPUT_DIR}...")
torch.save(student_model.state_dict(), os.path.join(OUTPUT_DIR, "student_model.pt"))
torch.save(lal_head.state_dict(), os.path.join(OUTPUT_DIR, "lal_head.pt"))

config = {
    "vocab_size": len(base_tokenizer),
    "hidden_dim": HIDDEN_DIM,
    "num_layers": STUDENT_LAYERS
}
with open(os.path.join(OUTPUT_DIR, "student_config.json"), "w") as f:
    json.dump(config, f)

log("Process finished successfully.")
