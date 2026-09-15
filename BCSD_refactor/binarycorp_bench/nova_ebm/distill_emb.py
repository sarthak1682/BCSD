import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
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
import argparse

# Setup paths to import shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.data_utils import set_seed, load_jsonl as load_binarycorp_jsonl, parse_bench_opt
from shared.student_model import StudentDistillationModule, LatentAttentionLayer, PositionalEncoding
from shared.losses import masked_mse_loss, contrastive_loss_positive_aware
from shared.collators import DistillCollator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SEED = 42
set_seed(SEED)

script_dir_file = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = "/home/ra72yeq/.cache/huggingface/hub/models--lt-asset--nova-1.3b/snapshots/4b4805bac4f13ef8bec678072ef60609ea3b0e77"

TEACHER_DIR = os.path.abspath(os.path.join(script_dir_file, "../../../../model_checkpoints/nova_ebm_s3only/s3_final"))

# Go up 3 levels to repository root where the jsonl files are located
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(script_dir, "./nvemb/output_benchset_rebalanced_train_nova.jsonl")

OUTPUT_DIR = "./model_checkpoints/nova_student_ebm_bench"

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

parser = argparse.ArgumentParser(description="Distill Nova EBM teacher into a small student")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU (default: 64)")
parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (default: 8)")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs (default: 50)")
parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length (default: 1024)")
parser.add_argument("--student_layers", type=int, default=2, help="Number of transformer layers in student (default: 2)")
parser.add_argument("--lambda_start", type=float, default=1.0, help="Initial distillation loss weight")
parser.add_argument("--lambda_end", type=float, default=0.05, help="Final distillation loss weight")
parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction (default: 0.1)")
parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience (default: 5)")
parser.add_argument("--early_stop_min_delta", type=float, default=1e-4, help="Min delta for early stopping (default: 1e-4)")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
GRAD_ACCUM = args.grad_accum
LR = args.lr
NUM_EPOCHS = args.num_epochs
MAX_LENGTH = args.max_length
STUDENT_LAYERS = args.student_layers
LAMBDA_START = args.lambda_start
LAMBDA_END = args.lambda_end
VAL_SPLIT = args.val_split
EARLY_STOP_PATIENCE = args.early_stop_patience
EARLY_STOP_MIN_DELTA = args.early_stop_min_delta
TOTAL_STEPS = None

device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, CACHE_DIR)
from modeling_nova import NovaForCausalLM, NovaTokenizer

log(f"Using device: {device}")

log("Loading Data...")
train_samples = load_binarycorp_jsonl(DATA_PATH)


def group_bench_samples(samples):
    grouped = defaultdict(list)
    for sample in samples:
        parse_bench_opt(sample["opt"])
        grouped[sample["id"]].append(sample)
    return grouped


def build_bench_pairs(grouped, func_ids, func_id_to_int, seed=42):
    """Build positive (query, positive, func_int) pairs via round-robin pairing."""
    rng = random.Random(seed)
    pairs = []

    for fid in func_ids:
        variants = grouped[fid]
        if len(variants) < 2:
            continue

        func_int = func_id_to_int[fid]
        shuffled = variants[:]
        rng.shuffle(shuffled)

        for idx, query in enumerate(shuffled):
            positive = shuffled[(idx + 1) % len(shuffled)]
            pairs.append((query["asm"], positive["asm"], func_int))

    rng.shuffle(pairs)
    log(f"Built {len(pairs)} pairs from {len(func_ids)} functions.")
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

if os.path.exists(TEACHER_DIR):
    log(f"Loading merged teacher from {TEACHER_DIR}...")
    teacher_model = NovaForCausalLM.from_pretrained(
        TEACHER_DIR,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
else:
    raise RuntimeError(f"S3-only teacher not found at {TEACHER_DIR} — run EBM S3-only first.")

teacher_model.requires_grad_(False)
teacher_model.eval()

HIDDEN_DIM = teacher_model.config.hidden_size
log(f"Teacher Hidden Dim: {HIDDEN_DIM}")

log("Initializing Student & Head...")
# Size the student embedding to the teacher's actual vocab. The EBM teacher was
# trained without the extra [MASK] token, so its embedding table can be smaller
# than len(base_tokenizer); distillation/eval never feed [MASK], so matching the
# teacher row count keeps the copy below valid.
teacher_emb = teacher_model.get_input_embeddings().weight.detach().clone()
VOCAB_SIZE = teacher_emb.shape[0]
if VOCAB_SIZE != len(base_tokenizer):
    log(f"Teacher vocab ({VOCAB_SIZE}) != tokenizer vocab ({len(base_tokenizer)}); using teacher vocab for student.")

student_model = StudentDistillationModule(
    vocab_size=VOCAB_SIZE,
    hidden_dim=HIDDEN_DIM,
    num_layers=STUDENT_LAYERS,
    pad_id=base_tokenizer.pad_token_id or 0
).to(device).to(torch.bfloat16)

lal_head = LatentAttentionLayer(hidden_dim=HIDDEN_DIM).to(device).to(torch.bfloat16)

log("Copying teacher embedding into student and freezing it...")
student_model.embedding.weight.data.copy_(teacher_emb)
student_model.embedding.requires_grad_(False)

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

            # Keep teacher forward in bfloat16 to avoid the FP32 logit
            # allocation inside modeling_nova that causes OOM.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                teacher_out = teacher_model(
                    input_ids=input_ids,
                    nova_attention_mask=nova_mask,
                    output_hidden_states=True
                )
            h_teacher = teacher_out.hidden_states[-1].detach()
            del teacher_out

            h_student = student_model(input_ids, key_padding_mask=key_padding_mask)

            loss_distill = masked_mse_loss(h_student, h_teacher, key_padding_mask)
            embeddings = lal_head(h_student, key_padding_mask=key_padding_mask, pool_mask=pool_mask)
            loss_contrastive = contrastive_loss_positive_aware(embeddings, func_ids)

            loss_total = loss_contrastive + (LAMBDA_END * loss_distill)
            total += loss_total.item()
            steps += 1

            del h_teacher, h_student, embeddings, loss_contrastive, loss_distill, loss_total
            del input_ids, nova_mask, key_padding_mask, pool_mask
            # Free GPU memory between val batches.
            torch.cuda.empty_cache()
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

    # Flush GPU memory accumulated during training before validation.
    gc.collect()
    torch.cuda.empty_cache()

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
    "vocab_size": VOCAB_SIZE,
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
