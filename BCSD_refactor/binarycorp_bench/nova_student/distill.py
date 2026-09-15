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
FULL_EBM_TEACHER_DIR = os.path.abspath(
    os.path.join(script_dir_file, "../../../../model_checkpoints/nova_ebm_bench/s3_final")
)

# Go up 3 levels to repository root where the jsonl files are located
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(script_dir, "./nvemb/output_benchset_rebalanced_train_nova.jsonl")

OUTPUT_DIR = "./model_checkpoints/nova_student_full_ebm_bench"

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
PHASE_A_EPOCHS = 20
CONTRASTIVE_MAX_EPOCHS = 10
RUN_ID = PHASE_A_EPOCHS
TOTAL_STEPS = None
MAX_LENGTH = 1024
STUDENT_LAYERS = 2
LAMBDA_START = 1.0
LAMBDA_END = 0.05
VAL_SPLIT = 0.1
CONTRASTIVE_EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_DELTA = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, CACHE_DIR)
from modeling_nova import NovaForCausalLM, NovaTokenizer

training_history = []

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

if os.path.exists(FULL_EBM_TEACHER_DIR):
    log(f"Loading merged full EBM teacher from {FULL_EBM_TEACHER_DIR}...")
    teacher_model = NovaForCausalLM.from_pretrained(
        FULL_EBM_TEACHER_DIR,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
else:
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
        log("WARNING: Full EBM teacher and adapter not found. Using raw Nova as teacher.")

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

log("Training student from scratch.")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

log(f"Student Module Parameters: {count_params(student_model):,}")
log(f"LAL Head Parameters:     {count_params(lal_head):,}")

mse_loss_fn = nn.MSELoss()

pair_collator = DistillCollator(nova_tokenizer, max_length=MAX_LENGTH)
pair_loader = DataLoader(train_pairs, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pair_collator, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_pairs, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pair_collator, num_workers=4, pin_memory=True) if val_pairs else None

TOTAL_STEPS = math.ceil(len(pair_loader) / GRAD_ACCUM) * PHASE_A_EPOCHS

optimizer = AdamW(list(student_model.parameters()) + list(lal_head.parameters()), lr=LR)
scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05*TOTAL_STEPS), TOTAL_STEPS)


def save_student_checkpoint(out_dir, suffix=""):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(out_dir, f"student_model{suffix}.pt"))
    torch.save(lal_head.state_dict(), os.path.join(out_dir, f"lal_head{suffix}.pt"))
    with open(os.path.join(out_dir, "student_config.json"), "w") as f:
        json.dump(
            {
                "vocab_size": len(base_tokenizer),
                "hidden_dim": HIDDEN_DIM,
                "num_layers": STUDENT_LAYERS,
            },
            f,
        )


def save_training_history():
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)


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


@torch.no_grad()
def evaluate_contrastive(student_model, lal_head, data_loader):
    if data_loader is None:
        return None
    student_model.eval()
    lal_head.eval()
    total = 0.0
    steps = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        func_ids = batch['func_ids'].to(device)
        key_padding_mask = batch['key_padding_mask'].to(device)
        pool_mask = batch['pool_mask'].to(device)

        h_student = student_model(input_ids, key_padding_mask=key_padding_mask)
        embeddings = lal_head(h_student, key_padding_mask=key_padding_mask, pool_mask=pool_mask)
        loss_contrastive = contrastive_loss_positive_aware(embeddings, func_ids)

        total += loss_contrastive.item()
        steps += 1

        del h_student, embeddings, loss_contrastive
        del input_ids, func_ids, key_padding_mask, pool_mask
        torch.cuda.empty_cache()
    student_model.train()
    lal_head.train()
    return total / max(1, steps)


log(f"Phase A: mixed MSE + contrastive distillation for {PHASE_A_EPOCHS} epochs / {TOTAL_STEPS} steps...")

student_model.train()
lal_head.train()
torch.cuda.empty_cache()
gc.collect()
optimizer.zero_grad()

global_step = 0
running_total = 0.0
running_contrastive = 0.0
running_distill = 0.0
running_count = 0
best_phase_a_val_loss = float("inf")

for epoch in range(PHASE_A_EPOCHS):
    for step, batch in enumerate(tqdm(pair_loader, desc=f"Phase A Epoch {epoch+1}")):
        
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
        running_count += 1
        loss_total = loss_total / GRAD_ACCUM
        
        loss_total.backward()
        
        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(pair_loader):
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(lal_head.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            if global_step % 20 == 0:
                avg_total = running_total / max(1, running_count)
                avg_contrastive = running_contrastive / max(1, running_count)
                avg_distill = running_distill / max(1, running_count)
                training_history.append({
                    "phase": "A",
                    "event": "train_step",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "loss": avg_total,
                    "contrastive_loss": avg_contrastive,
                    "distill_loss": avg_distill,
                    "lambda_mse": lambda_mse,
                })
                save_training_history()
                log_write(
                    f" Phase A Step {global_step}: "
                    f"Loss={avg_total:.4f} (Cont={avg_contrastive:.3f}, Dist={avg_distill:.3f}, "
                    f"lambda={lambda_mse:.3f})"
                )
            running_total = 0.0
            running_contrastive = 0.0
            running_distill = 0.0
            running_count = 0

        del h_selected, embeddings, loss_contrastive, loss_distill, loss_total
        del h_teacher, h_student, input_ids, nova_mask, key_padding_mask, pool_mask

    # Flush GPU memory accumulated during training before validation.
    gc.collect()
    torch.cuda.empty_cache()

    if val_loader is not None:
        val_loss = evaluate_distill(student_model, lal_head, teacher_model, val_loader)
        log(f"Phase A Epoch {epoch+1} Val Mixed Loss: {val_loss:.4f}")
        improved = val_loss + EARLY_STOP_MIN_DELTA < best_phase_a_val_loss
        training_history.append({
            "phase": "A",
            "event": "val_epoch",
            "epoch": epoch + 1,
            "mixed_loss": val_loss,
            "improved": improved,
        })
        save_training_history()

        if improved:
            best_phase_a_val_loss = val_loss
            save_student_checkpoint(os.path.join(OUTPUT_DIR, "phase_a_best"), "_best")

log("Phase A complete.")
save_student_checkpoint(os.path.join(OUTPUT_DIR, "phase_a_final"))

log("Releasing teacher before Phase C contrastive-only fine-tune...")
del teacher_model
gc.collect()
torch.cuda.empty_cache()

contrastive_total_steps = math.ceil(len(pair_loader) / GRAD_ACCUM) * CONTRASTIVE_MAX_EPOCHS
optimizer = AdamW(list(student_model.parameters()) + list(lal_head.parameters()), lr=LR)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    int(0.05 * contrastive_total_steps),
    contrastive_total_steps,
)
optimizer.zero_grad()

log(
    f"Phase C: contrastive-only fine-tune for up to {CONTRASTIVE_MAX_EPOCHS} epochs "
    f"(patience={CONTRASTIVE_EARLY_STOP_PATIENCE})."
)

global_step = 0
best_contrastive_val_loss = float("inf")
bad_epochs = 0

for epoch in range(CONTRASTIVE_MAX_EPOCHS):
    running_contrastive = 0.0
    running_count = 0

    for step, batch in enumerate(tqdm(pair_loader, desc=f"Phase C Epoch {epoch+1}")):
        input_ids = batch['input_ids'].to(device)
        func_ids = batch['func_ids'].to(device)
        key_padding_mask = batch['key_padding_mask'].to(device)
        pool_mask = batch['pool_mask'].to(device)

        h_student = student_model(input_ids, key_padding_mask=key_padding_mask)
        embeddings = lal_head(h_student, key_padding_mask=key_padding_mask, pool_mask=pool_mask)
        loss_contrastive = contrastive_loss_positive_aware(embeddings, func_ids)

        running_contrastive += loss_contrastive.item()
        running_count += 1
        (loss_contrastive / GRAD_ACCUM).backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(pair_loader):
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(lal_head.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 20 == 0:
                avg_contrastive = running_contrastive / max(1, running_count)
                training_history.append({
                    "phase": "C",
                    "event": "train_step",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "contrastive_loss": avg_contrastive,
                })
                save_training_history()
                log_write(f" Phase C Step {global_step}: Contrastive={avg_contrastive:.4f}")
            running_contrastive = 0.0
            running_count = 0

        del h_student, embeddings, loss_contrastive
        del input_ids, func_ids, key_padding_mask, pool_mask

    gc.collect()
    torch.cuda.empty_cache()

    if val_loader is not None:
        val_loss = evaluate_contrastive(student_model, lal_head, val_loader)
        log(f"Phase C Epoch {epoch+1} Val Contrastive Loss: {val_loss:.4f}")
        improved = val_loss + EARLY_STOP_MIN_DELTA < best_contrastive_val_loss
        training_history.append({
            "phase": "C",
            "event": "val_epoch",
            "epoch": epoch + 1,
            "contrastive_loss": val_loss,
            "improved": improved,
        })
        save_training_history()

        if improved:
            best_contrastive_val_loss = val_loss
            bad_epochs = 0
            save_student_checkpoint(os.path.join(OUTPUT_DIR, "student_best"), "_best")
        else:
            bad_epochs += 1
            if bad_epochs >= CONTRASTIVE_EARLY_STOP_PATIENCE:
                log("Phase C early stopping triggered.")
                break

save_student_checkpoint(os.path.join(OUTPUT_DIR, "phase_c_final"))
save_training_history()
log("Distillation + contrastive fine-tune complete.")

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
