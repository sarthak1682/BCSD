import os
import sys

# Setup paths to import shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.data_utils import set_seed, load_jsonl as load_binarycorp_jsonl, parse_bench_opt
from shared.nova_utils import make_bidirectional_nova_mask, setup_nova_tokenizer, load_nova
from shared.pooling import AttentionPooling
from shared.collators import PairCollator
from shared.losses import contrastive_loss_positive_aware

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
import json
from collections import defaultdict
from tqdm import tqdm
import gc
import torch.nn.functional as F
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup

MODEL_ID = "lt-asset/nova-1.3b"

# Fraction of positive pairs forced to span two different architectures (when the
# function has variants in >=2 archs). Injects the cross-arch gradient signal that
# plain round-robin pairing almost never produced.
CROSS_ARCH_PAIR_PROB = 0.7

set_seed(42)

# Load tokenizer and base model using shared utilities
base_tokenizer, nova_tokenizer, MASK_ID = setup_nova_tokenizer()
model = load_nova(device_id=0, base_tokenizer=base_tokenizer)

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

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../../../"))

# Support different dataset root directories
BENCH_TRAIN_PATH = os.path.join(repo_root, "nvemb", "output_benchset_rebalanced_train_nova.jsonl")
if not os.path.exists(BENCH_TRAIN_PATH):
    BENCH_TRAIN_PATH = os.path.join(repo_root, "output_benchset_rebalanced_train_nova.jsonl")

OUTPUT_DIR = os.path.join(script_dir, "nova_contrastive_bidir_noMNTP_bench_ca")
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


train_samples = load_binarycorp_jsonl(BENCH_TRAIN_PATH)


def variant_arch(sample):
    """Architecture of a sample, parsed from its bench opt label (e.g. 'mips64')."""
    return parse_bench_opt(sample["opt"])["architecture"]


def build_bench_pairs(samples, seed=42, cross_arch_prob=CROSS_ARCH_PAIR_PROB):
    """Build positive (query, positive, func_int) pairs for contrastive training.

    Each tuple becomes an adjacent positive pair in the collator/contrastive loss, so the
    cross-architecture learning signal depends entirely on whether the pair itself spans
    archs. For functions present in >=2 architectures we deliberately route most queries to
    a positive drawn from a *different* architecture; the rest fall back to round-robin
    neighbours (which still teach cross-opt / cross-compiler / cross-variant invariance).
    """
    grouped = defaultdict(list)
    for sample in samples:
        parse_bench_opt(sample["opt"])
        grouped[sample["id"]].append(sample)

    rng = random.Random(seed)
    func_id_to_int = {fid: i for i, fid in enumerate(sorted(grouped))}
    pairs = []
    skipped = 0
    cross_arch_count = 0
    same_arch_count = 0

    for fid, variants in grouped.items():
        if len(variants) < 2:
            skipped += 1
            continue

        func_int = func_id_to_int[fid]

        by_arch = defaultdict(list)
        for v in variants:
            by_arch[variant_arch(v)].append(v)
        multi_arch = len(by_arch) >= 2

        shuffled = variants[:]
        rng.shuffle(shuffled)

        for idx, query in enumerate(shuffled):
            q_arch = variant_arch(query)
            positive = None

            if multi_arch and rng.random() < cross_arch_prob:
                other_archs = [a for a in by_arch if a != q_arch]
                if other_archs:
                    chosen_arch = rng.choice(other_archs)
                    positive = rng.choice(by_arch[chosen_arch])

            if positive is None:
                positive = shuffled[(idx + 1) % len(shuffled)]

            if variant_arch(positive) != q_arch:
                cross_arch_count += 1
            else:
                same_arch_count += 1
            pairs.append((query["asm"], positive["asm"], func_int))

    log(
        f"Built {len(pairs)} bench pairs from {len(grouped)} functions "
        f"({skipped} skipped with <2 variants) | "
        f"cross-arch positives={cross_arch_count} same-arch positives={same_arch_count}."
    )
    return pairs


pairs = build_bench_pairs(train_samples, seed=42)
if not pairs:
    raise RuntimeError("No bench pairs were built; check dataset path and opt/id schema.")

log(f"Data: {BENCH_TRAIN_PATH}")
log(f"Output: {OUTPUT_DIR}")
log(f"CROSS_ARCH_PAIR_PROB={CROSS_ARCH_PAIR_PROB}")

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
            log_write(f"Step {step}: avg loss = {avg:.4f}")

    log(f"Epoch {epoch+1}: avg loss = {sum(epoch_losses)/len(epoch_losses):.4f}")

model.save_pretrained(OUTPUT_DIR)
pooling_path = os.path.join(OUTPUT_DIR, "pooling_head.pt")
torch.save(pooling_head.state_dict(), pooling_path)

run_config = {
    "cross_arch_pair_prob": CROSS_ARCH_PAIR_PROB,
    "batch_size": batch_size,
    "grad_accum": grad_accum,
    "lr": lr,
    "num_epochs": num_epochs,
    "num_pairs": len(pairs),
    "data_path": BENCH_TRAIN_PATH,
}
with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)

log(f"Saved LoRA adapter to {OUTPUT_DIR}")
log(f"Saved pooling head to {pooling_path}")
log(f"Saved run config to {os.path.join(OUTPUT_DIR, 'train_config.json')}")
log(f"Training log: {log_path}")
