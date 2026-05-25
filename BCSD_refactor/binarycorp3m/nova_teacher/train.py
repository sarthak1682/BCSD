import os
import sys

# Setup paths to import shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.data_utils import set_seed, load_jsonl as load_binarycorp_jsonl
from shared.nova_utils import make_bidirectional_nova_mask, setup_nova_tokenizer, load_nova
from shared.pooling import AttentionPooling
from shared.collators import PairCollator
from shared.losses import contrastive_loss

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
from tqdm import tqdm
import gc
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

MODEL_ID = "lt-asset/nova-1.3b"

set_seed(42)

# Load tokenizer and model using shared utilities
base_tokenizer, nova_tokenizer, MASK_ID = setup_nova_tokenizer()
model = load_nova(device_id=0, base_tokenizer=base_tokenizer)

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

# Go up 3 levels to repository root where the jsonl files are located
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

train_samples = load_binarycorp_jsonl(os.path.join(script_dir, "binarycorp3m_train_nova.jsonl"))
eval_samples = load_binarycorp_jsonl(os.path.join(script_dir, "binarycorp3m_test_nova.jsonl"))

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
num_epochs = 1

pair_collator = PairCollator(nova_tokenizer, max_length=1024)
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


output_dir = os.path.join(script_dir, "BCSD_refactor/binarycorp3m/nova_teacher/nova_contrastive_bidir_noMNTP_final")
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
torch.save(pooling_head.state_dict(), os.path.join(output_dir, "pooling_head.pt"))
print(f"saved to {output_dir}")
