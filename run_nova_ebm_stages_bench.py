"""Training script for Nova EBM stages 1-3 and evaluation (bench dataset).

Examples:
  python run_nova_ebm_stages_bench.py --stages 1,2,3
  python run_nova_ebm_stages_bench.py --stages 2,3 --s1_ckpt ./ckpts/s1_final
  python run_nova_ebm_stages_bench.py --stages eval --s3_ckpt ./ckpts/s3_final
"""

import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Prepend BCSD folder to sys.path so modules like metrics/models can be imported from other directories on the cluster
sys.path.insert(0, "/home/ra72yeq/projects/BCSD")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_bench import build_eval_pairs, compute_report, print_report_summary

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Paths
NOVA_CACHE_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--lt-asset--nova-1.3b"
    "/snapshots/4b4805bac4f13ef8bec678072ef60609ea3b0e77"
)
MODEL_ID           = "lt-asset/nova-1.3b"
TRAIN_DATA         = "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_train_nova.jsonl"
EVAL_DATA          = "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_test_nova.jsonl"
OUTPUT_DIR         = "./model_checkpoints/nova_ebm_bench"
POOLING_HEAD_FNAME = "pooling_head.pt"

# DoRA config (matches adapter_config.json in checkpoints)
DORA_CFG = dict(
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    use_dora       = True,
    bias           = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    modules_to_save= ["embed_tokens", "lm_head"],
)

# Hyperparameters
CFG = dict(
    seed           = 42,
    gpu_id         = 1,
    max_length     = 1024,
    max_grad_norm  = 1.0,
    warmup_ratio   = 0.03,
    log_interval   = 100,
    # Stage 1
    s1_epochs      = 1,    s1_batch = 4,   s1_grad_accum = 8,  s1_lr = 2e-5,
    bidir_pairs    = True,
    # Stage 2
    s2_epochs      = 1,    s2_batch = 4,   s2_grad_accum = 8,  s2_lr = 2e-4,
    mask_prob      = 0.15,
    # Stage 3
    s3_epochs      = 1,    s3_batch = 8,   s3_grad_accum = 8,  s3_lr = 3e-5,
    temperature    = 0.05,
    # Eval
    eval_batch_size = 16,
)


# Reproducibility

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Bidirectional mask helper

def make_bidirectional_nova_mask(nova_mask):
    """Symmetrize to make blocks bidirectional."""
    return np.maximum(nova_mask, nova_mask.T)


# Data loading

def load_binarycorp_jsonl(path: str):
    """Load .jsonl with columns: id, opt, asm"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return data


def parse_bench_opt(opt: str) -> Dict[str, str]:
    parts = opt.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected bench opt label: {opt}")
    return {
        "compiler": parts[0],
        "optimization": parts[1],
        "architecture": "_".join(parts[2:-1]),
        "variant": parts[-1],
    }


def asm_to_text(asm) -> str:
    return "\n".join(asm) if isinstance(asm, list) else asm


def group_samples_by_id(samples: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for sample in samples:
        parse_bench_opt(sample["opt"])
        grouped.setdefault(sample["id"], []).append(sample)
    return grouped


# Padding helper

def _pad_batch(
    ids_list:  List[np.ndarray],
    mask_list: List[np.ndarray],
    pad_id:    int,
    max_cap:   Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    max_len = max(len(x) for x in ids_list)
    if max_cap is not None:
        max_len = min(max_len, max_cap)
    B = len(ids_list)
    p_ids  = np.full((B, max_len), pad_id, dtype=np.int64)
    p_mask = np.zeros((B, max_len, max_len), dtype=np.float32)
    for i, (ids, msk) in enumerate(zip(ids_list, mask_list)):
        n = min(len(ids), max_len)
        p_ids[i, :n] = ids[:n]
        p_mask[i, :n, :n] = msk[:n, :n]
    return p_ids, p_mask


# DoRA wrapper

def _wrap_dora(model: nn.Module) -> nn.Module:
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, **DORA_CFG)
    return get_peft_model(model, cfg)


# Attention pooling

class AttentionPooling(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, hidden_states, label_positions):
        """
        hidden_states: [batch, seq_len, hidden_dim]
        label_positions: List of lists (indices of <label-N> tokens)
        """
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


# Stage 1: Cross-opt translation

class CrossOptTranslationDataset(Dataset):
    def __init__(self, samples: List[Dict], bidirectional: bool = True) -> None:
        self.pairs: List[Tuple[str, str]] = []
        grouped = group_samples_by_id(samples)
        skipped = 0
        for fid, variants in grouped.items():
            if len(variants) < 2:
                skipped += 1
                continue
            ordered = variants[:]
            random.shuffle(ordered)
            for idx, src in enumerate(ordered):
                tgt = ordered[(idx + 1) % len(ordered)]
                src_asm = asm_to_text(src["asm"])
                tgt_asm = asm_to_text(tgt["asm"])
                self.pairs.append((src_asm, tgt_asm))
                if bidirectional:
                    self.pairs.append((tgt_asm, src_asm))
        print(f"  {len(self.pairs):,} translation pairs "
              f"({'bidir' if bidirectional else 'unidir'}, {skipped} skipped)")

    def __len__(self):  return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


class TranslationCollator:
    """Causal-LM collator: source tokens get labels=-100, target tokens carry IDs."""

    def __init__(self, nova_tokenizer, max_length: int) -> None:
        self.tok  = nova_tokenizer
        self.maxL = max_length
        self.pad  = nova_tokenizer.tokenizer.pad_token_id or 0

    def __call__(self, batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        ids_l, lbl_l, msk_l = [], [], []
        for src, tgt in batch:
            r   = self.tok.encode(src, tgt, "1" * (len(src) + len(tgt)))
            L   = min(len(r["input_ids"]), self.maxL)
            ids_l.append(r["input_ids"][:L])
            lbl_l.append(r["labels"][:L])
            # Keep Nova's causal mask as-is for autoregressive training.
            msk_l.append(r["nova_attention_mask"][:L, :L])

        maxL = max(len(x) for x in ids_l)
        B    = len(batch)
        p_ids  = np.full((B, maxL), self.pad, dtype=np.int64)
        p_lbls = np.full((B, maxL), -100,     dtype=np.int64)
        p_msk  = np.zeros((B, maxL, maxL),    dtype=np.float32)
        for i, (ids, lbls, msk) in enumerate(zip(ids_l, lbl_l, msk_l)):
            n = len(ids)
            p_ids[i, :n] = ids; p_lbls[i, :n] = lbls; p_msk[i, :n, :n] = msk
        return {
            "input_ids":           torch.tensor(p_ids,  dtype=torch.long),
            "labels":              torch.tensor(p_lbls, dtype=torch.long),
            "nova_attention_mask": torch.tensor(p_msk,  dtype=torch.bfloat16),
        }


# Stage 2: MNTP

class MNTPCollator:
    def __init__(self, nova_tokenizer, mask_id, mask_prob=0.15, max_length=1024):
        self.nova_tokenizer = nova_tokenizer
        self.base_tokenizer = nova_tokenizer.tokenizer
        self.mask_id        = mask_id
        self.mask_prob      = mask_prob
        self.label_ids      = nova_tokenizer.labels
        self.max_length     = max_length

    def __call__(self, batch):
        all_input_ids, all_labels, all_masks = [], [], []

        for item in batch:
            text       = asm_to_text(item["asm"])
            char_types = "1" * len(text)
            result     = self.nova_tokenizer.encode("", text, char_types)

            input_ids = result['input_ids'].copy()
            labels    = np.full_like(result['labels'], -100)

            if len(input_ids) > self.max_length:
                input_ids                          = input_ids[:self.max_length]
                labels                             = labels[:self.max_length]
                result['nova_attention_mask']       = result['nova_attention_mask'][
                                                        :self.max_length, :self.max_length]

            for i in range(len(input_ids)):
                if input_ids[i] not in self.label_ids:
                    if np.random.random() < self.mask_prob:
                        labels[i]    = input_ids[i]
                        input_ids[i] = self.mask_id

            bidir_mask = make_bidirectional_nova_mask(result['nova_attention_mask'])
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_masks.append(bidir_mask)

        max_len = min(max(len(x) for x in all_input_ids), self.max_length)
        pad_id  = self.base_tokenizer.pad_token_id or 0

        padded_ids    = np.full((len(batch), max_len), pad_id, dtype=np.int64)
        padded_labels = np.full((len(batch), max_len), -100,   dtype=np.int64)
        padded_masks  = np.zeros((len(batch), max_len, max_len), dtype=np.float32)

        for i, (ids, labs, mask) in enumerate(zip(all_input_ids, all_labels, all_masks)):
            L = len(ids)
            padded_ids[i, :L]       = ids
            padded_labels[i, :L]    = labs
            padded_masks[i, :L, :L] = mask

        return {
            "input_ids":           torch.tensor(padded_ids),
            "labels":              torch.tensor(padded_labels),
            "nova_attention_mask": torch.tensor(padded_masks, dtype=torch.bfloat16),
        }


# Stage 3: Contrastive


class ContrastivePairDataset(Dataset):
    def __init__(self, samples: List[Dict]) -> None:
        grouped = group_samples_by_id(samples)
        func_id_to_int = {fid: i for i, fid in enumerate(sorted(grouped))}
        self.pairs: List[Tuple[str, str, int]] = []
        skipped = 0
        for fid, variants in grouped.items():
            if len(variants) < 2:
                skipped += 1
                continue
            ordered = variants[:]
            random.shuffle(ordered)
            func_int = func_id_to_int[fid]
            for idx, query in enumerate(ordered):
                positive = ordered[(idx + 1) % len(ordered)]
                self.pairs.append((asm_to_text(query["asm"]), asm_to_text(positive["asm"]), func_int))
        print(f"Paired samples: {len(self.pairs)} (Total items: {len(self.pairs) * 2})")
        print(f"  {skipped} functions skipped with <2 variants")

    def __len__(self):  return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


class PairCollator:
    def __init__(self, nova_tokenizer, max_length=1024):
        self.nova_tokenizer = nova_tokenizer
        self.max_length     = max_length
        self.label_ids      = nova_tokenizer.labels
        self.pad_id         = nova_tokenizer.tokenizer.pad_token_id or 0

    def __call__(self, batch):
        flat_texts = []
        func_ids = []
        instruct_template = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
        for query_asm, positive_asm, func_id in batch:
            flat_texts.extend([instruct_template + query_asm, positive_asm])
            func_ids.extend([func_id, func_id])

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
            L    = len(ids)
            mask = np.maximum(raw_mask[:L, :L], raw_mask[:L, :L].T)

            label_pos = [i for i, tid in enumerate(ids) if tid in self.label_ids]

            all_ids.append(ids)
            all_masks.append(mask)
            all_label_positions.append(label_pos)

        max_len   = max(len(x) for x in all_ids)
        pad_ids   = np.full((len(flat_texts), max_len), self.pad_id, dtype=np.int64)
        pad_masks = np.zeros((len(flat_texts), max_len, max_len), dtype=np.float32)

        for i, (ids, mask) in enumerate(zip(all_ids, all_masks)):
            L = len(ids)
            pad_ids[i, :L]       = ids
            pad_masks[i, :L, :L] = mask

        return {
            "input_ids":           torch.tensor(pad_ids),
            "nova_attention_mask": torch.tensor(pad_masks, dtype=torch.bfloat16),
            "label_positions":     all_label_positions,
            "func_ids":            torch.tensor(func_ids, dtype=torch.long),
        }


# Contrastive loss


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

# Training loop

def run_generic_train(
    model,
    dataloader,
    epochs:          int,
    lr:              float,
    grad_accum:      int,
    max_grad_norm:   float,
    warmup_ratio:    float,
    log_interval:    int,
    log_fn,
    contrastive:     bool                      = False,
    pooling_head:    Optional[AttentionPooling] = None,
    temperature:     float                     = 0.05,
    mntp:            bool                      = False,
    max_steps:       Optional[int]             = None,
) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]
    if pooling_head is not None:
        trainable += list(pooling_head.parameters())

    total_steps  = (len(dataloader) // grad_accum) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    optimizer    = AdamW(trainable, lr=lr, eps=1e-7)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    log_fn(f"  trainable params: {sum(p.numel() for p in trainable):,}"
           f"  |  steps: {total_steps}  |  warmup: {warmup_steps}  |  lr: {lr}")

    device = next(model.parameters()).device

    for epoch in range(epochs):
        log_fn(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        if pooling_head is not None:
            pooling_head.train()
        losses: List[float] = []
        skipped = 0
        optimizer.zero_grad()


        accumulated_embs = []
        accumulated_batches = []

        for step, batch in enumerate(tqdm(dataloader, desc=f"epoch {epoch + 1}")):
            if max_steps is not None and step >= max_steps:
                log_fn(f"  reached max_steps={max_steps}, stopping epoch early")
                break
            if contrastive:
                with torch.no_grad():
                    lpos = batch.pop("label_positions")
                    func_ids = batch.pop("func_ids").to(device)
                    b_dev = {k: v.to(device) for k, v in batch.items()}
                    if torch.isnan(b_dev["nova_attention_mask"]).any():
                        skipped += 1; continue
                        
                    out = model(**b_dev, output_hidden_states=True)
                    embs = pooling_head(out.hidden_states[-1], lpos)
                    accumulated_embs.append(embs.detach())
                    accumulated_batches.append((b_dev, lpos, func_ids))
                
                if len(accumulated_embs) == grad_accum or (step + 1) == len(dataloader):
                    full_embs = torch.cat(accumulated_embs, dim=0)
                    full_embs.requires_grad_(True)
                    full_func_ids = torch.cat([x[2] for x in accumulated_batches], dim=0)

                    loss = contrastive_loss_positive_aware(
                        full_embs,
                        full_func_ids,
                        temperature=temperature,
                    )
                    loss.backward()
                    
                    embs_grad = full_embs.grad
                    losses.append(loss.item())

                    chunk_start = 0
                    for (b_d, l_p, _) in accumulated_batches:
                        chunk_size = b_d["input_ids"].shape[0]
                        out = model(**b_d, output_hidden_states=True)
                        chunk_embs = pooling_head(out.hidden_states[-1], l_p)
                        
                        chunk_grad = embs_grad[chunk_start : chunk_start + chunk_size]
                        chunk_embs.backward(chunk_grad)
                        chunk_start += chunk_size

                    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad()
                    
                    accumulated_embs = []
                    accumulated_batches = []
                    
                    if (step // grad_accum) > 0 and (step // grad_accum) % log_interval == 0:
                        w = losses[-log_interval:]
                        log_fn(f"  step {step:>6}/{len(dataloader)}"
                               f"  loss={sum(w)/len(w):.4f}  skipped={skipped}")

            else:

                b = {k: v.to(device) for k, v in batch.items()}
                if torch.isnan(b["nova_attention_mask"]).any():
                    skipped += 1; optimizer.zero_grad(); continue
                
                if mntp:
                    outputs = model(**b)
                    logits = outputs.logits
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, logits.size(-1)), b["labels"].view(-1))
                else:
                    loss = model(**b).loss

                if not loss.requires_grad or torch.isnan(loss):
                    skipped += 1; optimizer.zero_grad(); continue

                (loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad()
                    if step % 200 == 0:
                        torch.cuda.empty_cache()

                losses.append(loss.item())
                if step > 0 and step % log_interval == 0:
                    w = losses[-log_interval:]
                    log_fn(f"  step {step:>6}/{len(dataloader)}"
                           f"  loss={sum(w)/len(w):.4f}  skipped={skipped}")

        if not contrastive and losses and len(losses) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        if losses:
            log_fn(f"  epoch {epoch+1} done  avg_loss={sum(losses)/len(losses):.4f}"
                   f"  valid={len(losses)}  skipped={skipped}")

# Embedding extraction


@torch.no_grad()
def encode_bench_texts(model, pooling_head, nova_tokenizer, texts,
                       batch_size=16, max_length=1024, device="cuda"):
    model.eval()
    pooling_head.eval()

    label_ids      = nova_tokenizer.labels
    base_tokenizer = nova_tokenizer.tokenizer
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_input_ids       = []
        batch_masks           = []
        batch_label_positions = []

        for text, char_types in batch_texts:
            result = nova_tokenizer.encode("", text, char_types)
            input_ids = result["input_ids"][:max_length]
            raw_mask  = result["nova_attention_mask"]
            L         = len(input_ids)
            mask      = np.maximum(raw_mask[:L, :L], raw_mask[:L, :L].T)
            label_pos = [j for j, tid in enumerate(input_ids) if tid in label_ids]

            batch_input_ids.append(input_ids)
            batch_masks.append(mask)
            batch_label_positions.append(label_pos)

        max_len = max(len(x) for x in batch_input_ids)
        pad_id  = base_tokenizer.pad_token_id or 0

        padded_ids   = np.full((len(batch_texts), max_len), pad_id, dtype=np.int64)
        padded_masks = np.zeros((len(batch_texts), max_len, max_len), dtype=np.float32)

        for j, (ids, mask) in enumerate(zip(batch_input_ids, batch_masks)):
            L = len(ids)
            padded_ids[j, :L]       = ids
            padded_masks[j, :L, :L] = mask

        input_ids_t = torch.tensor(padded_ids, dtype=torch.long, device=device)
        nova_mask_t = torch.tensor(padded_masks, dtype=torch.bfloat16, device=device)

        outputs = model(input_ids=input_ids_t, nova_attention_mask=nova_mask_t,
                        output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        pooled_batch = pooling_head(hidden, batch_label_positions)
        all_embeddings.append(pooled_batch.float().cpu())

        if i % 500 == 0:
            print(f"Encoded {min(i + len(batch_texts), len(texts))}/{len(texts)}")

    return torch.cat(all_embeddings, dim=0)


def extract_bench_eval_embeddings(model, pooling_head, nova_tokenizer, pairs,
                                  batch_size=16, max_length=1024, device="cuda"):
    instruct_template = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
    query_texts = []
    target_texts = []
    ids = []
    query_opts = []
    target_opts = []

    for pair in pairs:
        query_asm = asm_to_text(pair["query"]["asm"])
        target_asm = asm_to_text(pair["target"]["asm"])
        query_text = instruct_template + query_asm
        query_texts.append((query_text, "0" * len(instruct_template) + "1" * len(query_asm)))
        target_texts.append((target_asm, "1" * len(target_asm)))
        ids.append(pair["id"])
        query_opts.append(pair["query"]["opt"])
        target_opts.append(pair["target"]["opt"])

    print("Encoding query embeddings...")
    query_embeddings = encode_bench_texts(model, pooling_head, nova_tokenizer, query_texts,
                                          batch_size=batch_size, max_length=max_length, device=device)
    print("Encoding target embeddings...")
    target_embeddings = encode_bench_texts(model, pooling_head, nova_tokenizer, target_texts,
                                           batch_size=batch_size, max_length=max_length, device=device)

    return {
        "ids": ids,
        "query_opts": query_opts,
        "target_opts": target_opts,
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
    }


# Evaluation metrics

# Evaluation metrics removed in favor of EvaluationEngine


def run_eval(model, pooling_head, nova_tokenizer, eval_samples, args, log_fn) -> None:
    log_fn("\nEVALUATION")
    gc.collect(); torch.cuda.empty_cache()
    device = next(model.parameters()).device

    eval_pairs = build_eval_pairs(eval_samples, seed=args.seed)
    result = extract_bench_eval_embeddings(
        model,
        pooling_head,
        nova_tokenizer,
        eval_pairs,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        device=str(device),
    )
    emb_path = os.path.join(args.output_dir, "eval_bench_embeddings.pt")
    torch.save(result, emb_path)
    log_fn(f"  query embeddings {result['query_embeddings'].shape}  →  {emb_path}")
    log_fn(f"  target embeddings {result['target_embeddings'].shape}  →  {emb_path}")

    log_fn("\nEvaluating with bench metrics...")
    report = compute_report(result)
    report["source"] = os.path.abspath(emb_path)
    report["model"] = "nova_ebm_bench"
    report["data"] = os.path.abspath(EVAL_DATA)

    report_path = os.path.join(args.output_dir, "eval_bench_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log_fn(f"  report → {report_path}")

    print_report_summary(report)


# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages",     default="1,2,3",
                        help="e.g. '1,2,3' | '2,3' | 'eval'")
    parser.add_argument("--s1_ckpt",    default=None)
    parser.add_argument("--s2_ckpt",    default=None)
    parser.add_argument("--s3_ckpt",    default=None)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_length", type=int, default=CFG["max_length"])
    parser.add_argument("--seed",       type=int, default=CFG["seed"])
    args = parser.parse_args()

    for k, v in CFG.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    set_seed(args.seed)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir,
                            f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def log(msg: str) -> None:
        print(msg)
        with open(log_path, "a") as fh: fh.write(msg + "\n")

    log(f"Nova EBM  |  {datetime.now()}  |  stages={args.stages}")
    log(f"output_dir={args.output_dir}  device={device}\n")

    run_stages = set(s.strip().lower() for s in args.stages.split(","))

    log(f"Loading Nova from {NOVA_CACHE_DIR}")
    sys.path.insert(0, NOVA_CACHE_DIR)
    from modeling_nova import NovaForCausalLM, NovaTokenizer  

    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_tokenizer.add_special_tokens({'additional_special_tokens': ['[MASK]']})
    MASK_ID        = base_tokenizer.encode('[MASK]')[-1]
    nova_tokenizer = NovaTokenizer(base_tokenizer)

    def _load_nova(from_dir=None):
        src = from_dir or NOVA_CACHE_DIR
        log(f"  loading NovaForCausalLM from {src}")
        m = NovaForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16,
                                             device_map={"": 0})
        m.resize_token_embeddings(len(base_tokenizer))
        return m

    log("Loading datasets …")
    train_samples = load_binarycorp_jsonl(TRAIN_DATA)
    eval_samples  = load_binarycorp_jsonl(EVAL_DATA)

    # Stage 1
    if "1" in run_stages:
        log("\nStage 1: Cross-opt translation")
        base = _load_nova(args.s1_ckpt)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        model = _wrap_dora(base)
        model.enable_input_require_grads()

        loader = DataLoader(
            CrossOptTranslationDataset(train_samples, bidirectional=args.bidir_pairs),
            batch_size=args.s1_batch, shuffle=True, num_workers=0, pin_memory=True,
            collate_fn=TranslationCollator(nova_tokenizer, args.max_length),
        )
        run_generic_train(model, loader, args.s1_epochs, args.s1_lr,
                          args.s1_grad_accum, args.max_grad_norm,
                          args.warmup_ratio, args.log_interval, log)

        model   = model.merge_and_unload()
        s1_ckpt = os.path.join(args.output_dir, "s1_final")
        model.save_pretrained(s1_ckpt)
        log(f"  saved (merged) → {s1_ckpt}")
        args.s1_ckpt = s1_ckpt
        del model; gc.collect(); torch.cuda.empty_cache()

    # Stage 2
    if "2" in run_stages:
        log("\nStage 2: MNTP (bidirectional encoder)")
        base = _load_nova(args.s1_ckpt)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        model = _wrap_dora(base)
        model.enable_input_require_grads()

        loader = DataLoader(
            train_samples,
            batch_size=args.s2_batch, shuffle=True, num_workers=0, pin_memory=True,
            collate_fn=MNTPCollator(nova_tokenizer, MASK_ID,
                                    mask_prob=args.mask_prob,
                                    max_length=args.max_length),
        )
        run_generic_train(model, loader, args.s2_epochs, args.s2_lr,
                          args.s2_grad_accum, args.max_grad_norm,
                          args.warmup_ratio, args.log_interval, log, mntp=True,
                          max_steps=1000)

        model   = model.merge_and_unload()
        s2_ckpt = os.path.join(args.output_dir, "s2_final")
        model.save_pretrained(s2_ckpt)
        log(f"  saved (merged) → {s2_ckpt}")
        args.s2_ckpt = s2_ckpt
        del model; gc.collect(); torch.cuda.empty_cache()

    # Stage 3
    if "3" in run_stages:
        log("\nStage 3: Supervised contrastive + attention pooling")
        base = _load_nova(args.s2_ckpt)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        model = _wrap_dora(base)
        model.enable_input_require_grads()

        pooling_head = AttentionPooling(model.config.hidden_size).to(device).to(torch.bfloat16)

        loader = DataLoader(
            ContrastivePairDataset(train_samples),
            batch_size=args.s3_batch, shuffle=True, num_workers=0, pin_memory=True,
            collate_fn=PairCollator(nova_tokenizer, args.max_length),
        )
        run_generic_train(model, loader, args.s3_epochs, args.s3_lr,
                          args.s3_grad_accum, args.max_grad_norm,
                          args.warmup_ratio, args.log_interval, log,
                          contrastive=True, pooling_head=pooling_head,
                          temperature=args.temperature)

        model   = model.merge_and_unload()
        s3_ckpt = os.path.join(args.output_dir, "s3_final")
        os.makedirs(s3_ckpt, exist_ok=True)
        model.save_pretrained(s3_ckpt)
        torch.save(pooling_head.state_dict(), os.path.join(s3_ckpt, POOLING_HEAD_FNAME))
        log(f"  saved (merged) + pooling head → {s3_ckpt}")
        args.s3_ckpt = s3_ckpt

    # Eval
    if "eval" in run_stages or "3" in run_stages:
        log("\nEval")
        model        = _load_nova(args.s3_ckpt)
        pooling_head = AttentionPooling(model.config.hidden_size).to(device).to(torch.bfloat16)
        head_path    = os.path.join(args.s3_ckpt, POOLING_HEAD_FNAME) if args.s3_ckpt else None
        if head_path and os.path.isfile(head_path):
            pooling_head.load_state_dict(torch.load(head_path, map_location=device))
            log(f"  pooling head ← {head_path}")
        else:
            log("  no pooling head found, using random init")
        run_eval(model, pooling_head, nova_tokenizer, eval_samples, args, log)

    log("\nDone.")


if __name__ == "__main__":
    main()
