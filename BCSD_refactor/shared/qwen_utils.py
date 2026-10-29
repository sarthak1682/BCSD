import math
import sys
from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from shared.data_utils import EBM_BASE_SPECIAL_TOKENS
from shared.losses import cgte_loss, contrastive_loss_positive_aware
from shared.profiling import InferenceProfiler


def add_ebm_special_tokens(tokenizer) -> int:
    existing = list(tokenizer.additional_special_tokens or [])
    wanted = set(existing)
    wanted.update(EBM_BASE_SPECIAL_TOKENS)
    init_pieces = {}
    for token in sorted(wanted):
        if token not in existing:
            init_pieces[token] = tokenizer.encode(token, add_special_tokens=False)
    tokenizer.add_special_tokens({"additional_special_tokens": sorted(wanted)})
    tokenizer._ebm_init_pieces = init_pieces
    tokenizer.sep_token = "<SEP>"
    tokenizer.eos_token = "<EOS>"
    tokenizer.pad_token = "<PAD>"
    tokenizer.mask_token = "[MASK]"
    return tokenizer.convert_tokens_to_ids("[MASK]")


def resize_ebm_token_embeddings(model, tokenizer) -> None:
    """Resize a Qwen model and initialize new EBM tokens from old pieces."""
    old_vocab_size = model.get_input_embeddings().num_embeddings
    if old_vocab_size == len(tokenizer):
        return

    model.resize_token_embeddings(len(tokenizer))
    init_pieces = getattr(tokenizer, "_ebm_init_pieces", {})
    in_emb = model.get_input_embeddings().weight.data
    out_layer = model.get_output_embeddings()
    out_emb = out_layer.weight.data if out_layer is not None else None

    for token, piece_ids in init_pieces.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < old_vocab_size or token_id >= in_emb.size(0):
            continue
        usable = [pid for pid in piece_ids if 0 <= pid < old_vocab_size]
        if not usable:
            usable = list(range(min(old_vocab_size, in_emb.size(0))))
        in_emb[token_id].copy_(in_emb[usable].mean(dim=0))
        if out_emb is not None and token_id < out_emb.size(0):
            out_emb[token_id].copy_(out_emb[usable].mean(dim=0))


def patch_qwen_bidirectional(model: nn.Module) -> nn.Module:
    """Use padding-only attention, following L1NNA/binary_sim CustomQwen2BiModel."""
    body = getattr(model, "model", model)
    body = getattr(body, "model", body)
    if getattr(body, "_ebm_bidirectional_patched", False):
        return model

    def _update_causal_mask_bidir(
        self,
        attention_mask,
        input_tensor,
        cache_position=None,
        past_key_values=None,
        output_attentions=False,
        *args,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        dtype = input_tensor.dtype
        device = input_tensor.device
        batch_size, sequence_length = input_tensor.shape[:2]
        target_length = sequence_length
        if attention_mask is not None:
            target_length = attention_mask.shape[-1]

        min_dtype = torch.finfo(dtype).min
        mask = torch.zeros(
            (batch_size, 1, sequence_length, target_length),
            dtype=dtype,
            device=device,
        )
        if attention_mask is not None:
            key_padding = attention_mask[:, None, None, :].eq(0)
            mask = mask.masked_fill(key_padding, min_dtype)
        return mask

    body._update_causal_mask = MethodType(_update_causal_mask_bidir, body)
    body._ebm_bidirectional_patched = True
    return model

def make_bidirectional_4d_mask(
    input_ids: torch.Tensor,
    pad_token_id: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Convert a 2D padding mask into a 4D all-attend mask for Qwen2.

    Qwen2 with eager attention interprets a 4D ``attention_mask`` as the raw
    attention mask (0.0 = attend, ``min_dtype`` = mask-out).

    Returns ``[B, 1, L, L]`` where position *j* attends to position *k* iff
    neither *j* nor *k* is padding.
    """
    B, L = input_ids.shape
    pad_mask_2d = (input_ids == pad_token_id)  # [B, L]  True = pad
    # [B, 1, 1, L]
    key_pad = pad_mask_2d.unsqueeze(1).unsqueeze(2)
    # [B, 1, L, 1]
    qry_pad = pad_mask_2d.unsqueeze(1).unsqueeze(3)
    # any pad → mask that position pair  (use logical_or for CPU bfloat16 compat)
    mask_4d = torch.where(
        torch.logical_or(qry_pad, key_pad),
        torch.finfo(dtype).min,
        torch.tensor(0.0, dtype=dtype, device=input_ids.device),
    )
    return mask_4d


class QwenPoolingHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, L, D]
        attention_mask: torch.Tensor,  # [B, L]  or [B, 1, L, L]  (1=keep, 0=pad)
    ) -> torch.Tensor:
        if attention_mask.ndim == 4:
            mask_2d = attention_mask.squeeze(1)[:, 0, :]  # [B, L] — first query row
            mask_2d = (mask_2d == 0.0)
        elif attention_mask.ndim == 3:
            mask_2d = attention_mask[:, 0, :]
            mask_2d = (mask_2d == 0.0)
        else:
            mask_2d = attention_mask.bool()

        lengths = mask_2d.sum(dim=1) - 1  # [B]
        lengths = lengths.clamp(min=0)
        idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
        return hidden_states.gather(1, idx).squeeze(1)  # [B, D]


INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "


class QwenTranslationCollator:
    """Source labels are -100; target labels contain token IDs."""

    def __init__(self, tokenizer, max_length: int = 1024):
        self.tok = tokenizer
        self.maxL = max_length
        self.pad = tokenizer.pad_token_id or tokenizer.eos_token_id

    def _encode_pair(self, item) -> Tuple[List[int], List[int]]:
        if len(item) == 3:
            src, tgt, setting_tokens = item
            sep = "\n<SEP> " + setting_tokens + "\n"
        else:
            src, tgt = item
            sep = "\n<SEP>\n"

        src_ids = self.tok(src, add_special_tokens=False)["input_ids"]
        sep_ids = self.tok(sep, add_special_tokens=False)["input_ids"]
        tgt_ids = self.tok(tgt + "\n<EOS>", add_special_tokens=False)["input_ids"]

        available = max(1, self.maxL - len(sep_ids))
        tgt_take = min(len(tgt_ids), max(1, available // 2))
        src_take = min(len(src_ids), available - tgt_take)
        spare = available - src_take - tgt_take
        if spare > 0:
            extra_tgt = min(len(tgt_ids) - tgt_take, spare)
            tgt_take += extra_tgt
            spare -= extra_tgt
        if spare > 0:
            src_take += min(len(src_ids) - src_take, spare)

        src_slice = src_ids[-src_take:]
        tgt_slice = tgt_ids[-tgt_take:]
        input_ids = src_slice + sep_ids + tgt_slice
        labels = [-100] * (src_take + len(sep_ids)) + tgt_slice
        return input_ids[:self.maxL], labels[:self.maxL]

    def __call__(self, batch) -> dict:
        encoded = [self._encode_pair(item) for item in batch]
        max_len = max(len(ids) for ids, _ in encoded)
        input_ids = torch.full((len(batch), max_len), self.pad, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, (ids, labs) in enumerate(encoded):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            labels[i, :n] = torch.tensor(labs, dtype=torch.long)
            attn_mask[i, :n] = 1

        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


class QwenMNTPCollator:
    def __init__(
        self,
        tokenizer,
        mask_token_id: int,
        max_length: int = 1024,
        mask_prob: float = 0.15,
    ):
        self.tok = tokenizer
        self.mask = mask_token_id
        self.maxL = max_length
        self.mask_prob = mask_prob
        self.pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.special_ids = set(tokenizer.all_special_ids)

    def __call__(self, batch) -> dict:
        texts = [item["asm"] if isinstance(item, dict) else item for item in batch]
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.maxL,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        attn_2d = enc["attention_mask"].bool()
        labels = torch.full_like(input_ids, -100)

        for row in range(input_ids.size(0)):
            eligible = [
                idx for idx, tid in enumerate(input_ids[row].tolist())
                if attn_2d[row, idx] and tid not in self.special_ids
            ]
            if not eligible:
                continue
            mask_flags = torch.rand(len(eligible)) < self.mask_prob
            if not mask_flags.any():
                mask_flags[torch.randint(len(eligible), (1,)).item()] = True
            for flag, idx in zip(mask_flags.tolist(), eligible):
                if flag:
                    labels[row, idx] = input_ids[row, idx]
                    input_ids[row, idx] = self.mask

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


class QwenPairCollator:
    """Contrastive pair collator: produces flat batch of [query, target, query, target, ...]
    with 4D bidirectional masks and last-token positions.

    Query side gets the instruction template prepended.
    """

    def __init__(self, tokenizer, max_length: int = 1024):
        self.tok = tokenizer
        self.maxL = max_length
        self.pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.instruct_tok_len = len(tokenizer.encode(INSTRUCT_TEMPLATE, add_special_tokens=False))

    def __call__(self, batch):
        has_func_ids = len(batch[0]) > 2
        flat_texts: List[str] = []
        func_ids: List[int] = []

        for item in batch:
            if has_func_ids:
                q_asm, p_asm, fid = item
                q_text = q_asm if q_asm.startswith(INSTRUCT_TEMPLATE) else INSTRUCT_TEMPLATE + q_asm
                flat_texts.append(q_text)
                flat_texts.append(p_asm)
                func_ids.extend([fid, fid])
            else:
                q_asm, p_asm = item
                q_text = q_asm if q_asm.startswith(INSTRUCT_TEMPLATE) else INSTRUCT_TEMPLATE + q_asm
                flat_texts.append(q_text)
                flat_texts.append(p_asm)

        enc = self.tok(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=self.maxL,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attn_mask_2d = enc["attention_mask"].bool()

        result = {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "attn_mask_2d": attn_mask_2d,  # for last-token pooling
        }
        if has_func_ids:
            result["func_ids"] = torch.tensor(func_ids, dtype=torch.long)
        return result


def run_qwen_train(
    model: nn.Module,
    dataloader,
    epochs: int,
    lr: float,
    grad_accum: int,
    max_grad_norm: float,
    warmup_ratio: float,
    log_interval: int,
    log_fn,
    causal_lm: bool = True,
    mntp: bool = False,
    pooling_head: Optional[nn.Module] = None,
    temperature: float = 0.05,
    max_steps: Optional[int] = None,
    wandb_log=None,
    on_epoch_end: Optional[Callable[[int], None]] = None,
) -> None:
    """Stages 1 and 3 use translation and contrastive objectives, respectively."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if pooling_head is not None:
        trainable += list(pooling_head.parameters())

    batches_per_epoch = min(len(dataloader), max_steps) if max_steps else len(dataloader)
    total_steps = max(1, math.ceil(batches_per_epoch / grad_accum) * epochs)
    warmup_steps = int(warmup_ratio * total_steps)
    optimizer = AdamW(trainable, lr=lr, eps=1e-7)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    log_fn(
        f"  trainable params: {sum(p.numel() for p in trainable):,}"
        f"  |  steps: {total_steps}  |  warmup: {warmup_steps}  |  lr: {lr}"
    )

    device = next(model.parameters()).device

    for epoch in range(epochs):
        log_fn(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        if pooling_head is not None:
            pooling_head.train()
        losses: List[float] = []
        skipped = 0
        optimizer.zero_grad()

        if not causal_lm:
            accumulated_embs: List[torch.Tensor] = []
            accumulated_batches: List[dict] = []

        for step, batch in enumerate(tqdm(dataloader, desc=f"epoch {epoch + 1}")):
            if max_steps is not None and step >= max_steps:
                log_fn(f"  reached max_steps={max_steps}, stopping epoch early")
                break
            b_dev = {k: v.to(device) for k, v in batch.items()}

            if causal_lm:
                loss = model(**b_dev).loss
                if not loss.requires_grad or torch.isnan(loss):
                    skipped += 1
                    optimizer.zero_grad()
                    continue

                (loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if step % 200 == 0:
                        torch.cuda.empty_cache()

                losses.append(loss.item())
                if step > 0 and step % log_interval == 0:
                    w = losses[-log_interval:]
                    avg = sum(w) / len(w)
                    msg = f"  step {step:>6}/{len(dataloader)}  loss={avg:.4f}  skipped={skipped}"
                    print(msg, file=sys.stderr, flush=True)
                    log_fn(msg)
                    if wandb_log is not None:
                        wandb_log({"train/loss": avg, "train/step": step, "train/skipped": skipped})

            elif mntp:
                labels = b_dev.pop("labels")
                out = model(**b_dev)
                logits = out.logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                if not loss.requires_grad or torch.isnan(loss):
                    skipped += 1
                    optimizer.zero_grad()
                    continue

                (loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if step % 200 == 0:
                        torch.cuda.empty_cache()

                losses.append(loss.item())
                if step > 0 and step % log_interval == 0:
                    w = losses[-log_interval:]
                    avg = sum(w) / len(w)
                    msg = f"  step {step:>6}/{len(dataloader)}  loss={avg:.4f}  skipped={skipped}"
                    print(msg, file=sys.stderr, flush=True)
                    log_fn(msg)
                    if wandb_log is not None:
                        wandb_log({"train/loss": avg, "train/step": step, "train/skipped": skipped})

            else:
                with torch.no_grad():
                    func_ids = b_dev.pop("func_ids", None)
                    attn_mask_2d = b_dev.pop("attn_mask_2d", None)
                    if func_ids is not None:
                        func_ids = func_ids.to(device)

                    out = model(**b_dev, output_hidden_states=True)
                    embs = pooling_head(out.hidden_states[-1], attn_mask_2d)
                    accumulated_embs.append(embs.detach())
                    accumulated_batches.append((b_dev, attn_mask_2d, func_ids))

                if len(accumulated_embs) == grad_accum or (step + 1) == len(dataloader):
                    full_embs = torch.cat(accumulated_embs, dim=0)
                    full_embs.requires_grad_(True)

                    if accumulated_batches[0][2] is not None:
                        full_func_ids = torch.cat([x[2] for x in accumulated_batches], dim=0)
                        loss = contrastive_loss_positive_aware(
                            full_embs, full_func_ids, temperature=temperature
                        )
                    else:
                        x_emb = full_embs[0::2]
                        y_emb = full_embs[1::2]
                        loss = cgte_loss(x_emb, y_emb, temperature=temperature)

                    loss.backward()
                    embs_grad = full_embs.grad
                    losses.append(loss.item())

                    chunk_start = 0
                    for (b_d, m2d, _) in accumulated_batches:
                        chunk_size = b_d["input_ids"].shape[0]
                        out = model(**b_d, output_hidden_states=True)
                        chunk_embs = pooling_head(out.hidden_states[-1], m2d)
                        chunk_grad = embs_grad[chunk_start: chunk_start + chunk_size]
                        chunk_embs.backward(chunk_grad)
                        chunk_start += chunk_size

                    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    accumulated_embs = []
                    accumulated_batches = []

                    if (step // grad_accum) > 0 and (step // grad_accum) % log_interval == 0:
                        w = losses[-log_interval:]
                        avg = sum(w) / len(w)
                        msg = f"  step {step:>6}/{len(dataloader)}  loss={avg:.4f}  skipped={skipped}"
                        print(msg, file=sys.stderr, flush=True)
                        log_fn(msg)
                        if wandb_log is not None:
                            wandb_log({"train/loss": avg, "train/step": step, "train/skipped": skipped})

        if (causal_lm or mntp) and losses and len(losses) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if losses:
            log_fn(
                f"  epoch {epoch+1} done  avg_loss={sum(losses)/len(losses):.4f}"
                f"  valid={len(losses)}  skipped={skipped}"
            )
        if on_epoch_end is not None:
            on_epoch_end(epoch + 1)


@torch.no_grad()
def extract_qwen_embeddings(
    model: nn.Module,
    tokenizer,
    samples: List[dict],
    pooling_head: Optional[nn.Module] = None,
    batch_size: int = 16,
    max_length: int = 1024,
    device: str = "cuda",
    progress_every: int = 500,
) -> dict:
    """Extract embeddings from a Qwen model using last-token pooling.

    Returns ``{ids, opts, embeddings, stats}`` for 3M-style eval.
    """
    model.eval()

    all_embeddings = []
    all_ids = []
    all_opts = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    profiler = InferenceProfiler(device)
    is_warmup = True

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]

        texts = []
        for s in batch_samples:
            if s.get("opt", "O0") == "O0":
                texts.append(INSTRUCT_TEMPLATE + s["asm"])
            else:
                texts.append(s["asm"])

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn_2d = enc["attention_mask"].bool()
        attn_4d = make_bidirectional_4d_mask(input_ids, pad_id).to(device)

        with profiler:
            out = model(
                input_ids=input_ids,
                attention_mask=attn_4d,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-1]
            if pooling_head is not None:
                pooled = pooling_head(hidden, attn_2d.to(device))
            else:
                lengths = attn_2d.sum(dim=1) - 1
                lengths = lengths.clamp(min=0).to(device)
                idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.size(-1))
                pooled = hidden.gather(1, idx).squeeze(1)
        profiler.total_samples += len(batch_samples)

        if is_warmup:
            profiler.total_time_ms = 0.0
            profiler.total_samples = 0
            is_warmup = False

        all_embeddings.append(pooled.float().cpu())
        all_ids.extend(s["id"] for s in batch_samples)
        all_opts.extend(s.get("opt", "unknown") for s in batch_samples)

        if progress_every and i % progress_every == 0:
            print(f"Extracted {min(i + len(batch_samples), len(samples))}/{len(samples)}")

    return {
        "ids": all_ids,
        "opts": all_opts,
        "embeddings": torch.cat(all_embeddings, dim=0),
        "stats": profiler.get_stats(),
    }
