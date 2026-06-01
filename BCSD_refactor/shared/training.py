"""Unified training utilities for Nova model stages."""

from typing import List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from shared.losses import cgte_loss, contrastive_loss_positive_aware


def run_generic_train(
    model: nn.Module,
    dataloader,
    epochs: int,
    lr: float,
    grad_accum: int,
    max_grad_norm: float,
    warmup_ratio: float,
    log_interval: int,
    log_fn,
    contrastive: bool = False,
    pooling_head: Optional[nn.Module] = None,
    temperature: float = 0.05,
    mntp: bool = False,
    max_steps: Optional[int] = None,
    wandb_log=None,  # callable(dict) — pass wandb.log to enable per-step remote logging
) -> None:
    """Unified training loop supporting Causal-LM, MNTP, and contrastive training.

    Automatically switches between cgte_loss and contrastive_loss_positive_aware
    based on the presence of func_ids in the batch.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if pooling_head is not None:
        trainable += list(pooling_head.parameters())

    total_steps = (len(dataloader) // grad_accum) * epochs
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

        accumulated_embs = []
        accumulated_batches = []

        for step, batch in enumerate(tqdm(dataloader, desc=f"epoch {epoch + 1}")):
            if max_steps is not None and step >= max_steps:
                log_fn(f"  reached max_steps={max_steps}, stopping epoch early")
                break
            if contrastive:
                with torch.no_grad():
                    lpos = batch.pop("label_positions")
                    func_ids = batch.pop("func_ids", None)
                    if func_ids is not None:
                        func_ids = func_ids.to(device)
                    b_dev = {k: v.to(device) for k, v in batch.items()}
                    if torch.isnan(b_dev["nova_attention_mask"]).any():
                        skipped += 1
                        continue
                        
                    out = model(**b_dev, output_hidden_states=True)
                    embs = pooling_head(out.hidden_states[-1], lpos)
                    accumulated_embs.append(embs.detach())
                    accumulated_batches.append((b_dev, lpos, func_ids))
                
                if len(accumulated_embs) == grad_accum or (step + 1) == len(dataloader):
                    full_embs = torch.cat(accumulated_embs, dim=0)
                    full_embs.requires_grad_(True)
                    
                    if accumulated_batches[0][2] is not None:
                        full_func_ids = torch.cat([x[2] for x in accumulated_batches], dim=0)
                        loss = contrastive_loss_positive_aware(
                            full_embs,
                            full_func_ids,
                            temperature=temperature,
                        )
                    else:
                        x_emb = full_embs[0::2]
                        y_emb = full_embs[1::2]
                        loss = cgte_loss(x_emb, y_emb, temperature=temperature)
                        
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
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    accumulated_embs = []
                    accumulated_batches = []
                    
                    if (step // grad_accum) > 0 and (step // grad_accum) % log_interval == 0:
                        w = losses[-log_interval:]
                        avg = sum(w) / len(w)
                        msg = (
                            f"  step {step:>6}/{len(dataloader)}"
                            f"  loss={avg:.4f}  skipped={skipped}"
                        )
                        tqdm.write(msg)
                        log_fn(msg)
                        if wandb_log is not None:
                            wandb_log({"train/loss": avg, "train/step": step, "train/skipped": skipped})

            else:
                b = {k: v.to(device) for k, v in batch.items()}
                if torch.isnan(b["nova_attention_mask"]).any():
                    skipped += 1
                    optimizer.zero_grad()
                    continue
                
                if mntp:
                    outputs = model(**b)
                    logits = outputs.logits
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, logits.size(-1)), b["labels"].view(-1))
                else:
                    loss = model(**b).loss

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
                    msg = (
                        f"  step {step:>6}/{len(dataloader)}"
                        f"  loss={avg:.4f}  skipped={skipped}"
                    )
                    tqdm.write(msg)
                    log_fn(msg)
                    if wandb_log is not None:
                        wandb_log({"train/loss": avg, "train/step": step, "train/skipped": skipped})

        if not contrastive and losses and len(losses) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if losses:
            log_fn(
                f"  epoch {epoch+1} done  avg_loss={sum(losses)/len(losses):.4f}"
                f"  valid={len(losses)}  skipped={skipped}"
            )
