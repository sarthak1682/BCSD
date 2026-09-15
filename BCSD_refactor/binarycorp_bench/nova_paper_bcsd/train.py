#!/usr/bin/env python3
"""Fine-tune native-causal Nova for BCSD using the paper's [INST] pooling."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from binarycorp_bench.nova_paper_bcsd.core import (  # noqa: E402
    PaperBCSDCollator,
    bcsd_l2_contrastive_loss,
    build_o0_o3_pairs,
    mean_label_pool,
)
from shared.data_utils import group_samples_by_id, load_jsonl, parse_bench_opt, set_seed  # noqa: E402
from shared.nova_utils import MODEL_ID, NOVA_CACHE_DIR, load_nova  # noqa: E402


@dataclass
class RunConfig:
    data: str
    output_dir: str
    seed: int
    tuning: str
    epochs: int
    pair_batch_size: int
    grad_accum: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    temperature: float
    max_length: int
    val_fraction: float
    pairs_per_function: int
    num_workers: int
    device_id: int
    max_train_pairs: Optional[int]
    max_val_pairs: Optional[int]
    dora_rank: int
    dora_alpha: int
    dora_dropout: float
    backbone: str = MODEL_ID
    backbone_snapshot: str = Path(NOVA_CACHE_DIR).name
    mask: str = "official_native_hierarchical_causal"
    mask_symmetrized: bool = False
    query_instruction: bool = False
    pooling: str = "mean_last_layer_hidden_states_of_nova_label_tokens"
    train_pairing: str = "O0_query_to_O3_target"
    train_loss: str = "cross_entropy_over_negative_l2_distance"
    paper_reference: str = "Nova ICLR 2025 Sections 3.5, 4.3, Appendix A.2"


def parse_args() -> argparse.Namespace:
    default_data = BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_train_nova.jsonl"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(default_data))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tuning",
        choices=("dora", "full"),
        default="dora",
        help="Use DoRA adapters for feasible single-GPU runs, or full for the closest paper-faithful run.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=8,
        help="O0/O3 function pairs per forward; Nova sees twice this many sequences.",
    )
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--pairs-per-function", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--dora-rank", type=int, default=16)
    parser.add_argument("--dora-alpha", type=int, default=32)
    parser.add_argument("--dora-dropout", type=float, default=0.05)
    return parser.parse_args()


def log(message: str, log_path: Path) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def load_official_nova(device_id: int):
    if NOVA_CACHE_DIR not in sys.path:
        sys.path.insert(0, NOVA_CACHE_DIR)
    from modeling_nova import NovaTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=NOVA_CACHE_DIR,
        local_files_only=True,
    )
    nova_tokenizer = NovaTokenizer(base_tokenizer)
    causal_lm = load_nova(device_id=device_id, base_tokenizer=None)
    return nova_tokenizer, causal_lm


def maybe_attach_dora(model, args: argparse.Namespace):
    if args.tuning != "dora":
        model.requires_grad_(True)
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.dora_rank,
        lora_alpha=args.dora_alpha,
        lora_dropout=args.dora_dropout,
        bias="none",
        use_dora=True,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return get_peft_model(model, config)


def backbone_module(model):
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base_model.model


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def encode_batch(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = backbone_module(model)(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        nova_attention_mask=batch["nova_attention_mask"],
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return mean_label_pool(outputs.last_hidden_state, batch["label_mask"])


@torch.no_grad()
def validate(
    model,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> float:
    model.eval()
    total = 0.0
    steps = 0
    for cpu_batch in tqdm(loader, desc="validation", leave=False):
        batch = move_batch(cpu_batch, device)
        embeddings = encode_batch(model, batch)
        loss = bcsd_l2_contrastive_loss(
            embeddings[0::2],
            embeddings[1::2],
            temperature=temperature,
        )
        total += loss.item()
        steps += 1
        del batch, embeddings, loss
    model.train()
    return total / max(steps, 1)


def save_checkpoint(model, checkpoint_dir: Path, config: RunConfig, metadata: Dict) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    with (checkpoint_dir / "paper_bcsd_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": asdict(config), **metadata}, handle, indent=2)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This 1.3B Nova fine-tuning baseline requires a CUDA device")
    if args.epochs < 1 or args.pair_batch_size < 1 or args.grad_accum < 1:
        raise ValueError("epochs, pair-batch-size, and grad-accum must be positive")
    if args.pairs_per_function < 1:
        raise ValueError("pairs-per-function must be positive")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output_dir
        or (REPO_ROOT / "model_checkpoints" / "nova_paper_bcsd" / run_id)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "train.log"

    config = RunConfig(
        data=str(Path(args.data).resolve()),
        output_dir=str(output_dir.resolve()),
        seed=args.seed,
        tuning=args.tuning,
        epochs=args.epochs,
        pair_batch_size=args.pair_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        temperature=args.temperature,
        max_length=args.max_length,
        val_fraction=args.val_fraction,
        pairs_per_function=args.pairs_per_function,
        num_workers=args.num_workers,
        device_id=args.device_id,
        max_train_pairs=args.max_train_pairs,
        max_val_pairs=args.max_val_pairs,
        dora_rank=args.dora_rank,
        dora_alpha=args.dora_alpha,
        dora_dropout=args.dora_dropout,
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    set_seed(args.seed)
    log("Loading bench data and creating an O0/O3 function-disjoint split...", log_path)
    samples = load_jsonl(args.data)
    grouped = group_samples_by_id(samples)
    eligible_ids = []
    for fid, rows in grouped.items():
        opts = {parse_bench_opt(row["opt"])["optimization"] for row in rows}
        if "O0" in opts and "O3" in opts:
            eligible_ids.append(fid)
    random.Random(args.seed).shuffle(eligible_ids)
    val_count = max(1, int(len(eligible_ids) * args.val_fraction))
    val_ids = eligible_ids[:val_count]
    train_ids = eligible_ids[val_count:]

    train_pairs, train_stats = build_o0_o3_pairs(
        samples,
        train_ids,
        seed=args.seed,
        pairs_per_function=args.pairs_per_function,
    )
    val_pairs, val_stats = build_o0_o3_pairs(
        samples,
        val_ids,
        seed=args.seed + 1,
        pairs_per_function=args.pairs_per_function,
    )
    if args.max_train_pairs is not None:
        train_pairs = train_pairs[: args.max_train_pairs]
    if args.max_val_pairs is not None:
        val_pairs = val_pairs[: args.max_val_pairs]
    if not train_pairs or not val_pairs:
        raise RuntimeError("No O0/O3 train or validation pairs were built")

    log(
        f"Eligible functions train/val: {len(train_ids):,}/{len(val_ids):,}; "
        f"pairs train/val: {len(train_pairs):,}/{len(val_pairs):,}",
        log_path,
    )
    log(f"Train pair stats: {train_stats}", log_path)
    log(f"Val pair stats: {val_stats}", log_path)

    log("Loading official Nova-1.3B tokenizer/model (no added [MASK] token)...", log_path)
    nova_tokenizer, causal_lm = load_official_nova(args.device_id)
    model = maybe_attach_dora(causal_lm, args)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()
    device = next(model.parameters()).device

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found")
    log(
        f"Tuning mode: {args.tuning}; trainable params: "
        f"{sum(p.numel() for p in trainable_params):,}",
        log_path,
    )

    collator = PaperBCSDCollator(nova_tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        train_pairs,
        batch_size=args.pair_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_pairs,
        batch_size=args.pair_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-7,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_updates),
        num_training_steps=total_updates,
    )
    log(
        "Mask contract: official native hierarchical causal mask, no "
        "symmetrization; pooling: mean over Nova <label-*> token states.",
        log_path,
    )

    history = []
    best_val = float("inf")
    global_update = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        running_steps = 0

        for step, cpu_batch in enumerate(
            tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"),
            start=1,
        ):
            batch = move_batch(cpu_batch, device)
            embeddings = encode_batch(model, batch)
            loss = bcsd_l2_contrastive_loss(
                embeddings[0::2],
                embeddings[1::2],
                temperature=args.temperature,
            )
            (loss / args.grad_accum).backward()
            running += loss.item()
            running_steps += 1

            should_update = step % args.grad_accum == 0 or step == len(train_loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if global_update % 20 == 0:
                    log(
                        f"update={global_update}/{total_updates} "
                        f"loss={running / running_steps:.4f}",
                        log_path,
                    )
                    running = 0.0
                    running_steps = 0

            del batch, embeddings, loss

        val_loss = validate(model, val_loader, device, args.temperature)
        record = {"epoch": epoch + 1, "global_update": global_update, "val_loss": val_loss}
        history.append(record)
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        log(f"epoch={epoch + 1} validation loss={val_loss:.4f}", log_path)

        metadata = {"epoch": epoch + 1, "validation": record}
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, output_dir / "best", config, metadata)

    final_metadata = {"epoch": args.epochs, "validation": history[-1]}
    save_checkpoint(model, output_dir / "final", config, final_metadata)
    log(f"Training complete: {output_dir}", log_path)


if __name__ == "__main__":
    main()
