import argparse
import gc
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../../")))

from shared.collators import PairCollator
from shared.data_utils import asm_to_text, load_jsonl, parse_bench_opt, set_seed
from shared.losses import contrastive_loss_positive_aware
from shared.nova_utils import load_nova, setup_nova_tokenizer
from shared.pooling import MeanLabelPooling

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DORA_CFG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    use_dora=True,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)


def default_train_path():
    repo_root = os.path.abspath(os.path.join(script_dir, "../../../"))
    candidates = [
        os.path.join(repo_root, "nvemb", "output_benchset_rebalanced_train_nova.jsonl"),
        os.path.join(repo_root, "output_benchset_rebalanced_train_nova.jsonl"),
        "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_train_nova.jsonl",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Nova for BCSD with inst/label-token mean pooling."
    )
    parser.add_argument("--output_dir", default="./model_checkpoints/nova_teacher_instmean_bench")
    parser.add_argument("--train_data", default=default_train_path())
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--cross_arch_pair_prob", type=float, default=0.0)
    return parser.parse_args()


def ensure_new_output_dir(output_dir):
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(
            "Refusing to overwrite a non-empty output directory:\n"
            f"  {output_dir}\n"
            "Choose a fresh --output_dir."
        )
    os.makedirs(output_dir, exist_ok=True)


def make_logger(output_dir):
    log_path = os.path.join(output_dir, f"log_instmean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def log(msg, print_to_stdout=True):
        if print_to_stdout:
            print(msg)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    return log, log_path


def sample_arch(sample):
    return parse_bench_opt(sample["opt"])["architecture"]


def build_bench_pairs(samples, seed=42, cross_arch_pair_prob=0.0):
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

        by_arch = defaultdict(list)
        for variant in variants:
            by_arch[sample_arch(variant)].append(variant)

        shuffled = variants[:]
        rng.shuffle(shuffled)
        func_int = func_id_to_int[fid]

        for idx, query in enumerate(shuffled):
            q_arch = sample_arch(query)
            positive = None
            if len(by_arch) >= 2 and rng.random() < cross_arch_pair_prob:
                other_archs = [arch for arch in by_arch if arch != q_arch]
                if other_archs:
                    positive = rng.choice(by_arch[rng.choice(other_archs)])
            if positive is None:
                positive = shuffled[(idx + 1) % len(shuffled)]

            if sample_arch(positive) == q_arch:
                same_arch_count += 1
            else:
                cross_arch_count += 1
            pairs.append((asm_to_text(query["asm"]), asm_to_text(positive["asm"]), func_int))

    rng.shuffle(pairs)
    return pairs, {
        "functions": len(grouped),
        "skipped": skipped,
        "cross_arch_positives": cross_arch_count,
        "same_arch_positives": same_arch_count,
    }


def save_metadata(output_dir, base_tokenizer, args, pair_stats, history, log_path):
    base_tokenizer.save_pretrained(output_dir)
    config = {
        "model": "nova_teacher_instmean",
        "pooling": "mean_label_positions",
        "pooling_has_trainable_parameters": False,
        "note": "No CLS token and no learned pooling head; mean over Nova label/instruction final hidden states.",
        "train_data": os.path.abspath(args.train_data),
        "epochs": args.epochs,
        "batch_size_pairs": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_pairs_per_optimizer_step": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "max_length": args.max_length,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "temperature": args.temperature,
        "cross_arch_pair_prob": args.cross_arch_pair_prob,
        "seed": args.seed,
        "pair_stats": pair_stats,
        "dora": DORA_CFG,
        "log_path": os.path.abspath(log_path),
    }
    with open(os.path.join(output_dir, "train_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    with open(os.path.join(output_dir, "training_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    ensure_new_output_dir(output_dir)
    log, log_path = make_logger(output_dir)

    set_seed(args.seed)
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Train data not found: {args.train_data}")

    log("Nova inst-mean BCSD fine-tune")
    log(f"Output directory: {output_dir}")
    log(f"Train data:       {args.train_data}")
    log("Pooling:          mean over Nova label/instruction token positions")
    log("Objective:        contrastive only")

    base_tokenizer, nova_tokenizer, _ = setup_nova_tokenizer()
    pad_id = base_tokenizer.pad_token_id or 0
    train_samples = load_jsonl(args.train_data)
    pairs, pair_stats = build_bench_pairs(
        train_samples,
        seed=args.seed,
        cross_arch_pair_prob=args.cross_arch_pair_prob,
    )
    if not pairs:
        raise RuntimeError("No bench pairs were built; check dataset path and opt/id schema.")
    log(
        f"Built {len(pairs)} pairs from {pair_stats['functions']} functions "
        f"({pair_stats['skipped']} skipped) | "
        f"cross-arch positives={pair_stats['cross_arch_positives']} "
        f"same-arch positives={pair_stats['same_arch_positives']}"
    )

    model = load_nova(device_id=0, base_tokenizer=base_tokenizer)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, **DORA_CFG))
    model.print_trainable_parameters()
    device = next(model.parameters()).device

    pooling = MeanLabelPooling().to(device)
    loader = DataLoader(
        pairs,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PairCollator(nova_tokenizer, max_length=args.max_length),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    optimizer = AdamW(trainable_params, lr=args.lr, eps=1e-7)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    log(f"Trainable params: {sum(p.numel() for p in trainable_params):,} | optimizer steps: {total_steps}")

    history = []
    global_step = 0
    model.train()
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        losses = []
        running_loss = 0.0
        running_count = 0

        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}")):
            batch_inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k not in ("label_positions", "func_ids")
            }
            label_positions = batch["label_positions"]
            func_ids = batch["func_ids"].to(device)
            key_padding_mask = batch_inputs["input_ids"].eq(pad_id)

            outputs = model(**batch_inputs, output_hidden_states=True)
            embeddings = pooling(outputs.hidden_states[-1], label_positions, key_padding_mask=key_padding_mask)
            loss = contrastive_loss_positive_aware(embeddings, func_ids, temperature=args.temperature)
            losses.append(loss.item())
            running_loss += loss.item()
            running_count += 1
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_interval == 0:
                    log(f"Step {global_step}: train_contrastive={running_loss / max(1, running_count):.4f}", False)
                    running_loss = 0.0
                    running_count = 0

            del outputs, embeddings, loss

        epoch_loss = sum(losses) / max(1, len(losses))
        history.append({"epoch": epoch + 1, "train_contrastive_loss": epoch_loss, "optimizer_steps": global_step})
        log(f"Epoch {epoch + 1} train contrastive loss: {epoch_loss:.4f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    model.save_pretrained(output_dir)
    save_metadata(output_dir, base_tokenizer, args, pair_stats, history, log_path)
    log(f"Saved LoRA/DoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
