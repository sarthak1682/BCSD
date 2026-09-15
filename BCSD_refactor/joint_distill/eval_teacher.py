#!/usr/bin/env python3
"""Evaluate EBM-full causal translation on the student's exact LM protocol."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from joint_distill.data import JointDistillCollator, JointPairDataset  # noqa: E402
from joint_distill.eval import LMOnlyCollator, make_pairs, move_batch  # noqa: E402
from joint_distill.losses import (  # noqa: E402
    shifted_causal_cross_entropy,
    shifted_token_accuracy,
)
from joint_distill.teacher import FullEBMTeacher  # noqa: E402
from shared.data_utils import load_jsonl, set_seed  # noqa: E402
from joint_distill.tokenizer import load_ebm_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-dir",
        default=str(BCSD_ROOT / "model_checkpoints" / "nova_ebm_bench" / "s3_final"),
    )
    parser.add_argument(
        "--data",
        default=str(BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_test_nova.jsonl"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--pair-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1_024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--condition-on-target",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EBM-full evaluation requires CUDA")
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}")

    _, nova_tokenizer = load_ebm_tokenizer()
    samples = load_jsonl(args.data)
    pairs = make_pairs(samples, args.seed)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    collator = JointDistillCollator(
        nova_tokenizer,
        max_length=args.max_length,
        condition_on_target=args.condition_on_target,
    )
    loader = DataLoader(
        JointPairDataset(pairs),
        batch_size=args.pair_batch_size,
        shuffle=False,
        collate_fn=LMOnlyCollator(collator),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    teacher = FullEBMTeacher.load(args.teacher_dir, device_id=args.device_id)

    loss_total = 0.0
    correct = 0
    tokens = 0
    steps = 0
    for cpu_batch in tqdm(loader, desc="EBM-full translation"):
        batch = move_batch(cpu_batch, device)
        outputs = teacher.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            nova_attention_mask=batch["nova_attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        loss = shifted_causal_cross_entropy(outputs.logits, batch["labels"])
        accuracy = shifted_token_accuracy(outputs.logits, batch["labels"])
        loss_total += loss.item()
        correct += accuracy[0]
        tokens += accuracy[1]
        steps += 1

    cross_entropy = loss_total / max(steps, 1)
    report = {
        "model": "nova_ebm_full_s3",
        "teacher_dir": str(Path(args.teacher_dir).resolve()),
        "data": str(Path(args.data).resolve()),
        "condition_on_target": args.condition_on_target,
        "max_length": args.max_length,
        "pairs": len(pairs),
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(min(cross_entropy, 20.0)),
        "token_accuracy": correct / max(tokens, 1),
        "tokens": tokens,
        "batches": steps,
        "metric_scope": "assembly_to_assembly_teacher_forced",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Teacher translation report: {output}")


if __name__ == "__main__":
    main()
