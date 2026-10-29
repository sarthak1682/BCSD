#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from binarycorp_bench.eval_bench import compute_report, print_report_summary  # noqa: E402
from joint_scratch.data import (  # noqa: E402
    JointPairDataset,
    JointScratchCollator,
    load_variant_pairs,
)
from joint_scratch.model import ScratchJointModel  # noqa: E402
from joint_scratch.train import LossWeights, resolve_device, validate  # noqa: E402
from shared.data_utils import build_eval_pairs, load_jsonl  # noqa: E402


DEFAULT_TEST_PATH = str(
    REPO_ROOT.parent / "nvemb"
    / "output_benchset_rebalanced_test_clap_strict.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or run dir.")
    parser.add_argument("--data", default=DEFAULT_TEST_PATH)
    parser.add_argument("--split", choices=("best", "final", "direct"), default="best")
    parser.add_argument(
        "--protocol",
        choices=("bench", "training-pairs"),
        default="bench",
        help=(
            "bench matches binarycorp_bench/eval_bench.py; training-pairs keeps "
            "the internal diagnostic loss/identity-recall metric."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-eval-pairs", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--lm-target-fraction", type=float, default=None)
    parser.add_argument("--raw-chars-per-token", type=int, default=None)
    parser.add_argument("--contrastive-temperature", type=float, default=None)
    parser.add_argument("--lambda-lm", type=float, default=None)
    parser.add_argument("--lambda-retrieval", type=float, default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bfloat16 autocast on CUDA.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_checkpoint(path: Path, split: str) -> Path:
    if split == "direct":
        return path
    candidate = path / split
    if candidate.is_dir():
        return candidate
    if (path / "model_config.json").is_file():
        return path
    raise FileNotFoundError(f"could not find checkpoint at {candidate} or {path}")


def load_run_config(checkpoint_arg: Path, checkpoint_dir: Path) -> dict:
    candidates = [
        checkpoint_arg / "run_config.json",
        checkpoint_dir.parent / "run_config.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class RetrievalSampleCollator:
    def __init__(self, base_collator: JointScratchCollator) -> None:
        self.base_collator = base_collator

    def __call__(self, batch):
        rows = [self.base_collator._encode_retrieval(sample["asm"]) for sample in batch]
        length = max(len(row) for row in rows)
        input_ids = torch.full(
            (len(rows), length),
            self.base_collator.pad_id,
            dtype=torch.long,
        )
        attention = torch.zeros((len(rows), length), dtype=torch.bool)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.from_numpy(row)
            attention[index, : len(row)] = True
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "pool_mask": attention.clone(),
        }


@torch.no_grad()
def encode_retrieval_samples(
    model: ScratchJointModel,
    samples,
    collator: JointScratchCollator,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    amp: bool,
    desc: str,
) -> torch.Tensor:
    loader = DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=RetrievalSampleCollator(collator),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    embeddings = []
    for batch in tqdm(loader, desc=desc):
        batch = {
            key: value.to(device, non_blocking=True) for key, value in batch.items()
        }
        with autocast_context(device, amp):
            output = model.forward_retrieval(
                batch["input_ids"],
                batch["attention_mask"],
                batch["pool_mask"],
            )
        embeddings.append(output.embeddings.detach().float().cpu())
    return torch.cat(embeddings, dim=0)


def build_collator(args: argparse.Namespace, run_config: dict) -> JointScratchCollator:
    from joint_scratch.tokenizer import load_nova_tokenizer

    _, nova_tokenizer = load_nova_tokenizer()
    max_length = args.max_length or run_config.get("model", {}).get("max_length", 512)
    return JointScratchCollator(
        nova_tokenizer,
        max_length=max_length,
        lm_target_fraction=(
            args.lm_target_fraction
            if args.lm_target_fraction is not None
            else run_config.get("lm_target_fraction", 0.5)
        ),
        mlm_probability=run_config.get("mlm_probability", 0.15),
        raw_chars_per_token=(
            args.raw_chars_per_token
            if args.raw_chars_per_token is not None
            else run_config.get("raw_chars_per_token", 8)
        ),
    )


def evaluate_bench(
    args: argparse.Namespace,
    run_config: dict,
    checkpoint_dir: Path,
    model: ScratchJointModel,
    collator: JointScratchCollator,
    device: torch.device,
) -> dict:
    samples = load_jsonl(args.data)
    pairs = build_eval_pairs(samples, seed=run_config.get("seed", 42))
    if args.max_eval_pairs is not None:
        pairs = pairs[: args.max_eval_pairs]
    if not pairs:
        raise RuntimeError("evaluation requires at least one pair")

    batch_size = args.batch_size or run_config.get("pair_batch_size", 64)
    num_workers = (
        args.num_workers if args.num_workers is not None else run_config.get("num_workers", 4)
    )
    amp = bool(args.amp and device.type == "cuda")
    result = {
        "ids": [pair["id"] for pair in pairs],
        "query_opts": [pair["query"]["opt"] for pair in pairs],
        "target_opts": [pair["target"]["opt"] for pair in pairs],
        "query_embeddings": encode_retrieval_samples(
            model,
            [pair["query"] for pair in pairs],
            collator,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            amp=amp,
            desc="encoding queries",
        ),
        "target_embeddings": encode_retrieval_samples(
            model,
            [pair["target"] for pair in pairs],
            collator,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            amp=amp,
            desc="encoding targets",
        ),
    }
    report = compute_report(result)
    print_report_summary(report)
    return {
        "checkpoint": str(checkpoint_dir),
        "data": str(Path(args.data).resolve()),
        "protocol": "bench",
        "pairs": len(pairs),
        "batch_size": batch_size,
        "device": str(device),
        "report": report,
    }


def evaluate_training_pairs(
    args: argparse.Namespace,
    run_config: dict,
    checkpoint_dir: Path,
    model: ScratchJointModel,
    collator: JointScratchCollator,
    device: torch.device,
) -> dict:
    pairs = load_variant_pairs(args.data, seed=run_config.get("seed", 42))
    if args.max_eval_pairs is not None:
        pairs = pairs[: args.max_eval_pairs]
    if not pairs:
        raise RuntimeError("evaluation requires at least one pair")

    batch_size = args.batch_size or run_config.get("pair_batch_size", 64)
    num_workers = (
        args.num_workers if args.num_workers is not None else run_config.get("num_workers", 4)
    )
    loader = DataLoader(
        JointPairDataset(pairs),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    weights_config = run_config.get("loss_weights", {})
    weights = LossWeights(
        lm=args.lambda_lm if args.lambda_lm is not None else weights_config.get("lm", 1.0),
        mlm=weights_config.get("mlm", 1.0),
        retrieval=(
            args.lambda_retrieval
            if args.lambda_retrieval is not None
            else weights_config.get("retrieval", 1.0)
        ),
    )
    eval_args = argparse.Namespace(
        contrastive_temperature=(
            args.contrastive_temperature
            if args.contrastive_temperature is not None
            else run_config.get("contrastive_temperature", 0.05)
        ),
        amp=bool(args.amp and device.type == "cuda"),
        val_batches=args.val_batches,
        retrieval_recall_device=device if device.type == "cuda" else None,
        training_objectives=run_config.get("training_objectives", "lm_retrieval"),
    )
    metrics = validate(model, loader, device, eval_args, weights)
    return {
        "checkpoint": str(checkpoint_dir),
        "data": str(Path(args.data).resolve()),
        "protocol": "training-pairs",
        "pairs": len(pairs),
        "batch_size": batch_size,
        "device": str(device),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    checkpoint_arg = Path(args.checkpoint).resolve()
    checkpoint_dir = resolve_checkpoint(checkpoint_arg, args.split)
    run_config = load_run_config(checkpoint_arg, checkpoint_dir)
    device = resolve_device(args.device)
    model = ScratchJointModel.from_pretrained(checkpoint_dir, map_location="cpu").to(device)
    model.eval()
    collator = build_collator(args, run_config)

    if args.protocol == "bench":
        payload = evaluate_bench(
            args, run_config, checkpoint_dir, model, collator, device
        )
    else:
        payload = evaluate_training_pairs(
            args, run_config, checkpoint_dir, model, collator, device
        )

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
