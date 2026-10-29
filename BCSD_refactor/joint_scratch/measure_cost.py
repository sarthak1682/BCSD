#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from joint_scratch.data import JointScratchCollator  # noqa: E402
from joint_scratch.evaluate import (  # noqa: E402
    RetrievalSampleCollator,
    autocast_context,
    load_run_config,
    resolve_checkpoint,
)
from joint_scratch.model import ScratchJointModel  # noqa: E402
from joint_scratch.tokenizer import load_nova_tokenizer  # noqa: E402
from joint_scratch.train import resolve_device  # noqa: E402
from shared.data_utils import build_eval_pairs, load_jsonl  # noqa: E402
from shared.profiling import InferenceProfiler  # noqa: E402

DEFAULT_TEST_PATH = str(
    REPO_ROOT.parent / "nvemb"
    / "output_benchset_rebalanced_test_clap_strict.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or run dir.")
    parser.add_argument("--data", default=DEFAULT_TEST_PATH)
    parser.add_argument("--split", choices=("best", "final", "direct"), default="final")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


@torch.no_grad()
def encode(
    model,
    samples,
    collator,
    device,
    *,
    batch_size,
    num_workers,
    amp,
    profiler,
    desc,
):
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
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast_context(device, amp):
            if profiler is not None:
                with profiler:
                    output = model.forward_retrieval(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["pool_mask"],
                    )
            else:
                output = model.forward_retrieval(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["pool_mask"],
                )
        if profiler is not None:
            profiler.total_samples += batch["input_ids"].shape[0]
        embeddings.append(output.embeddings.detach().float().cpu())
    return torch.cat(embeddings, dim=0)


def main() -> None:
    args = parse_args()
    checkpoint_arg = Path(args.checkpoint).resolve()
    checkpoint_dir = resolve_checkpoint(checkpoint_arg, args.split)
    run_config = load_run_config(checkpoint_arg, checkpoint_dir)
    device = resolve_device(args.device)

    model = ScratchJointModel.from_pretrained(checkpoint_dir, map_location="cpu").to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded joint-scratch model with {n_params/1e6:.1f}M parameters on {device}")

    _, nova_tokenizer = load_nova_tokenizer()
    model_cfg = run_config.get("model", {})
    max_length = (
        args.max_length
        or model_cfg.get("max_length")
        or run_config.get("max_length")
        or model.config.max_length
    )
    collator = JointScratchCollator(
        nova_tokenizer,
        max_length=max_length,
        lm_target_fraction=run_config.get("lm_target_fraction", 0.5),
        mlm_probability=run_config.get("mlm_probability", 0.15),
        raw_chars_per_token=run_config.get("raw_chars_per_token", 8),
    )

    samples = load_jsonl(args.data)
    pairs = build_eval_pairs(samples, seed=run_config.get("seed", 42))
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    queries = [pair["query"] for pair in pairs]
    targets = [pair["target"] for pair in pairs]
    amp = bool(args.amp and device.type == "cuda")

    print("Warm-up pass...")
    encode(
        model,
        queries[: args.batch_size],
        collator,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=amp,
        profiler=None,
        desc="warmup",
    )

    profiler = InferenceProfiler(device)
    print("Profiling query pass...")
    encode(
        model,
        queries,
        collator,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=amp,
        profiler=profiler,
        desc="queries",
    )
    print("Profiling target pass...")
    encode(
        model,
        targets,
        collator,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=amp,
        profiler=profiler,
        desc="targets",
    )

    stats = profiler.get_stats()
    payload = {
        "checkpoint": str(checkpoint_dir),
        "data": str(Path(args.data).resolve()),
        "pairs": len(pairs),
        "encoded_functions": profiler.total_samples,
        "batch_size": args.batch_size,
        "parameters_millions": round(n_params / 1e6, 1),
        "latency_ms_per_function": round(stats["avg_ms_per_sample"], 4),
        "peak_memory_mb": round(stats["peak_memory_mb"], 1),
        "device": str(device),
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
