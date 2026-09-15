#!/usr/bin/env python3
"""Evaluate one frozen native-causal Nova pooling variant on the bench set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from binarycorp_bench.eval_bench import compute_report, print_report_summary  # noqa: E402
from binarycorp_bench.nova_causal_pooling.core import (  # noqa: E402
    CausalNovaPairCollator,
    FrozenCausalNova,
    LinearMeanProbe,
    MaskedAttentionPooling,
    masked_mean_pool,
)
from shared.data_utils import asm_to_text, build_eval_pairs, get_embeddings_dir, load_jsonl, set_seed  # noqa: E402
from shared.nova_utils import MODEL_ID, NOVA_CACHE_DIR, load_nova  # noqa: E402
from shared.profiling import InferenceProfiler  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pooling",
        required=True,
        choices=("mean", "linear", "attention"),
        help="Run variants separately so latency and memory remain attributable.",
    )
    parser.add_argument(
        "--data",
        default=str(BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_test_nova.jsonl"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Training run directory; required for linear and attention pooling.",
    )
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


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


def resolve_checkpoint_inputs(
    pooling: str,
    checkpoint_arg: str | None,
    checkpoint_kind: str,
) -> Tuple[Path | None, Path | None]:
    if checkpoint_arg is None:
        return None, None

    checkpoint_path = Path(checkpoint_arg).resolve()
    if checkpoint_path.is_file():
        run_dir = checkpoint_path.parent
        return run_dir, checkpoint_path

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Missing checkpoint path: {checkpoint_path}")

    if (checkpoint_path / "config.json").is_file():
        run_dir = checkpoint_path
    elif (checkpoint_path.parent / "config.json").is_file():
        run_dir = checkpoint_path.parent
    else:
        raise FileNotFoundError(
            "Missing training config. Pass the training run directory, for example "
            ".../model_checkpoints/nova_causal_pooling/20260719_130844"
        )

    if pooling == "mean":
        return run_dir, None

    filename = f"{pooling}_head_{checkpoint_kind}.pt"
    head_path = run_dir / filename
    if not head_path.is_file():
        legacy_path = checkpoint_path / filename
        if legacy_path.is_file():
            head_path = legacy_path
        else:
            raise FileNotFoundError(f"Missing head checkpoint: {head_path}")
    return run_dir, head_path


def load_training_config(run_dir: Path | None) -> Dict:
    if run_dir is None:
        return {}
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing training config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("mask_symmetrized") is not False:
        raise ValueError("Checkpoint is not marked as native-causal/no-symmetrization")
    if config.get("backbone_trainable") is not False or config.get("adapter") is not None:
        raise ValueError("Checkpoint provenance is not frozen raw Nova")
    return config


def load_head(
    pooling: str,
    hidden_dim: int,
    checkpoint_path: Path | None,
    device: torch.device,
):
    if pooling == "mean":
        return None, None
    if checkpoint_path is None:
        raise ValueError("--checkpoint-dir is required for a trained pooling head")

    if pooling == "linear":
        head = LinearMeanProbe(hidden_dim)
    else:
        head = MaskedAttentionPooling(hidden_dim)

    payload = torch.load(checkpoint_path, map_location="cpu")
    head.load_state_dict(payload["state_dict"])
    head = head.to(device=device, dtype=torch.bfloat16).eval()
    return head, checkpoint_path


def pool_embeddings(
    pooling: str,
    hidden: torch.Tensor,
    pool_mask: torch.Tensor,
    head,
) -> torch.Tensor:
    mean_embedding = masked_mean_pool(hidden, pool_mask)
    if pooling == "mean":
        return mean_embedding
    if pooling == "linear":
        return head(mean_embedding)
    return head(hidden, pool_mask)


@torch.no_grad()
def encode_pairs(
    backbone: FrozenCausalNova,
    head,
    pooling: str,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    profiler = InferenceProfiler(device)
    query_embeddings = []
    target_embeddings = []

    # Warm up kernels and allocator without contaminating measurements.
    first_batch = next(iter(loader))
    warm = {key: value.to(device) for key, value in first_batch.items()}
    hidden = backbone(
        warm["input_ids"], warm["attention_mask"], warm["nova_attention_mask"]
    )
    _ = pool_embeddings(pooling, hidden, warm["pool_mask"], head)
    del warm, hidden
    torch.cuda.synchronize(device)

    for cpu_batch in tqdm(loader, desc=f"eval {pooling}"):
        batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
        with profiler:
            hidden = backbone(
                batch["input_ids"],
                batch["attention_mask"],
                batch["nova_attention_mask"],
            )
            embeddings = pool_embeddings(pooling, hidden, batch["pool_mask"], head)
            embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
        profiler.total_samples += embeddings.shape[0]
        query_embeddings.append(embeddings[0::2].cpu())
        target_embeddings.append(embeddings[1::2].cpu())
        del batch, hidden, embeddings

    return (
        torch.cat(query_embeddings, dim=0),
        torch.cat(target_embeddings, dim=0),
        profiler.get_stats(),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This 1.3B Nova experiment requires a CUDA device")

    run_dir, head_checkpoint_path = resolve_checkpoint_inputs(
        args.pooling,
        args.checkpoint_dir,
        args.checkpoint_kind,
    )
    training_config = load_training_config(run_dir)
    max_length = args.max_length or int(training_config.get("max_length", 1024))
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}")

    samples = load_jsonl(args.data)
    eval_pairs = build_eval_pairs(samples, seed=args.seed)
    if args.max_pairs is not None:
        eval_pairs = eval_pairs[: args.max_pairs]
    if not eval_pairs:
        raise RuntimeError("No evaluation pairs were built")

    pair_rows = [
        (
            asm_to_text(pair["query"]["asm"]),
            asm_to_text(pair["target"]["asm"]),
            pair_index,
        )
        for pair_index, pair in enumerate(eval_pairs)
    ]

    print("Loading official Nova-1.3B weights with native causal masks...", flush=True)
    nova_tokenizer, causal_lm = load_official_nova(args.device_id)
    backbone = FrozenCausalNova(causal_lm)
    collator = CausalNovaPairCollator(nova_tokenizer, max_length=max_length)
    loader = DataLoader(
        pair_rows,
        batch_size=args.pair_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    head, checkpoint_path = load_head(
        args.pooling,
        backbone.hidden_dim,
        head_checkpoint_path,
        device,
    )

    query_embeddings, target_embeddings, stats = encode_pairs(
        backbone, head, args.pooling, loader, device
    )
    result = {
        "ids": [pair["id"] for pair in eval_pairs],
        "query_opts": [pair["query"]["opt"] for pair in eval_pairs],
        "target_opts": [pair["target"]["opt"] for pair in eval_pairs],
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
        "stats": stats,
        "experiment": {
            "model": MODEL_ID,
            "model_snapshot": Path(NOVA_CACHE_DIR).name,
            "weights": "official_foundation_model",
            "backbone_trainable": False,
            "adapter": None,
            "mask": "official_native_hierarchical_causal",
            "mask_symmetrized": False,
            "pooling": args.pooling,
            "pooling_positions": "assembly_tokens_only",
            "query_instruction": True,
            "max_length": max_length,
            "head_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        },
    }

    model_name = f"nova_causal_{args.pooling}"
    default_dir = Path(get_embeddings_dir("binarycorp_bench", model_name))
    output_path = Path(args.output or (default_dir / "eval_bench_embeddings.pt"))
    report_path = Path(args.report or (default_dir / "eval_bench_metrics.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)

    report = compute_report(result)
    report["model"] = model_name
    report["data"] = str(Path(args.data).resolve())
    report["source"] = str(output_path.resolve())
    report["experiment"] = result["experiment"]
    report["inference"] = stats
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print_report_summary(report)
    print(
        f"\n{model_name}: {stats['avg_ms_per_sample']:.2f} ms/function, "
        f"{stats['peak_memory_mb']:.1f} MB peak"
    )
    print(f"Embeddings: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
