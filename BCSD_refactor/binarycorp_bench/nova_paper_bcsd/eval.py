#!/usr/bin/env python3
"""Evaluate the Nova-paper BCSD fine-tuning baseline on the bench set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from binarycorp_bench.eval_bench import compute_report, print_report_summary  # noqa: E402
from binarycorp_bench.nova_paper_bcsd.core import (  # noqa: E402
    PaperBCSDTextCollator,
    mean_label_pool,
)
from shared.data_utils import asm_to_text, build_eval_pairs, get_embeddings_dir, load_jsonl, set_seed  # noqa: E402
from shared.nova_utils import MODEL_ID, NOVA_CACHE_DIR, load_nova  # noqa: E402
from shared.profiling import InferenceProfiler  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Training run directory, or a concrete best/final checkpoint directory.",
    )
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument(
        "--data",
        default=str(BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_test_nova.jsonl"),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def resolve_checkpoint(checkpoint_arg: str, checkpoint_kind: str) -> Tuple[Path, Path, Dict]:
    path = Path(checkpoint_arg).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint path: {path}")

    if (path / "run_config.json").is_file():
        run_dir = path
        checkpoint_dir = run_dir / checkpoint_kind
    else:
        checkpoint_dir = path
        run_dir = path.parent

    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        metadata_path = checkpoint_dir / "paper_bcsd_metadata.json"
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            return run_dir, checkpoint_dir, metadata["config"]
        raise FileNotFoundError(
            "Missing run_config.json. Pass the training run directory or its best/final subdirectory."
        )

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing {checkpoint_kind} checkpoint: {checkpoint_dir}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("mask_symmetrized") is not False:
        raise ValueError("Checkpoint is not marked as native-causal/no-symmetrization")
    if config.get("query_instruction") is not False:
        raise ValueError("Checkpoint is not marked as assembly-only/no instruction prompt")
    return run_dir, checkpoint_dir, config


def load_official_tokenizer():
    if NOVA_CACHE_DIR not in sys.path:
        sys.path.insert(0, NOVA_CACHE_DIR)
    from modeling_nova import NovaTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=NOVA_CACHE_DIR,
        local_files_only=True,
    )
    return NovaTokenizer(base_tokenizer)


def load_model(checkpoint_dir: Path, config: Dict, device_id: int):
    tuning = config.get("tuning")
    if tuning == "dora":
        from peft import PeftModel

        base_model = load_nova(device_id=device_id, base_tokenizer=None)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    elif tuning == "full":
        model = load_nova(from_dir=str(checkpoint_dir), device_id=device_id, base_tokenizer=None)
    else:
        raise ValueError(f"Unsupported tuning mode in checkpoint config: {tuning!r}")

    model.config.use_cache = False
    model.eval()
    return model


def backbone_module(model):
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base_model.model


@torch.no_grad()
def encode_texts(
    model,
    collator: PaperBCSDTextCollator,
    texts,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    profiler: InferenceProfiler,
) -> torch.Tensor:
    loader = DataLoader(
        texts,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    first_batch = next(iter(loader))
    warm = {key: value.to(device) for key, value in first_batch.items()}
    hidden = backbone_module(model)(
        input_ids=warm["input_ids"],
        attention_mask=warm["attention_mask"],
        nova_attention_mask=warm["nova_attention_mask"],
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    ).last_hidden_state
    _ = mean_label_pool(hidden, warm["label_mask"])
    del warm, hidden
    torch.cuda.synchronize(device)

    embeddings = []
    for cpu_batch in tqdm(loader, desc="encode"):
        batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
        with profiler:
            hidden = backbone_module(model)(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                nova_attention_mask=batch["nova_attention_mask"],
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            pooled = mean_label_pool(hidden, batch["label_mask"])
            pooled = F.normalize(pooled.float(), p=2, dim=1)
        profiler.total_samples += pooled.shape[0]
        embeddings.append(pooled.cpu())
        del batch, hidden, pooled
    return torch.cat(embeddings, dim=0)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This 1.3B Nova evaluation requires a CUDA device")

    run_dir, checkpoint_dir, config = resolve_checkpoint(args.checkpoint_dir, args.checkpoint_kind)
    max_length = args.max_length or int(config.get("max_length", 2048))
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}")

    samples = load_jsonl(args.data)
    eval_pairs = build_eval_pairs(samples, seed=args.seed)
    if args.max_pairs is not None:
        eval_pairs = eval_pairs[: args.max_pairs]
    if not eval_pairs:
        raise RuntimeError("No evaluation pairs were built")

    query_texts = [asm_to_text(pair["query"]["asm"]) for pair in eval_pairs]
    target_texts = [asm_to_text(pair["target"]["asm"]) for pair in eval_pairs]

    print(f"Loading Nova-paper BCSD checkpoint: {checkpoint_dir}", flush=True)
    nova_tokenizer = load_official_tokenizer()
    model = load_model(checkpoint_dir, config, args.device_id)
    collator = PaperBCSDTextCollator(nova_tokenizer, max_length=max_length)
    profiler = InferenceProfiler(device)

    print("Encoding queries...", flush=True)
    query_embeddings = encode_texts(
        model,
        collator,
        query_texts,
        batch_size=args.pair_batch_size,
        num_workers=args.num_workers,
        device=device,
        profiler=profiler,
    )
    print("Encoding targets...", flush=True)
    target_embeddings = encode_texts(
        model,
        collator,
        target_texts,
        batch_size=args.pair_batch_size,
        num_workers=args.num_workers,
        device=device,
        profiler=profiler,
    )
    stats = profiler.get_stats()

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
            "checkpoint": str(checkpoint_dir),
            "training_run": str(run_dir),
            "tuning": config.get("tuning"),
            "mask": "official_native_hierarchical_causal",
            "mask_symmetrized": False,
            "pooling": "mean_last_layer_hidden_states_of_nova_label_tokens",
            "query_instruction": False,
            "train_pairing": config.get("train_pairing"),
            "eval_normalized": True,
            "max_length": max_length,
        },
    }

    model_name = f"nova_paper_bcsd_{config.get('tuning')}"
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

