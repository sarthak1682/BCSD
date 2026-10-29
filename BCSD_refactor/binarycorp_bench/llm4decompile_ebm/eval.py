#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from binarycorp_bench.eval_bench import compute_report, print_report_summary  # noqa: E402
from binarycorp_bench.llm4decompile_causal_pooling.core import (  # noqa: E402
    MODEL_ID,
    MaskedAttentionPooling,
    default_jsonl,
    load_llm4decompile,
)
from binarycorp_bench.llm4decompile_ebm.collators import (  # noqa: E402
    L4DAssemblyCollator,
)
from shared.data_utils import (  # noqa: E402
    asm_to_text,
    build_eval_pairs,
    get_embeddings_dir,
    load_jsonl,
    set_seed,
)
from shared.profiling import InferenceProfiler  # noqa: E402


POOLING_HEAD_FNAME = "pooling_head.pt"


def _load_state_dict(path: Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def load_ebm_for_inference(
    checkpoint: str | Path,
    device: torch.device,
    *,
    head_checkpoint: str | Path | None = None,
    bidirectional: bool = False,
):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Missing merged Stage-3 checkpoint: {checkpoint}")
    tokenizer, model = load_llm4decompile(
        str(checkpoint),
        device,
        attn_implementation="eager" if bidirectional else None,
    )
    if bidirectional:
        from binarycorp_bench.llm4decompile_repkd.teacher import (
            patch_decoder_honor_4d_mask,
        )

        model = patch_decoder_honor_4d_mask(model)
    model.eval()
    model._l4d_bidirectional = bool(bidirectional)
    head_path = (
        Path(head_checkpoint).expanduser().resolve()
        if head_checkpoint is not None
        else checkpoint / POOLING_HEAD_FNAME
    )
    if not head_path.is_file():
        raise FileNotFoundError(f"Missing Stage-3 pooling head: {head_path}")
    pooling_head = MaskedAttentionPooling(int(model.config.hidden_size))
    pooling_head.load_state_dict(_load_state_dict(head_path))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    pooling_head = pooling_head.to(device=device, dtype=dtype).eval()
    return tokenizer, model, pooling_head, head_path


def _last_hidden(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    transformer = getattr(model, "model", None)
    if transformer is not None:
        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return outputs.hidden_states[-1]


@torch.no_grad()
def encode_loader(
    model,
    pooling_head,
    loader: DataLoader,
    device: torch.device,
    *,
    description: str = "encode",
    profiler: Optional[InferenceProfiler] = None,
    bidirectional: bool = False,
    pad_id: Optional[int] = None,
) -> torch.Tensor:
    embeddings = []
    model.eval()
    pooling_head.eval()
    if bidirectional and pad_id is None:
        raise ValueError("bidirectional encoding requires pad_id")
    dtype = next(model.parameters()).dtype
    for cpu_batch in tqdm(loader, desc=description):
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in cpu_batch.items()
        }
        attention_mask = batch["attention_mask"]
        if bidirectional:
            from binarycorp_bench.llm4decompile_repkd.teacher import attention_mask_4d

            attention_mask = attention_mask_4d(
                batch["input_ids"], int(pad_id), bidirectional=True, dtype=dtype
            )
        context = profiler if profiler is not None else _NullContext()
        with context:
            hidden = _last_hidden(
                model, batch["input_ids"], attention_mask
            )
            pooled = pooling_head(hidden, batch["pool_mask"])
            pooled = functional.normalize(pooled.float(), p=2, dim=1)
        if profiler is not None:
            profiler.total_samples += pooled.shape[0]
        embeddings.append(pooled.cpu())
    if not embeddings:
        raise ValueError("Cannot encode an empty dataloader")
    return torch.cat(embeddings, dim=0)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def encode_assemblies(
    model,
    pooling_head,
    tokenizer,
    assemblies: Sequence,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    num_workers: int = 0,
    profiler: Optional[InferenceProfiler] = None,
    description: str = "encode",
    bidirectional: bool = False,
) -> torch.Tensor:
    texts = [asm_to_text(assembly) for assembly in assemblies]
    collator = L4DAssemblyCollator(tokenizer, max_length=max_length)
    loader = DataLoader(
        texts,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    return encode_loader(
        model,
        pooling_head,
        loader,
        device,
        description=description,
        profiler=profiler,
        bidirectional=bidirectional,
        pad_id=tokenizer.pad_token_id,
    )


def evaluate_retrieval(
    model,
    pooling_head,
    tokenizer,
    samples: Sequence[Dict],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    seed: int,
    num_workers: int = 0,
    max_pairs: Optional[int] = None,
    bidirectional: bool = False,
) -> Tuple[Dict, Dict]:
    pairs = build_eval_pairs(list(samples), seed=seed)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError("No evaluation pairs were built")

    # One warm-up batch is intentionally excluded from latency statistics.
    warm_assemblies = [pair["query"]["asm"] for pair in pairs[:batch_size]]
    encode_assemblies(
        model,
        pooling_head,
        tokenizer,
        warm_assemblies,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        num_workers=0,
        description="warmup",
        bidirectional=bidirectional,
    )

    profiler = InferenceProfiler(device)
    query_embeddings = encode_assemblies(
        model,
        pooling_head,
        tokenizer,
        [pair["query"]["asm"] for pair in pairs],
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        num_workers=num_workers,
        profiler=profiler,
        description="queries",
        bidirectional=bidirectional,
    )
    target_embeddings = encode_assemblies(
        model,
        pooling_head,
        tokenizer,
        [pair["target"]["asm"] for pair in pairs],
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        num_workers=num_workers,
        profiler=profiler,
        description="targets",
        bidirectional=bidirectional,
    )
    result = {
        "ids": [pair["id"] for pair in pairs],
        "query_opts": [pair["query"]["opt"] for pair in pairs],
        "target_opts": [pair["target"]["opt"] for pair in pairs],
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
        "stats": profiler.get_stats(),
    }
    return result, compute_report(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-ckpt", "--s3_ckpt", required=True)
    parser.add_argument("--data", "--eval_data", default=default_jsonl("test"))
    parser.add_argument("--head-checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--eval-tag", "--eval_tag", default=None)
    parser.add_argument("--batch-size", "--eval_batch_size", type=int, default=16)
    parser.add_argument("--max-length", "--max_length", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device-id", "--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--bidir-attention",
        "--bidir_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use bidirectional attention."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LLM4Decompile EBM evaluation requires a CUDA device")
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}")
    tokenizer, model, pooling_head, head_path = load_ebm_for_inference(
        args.s3_ckpt,
        device,
        head_checkpoint=args.head_checkpoint,
        bidirectional=args.bidir_attention,
    )
    samples = load_jsonl(args.data)
    result, report = evaluate_retrieval(
        model,
        pooling_head,
        tokenizer,
        samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        seed=args.seed,
        num_workers=args.num_workers,
        max_pairs=args.max_pairs,
        bidirectional=args.bidir_attention,
    )

    suffix = f"_{args.eval_tag}" if args.eval_tag else ""
    default_dir = Path(get_embeddings_dir("binarycorp_bench", "llm4decompile_ebm"))
    output_path = Path(
        args.output or (default_dir / f"eval_bench_embeddings{suffix}.pt")
    )
    report_path = Path(
        args.report or (default_dir / f"eval_bench_metrics{suffix}.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    experiment = {
        "model": MODEL_ID,
        "checkpoint": str(Path(args.s3_ckpt).resolve()),
        "backbone_finetuned": True,
        "adapter_training": "DoRA_then_merged",
        "mask": "bidirectional_4d" if args.bidir_attention else "native_causal",
        "mask_symmetrized": bool(args.bidir_attention),
        "pooling": "masked_attention",
        "pooling_positions": "func_header_and_assembly_tokens_only",
        "query_instruction": "llm4decompile_v1.5_decompile_prompt",
        "max_length": args.max_length,
        "pooling_head": str(head_path),
        "data": str(Path(args.data).resolve()),
    }
    result["experiment"] = experiment
    report.update(
        {
            "model": "llm4decompile_ebm",
            "data": str(Path(args.data).resolve()),
            "source": str(output_path.resolve()),
            "experiment": experiment,
            "inference": result["stats"],
        }
    )
    torch.save(result, output_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print_report_summary(report)
    print(
        f"\nllm4decompile_ebm: {result['stats']['avg_ms_per_sample']:.2f} "
        f"ms/function, {result['stats']['peak_memory_mb']:.1f} MB peak"
    )
    print(f"Embeddings: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
