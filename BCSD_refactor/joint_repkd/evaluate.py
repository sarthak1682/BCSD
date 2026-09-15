#!/usr/bin/env python3
"""Evaluate a joint_repkd checkpoint with a frozen-representation BCSD probe.

The repkd pretraining trains the shared backbone (causal NTP + bidirectional
MLM + hidden-state KD) but never trains the retrieval head.  This script
therefore probes the frozen backbone directly: it encodes each assembly through
the bidirectional (MLM) path and mean-pools the final-layer hidden states over
non-padding tokens.  This is the "pre-CL" representation, before any
contrastive retrieval fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from binarycorp_bench.eval_bench import compute_report, print_report_summary  # noqa: E402
from joint_scratch.model import ScratchJointModel  # noqa: E402
from shared.data_utils import build_eval_pairs, load_jsonl  # noqa: E402


DEFAULT_TEST_PATH = (
    "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/"
    "output_benchset_rebalanced_test_nova.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("best", "final", "direct"), default="best")
    parser.add_argument("--data", default=DEFAULT_TEST_PATH)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-eval-pairs", type=int, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pooling",
        choices=("mean_pool", "retrieval_head"),
        default="retrieval_head",
        help=(
            "mean_pool probes the frozen bidirectional MLM hidden states (the "
            "pre-CL representation); retrieval_head uses the trained attention "
            "pooling + projection head (the after-CL model)."
        ),
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


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def encode_samples(
    model, tokenizer_wrapper, samples, device, *, batch_size, num_workers, amp, desc, pooling,
):
    base_tokenizer = tokenizer_wrapper.tokenizer
    pad_id = base_tokenizer.pad_token_id
    max_length = model.config.max_length

    class SampleList:
        def __getitem__(self, index):
            return samples[index]

        def __len__(self):
            return len(samples)

    def collate(batch):
        rows = []
        for sample in batch:
            text = sample["asm"]
            if isinstance(text, list):
                text = "\n".join(text)
            encoded = tokenizer_wrapper.encode("", text, "1" * len(text))
            ids = encoded["input_ids"][:max_length]
            rows.append(ids)
        length = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
        attention = torch.zeros((len(rows), length), dtype=torch.bool)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.from_numpy(row)
            attention[index, : len(row)] = True
        return {"input_ids": input_ids, "attention_mask": attention}

    loader = DataLoader(
        SampleList(),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    embeddings = []
    for batch in tqdm(loader, desc=desc):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention = batch["attention_mask"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            if pooling == "retrieval_head":
                output = model.forward_retrieval(input_ids, attention, attention)
                embeddings.append(output.embeddings.detach().float().cpu())
                continue
            output = model.forward_mlm(input_ids, attention)
        hidden = output.hidden_states.float()
        mask = attention.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = resolve_checkpoint(Path(args.checkpoint).resolve(), args.split)

    from joint_scratch.tokenizer import load_nova_tokenizer  # noqa: E402

    _, nova_tokenizer = load_nova_tokenizer()
    model = ScratchJointModel.from_pretrained(checkpoint, map_location="cpu").to(device)
    model.eval()

    samples = load_jsonl(args.data)
    pairs = build_eval_pairs(samples, seed=42)
    if args.max_eval_pairs is not None:
        pairs = pairs[: args.max_eval_pairs]
    if not pairs:
        raise RuntimeError("evaluation requires at least one pair")

    queries = [pair["query"] for pair in pairs]
    targets = [pair["target"] for pair in pairs]
    encode_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp and device.type == "cuda",
        pooling=args.pooling,
    )
    query_embeddings = encode_samples(
        model, nova_tokenizer, queries, device, desc="encoding queries", **encode_kwargs
    )
    target_embeddings = encode_samples(
        model, nova_tokenizer, targets, device, desc="encoding targets", **encode_kwargs
    )

    result = {
        "ids": [pair["id"] for pair in pairs],
        "query_opts": [pair["query"]["opt"] for pair in pairs],
        "target_opts": [pair["target"]["opt"] for pair in pairs],
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
    }
    report = compute_report(result)
    report["checkpoint"] = str(checkpoint)
    report["data"] = str(Path(args.data).resolve())
    report["pooling"] = (
        "retrieval_head"
        if args.pooling == "retrieval_head"
        else "mean_pool_bidirectional_mlm_hidden"
    )
    report["parameters"] = sum(p.numel() for p in model.parameters())
    print_report_summary(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, out.with_suffix(".embeddings.pt"))
        with out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Saved report: {out}")
        print(f"Saved embeddings: {out.with_suffix('.embeddings.pt')}")


if __name__ == "__main__":
    main()
