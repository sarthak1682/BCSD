#!/usr/bin/env python3
"""Evaluate a compact joint student on BCSD retrieval and causal translation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from binarycorp_bench.eval_bench import compute_report, print_report_summary  # noqa: E402
from joint_distill.data import (  # noqa: E402
    JointDistillCollator,
    JointPair,
    JointPairDataset,
)
from joint_distill.losses import (  # noqa: E402
    shifted_causal_cross_entropy,
    shifted_token_accuracy,
)
from joint_distill.model import TaskConditionedJointStudent  # noqa: E402
from shared.data_utils import asm_to_text, build_eval_pairs, load_jsonl, set_seed  # noqa: E402
from joint_distill.tokenizer import load_ebm_tokenizer  # noqa: E402
from shared.profiling import InferenceProfiler  # noqa: E402


class PredOnlyCollator:
    def __init__(self, collator: JointDistillCollator) -> None:
        self.collator = collator

    def __call__(self, batch):
        return self.collator.collate_pred(batch)


class LMOnlyCollator:
    def __init__(self, collator: JointDistillCollator) -> None:
        self.collator = collator

    def __call__(self, batch):
        return self.collator.collate_lm(batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data",
        default=str(BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_test_nova.jsonl"),
    )
    parser.add_argument("--task", choices=("retrieval", "translation", "both"), default="both")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pair-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--condition-on-target",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to the training run's setting when run_config.json is available.",
    )
    parser.add_argument(
        "--generation-pairs",
        type=int,
        default=100,
        help="Number of pairs for exact greedy translation; 0 disables generation.",
    )
    parser.add_argument("--generation-max-new-tokens", type=int, default=256)
    return parser.parse_args()


def resolve_training_setting(checkpoint: Path, name: str, default):
    candidates = (checkpoint / "run_config.json", checkpoint.parent / "run_config.json")
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle).get(name, default)
    return default


def make_pairs(samples: List[Dict], seed: int) -> List[JointPair]:
    rows = build_eval_pairs(samples, seed=seed)
    return [
        JointPair(
            function_id=row["id"],
            numeric_id=index,
            source_asm=asm_to_text(row["query"]["asm"]),
            target_asm=asm_to_text(row["target"]["asm"]),
            source_opt=row["query"]["opt"],
            target_opt=row["target"]["opt"],
        )
        for index, row in enumerate(rows)
    ]


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_retrieval(
    model: TaskConditionedJointStudent,
    loader: DataLoader,
    pairs: List[JointPair],
    device: torch.device,
) -> tuple[Dict, Dict]:
    model.eval()
    profiler = InferenceProfiler(device)
    query_embeddings = []
    target_embeddings = []

    first = move_batch(next(iter(loader)), device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = model.forward_pred(
            first["input_ids"], first["attention_mask"], first["pool_mask"]
        )
    torch.cuda.synchronize(device)

    for cpu_batch in tqdm(loader, desc="retrieval"):
        batch = move_batch(cpu_batch, device)
        with profiler, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings = model.forward_pred(
                batch["input_ids"],
                batch["attention_mask"],
                batch["pool_mask"],
                normalize=True,
            ).embeddings
        profiler.total_samples += embeddings.shape[0]
        query_embeddings.append(embeddings[0::2].cpu())
        target_embeddings.append(embeddings[1::2].cpu())

    result = {
        "ids": [pair.function_id for pair in pairs],
        "query_opts": [pair.source_opt for pair in pairs],
        "target_opts": [pair.target_opt for pair in pairs],
        "query_embeddings": torch.cat(query_embeddings, dim=0),
        "target_embeddings": torch.cat(target_embeddings, dim=0),
        "stats": profiler.get_stats(),
    }
    return result, compute_report(result)


@torch.no_grad()
def evaluate_translation_teacher_forced(
    model: TaskConditionedJointStudent,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    steps = 0
    correct = 0
    tokens = 0
    for cpu_batch in tqdm(loader, desc="translation teacher-forced"):
        batch = move_batch(cpu_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward_lm(
                batch["input_ids"], batch["attention_mask"]
            ).logits
        loss = shifted_causal_cross_entropy(logits, batch["labels"])
        accuracy = shifted_token_accuracy(logits, batch["labels"])
        total_loss += loss.item()
        correct += accuracy[0]
        tokens += accuracy[1]
        steps += 1
    cross_entropy = total_loss / max(steps, 1)
    return {
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(min(cross_entropy, 20.0)),
        "token_accuracy": correct / max(tokens, 1),
        "tokens": tokens,
        "batches": steps,
    }


@torch.no_grad()
def evaluate_greedy_exact(
    model: TaskConditionedJointStudent,
    pairs: List[JointPair],
    collator: JointDistillCollator,
    device: torch.device,
    *,
    limit: int,
    max_new_tokens: int,
) -> Dict[str, float]:
    if limit <= 0:
        return {"eligible": 0, "exact": 0, "exact_match": 0.0}
    exact = 0
    eligible = 0
    for pair in tqdm(pairs[:limit], desc="translation greedy"):
        input_text, input_types = collator._lm_text(pair)
        prompt = collator.tokenizer.encode(input_text, "", input_types)
        target = collator.tokenizer.encode(
            "", pair.target_asm, "1" * len(pair.target_asm)
        )
        prompt_ids = list(prompt["input_ids"])
        target_ids = list(target["input_ids"])
        if (
            not target_ids
            or len(target_ids) > max_new_tokens
            or len(prompt_ids) + len(target_ids) > model.config.max_length
        ):
            continue
        source = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(source, max_new_tokens=len(target_ids))
        prediction = generated[0, len(prompt_ids) :].tolist()
        exact += int(prediction == target_ids)
        eligible += 1
    return {
        "eligible": eligible,
        "exact": exact,
        "exact_match": exact / max(eligible, 1),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA device")
    set_seed(args.seed)

    checkpoint = Path(args.checkpoint).resolve()
    run_label = f"{checkpoint.parent.name}_{checkpoint.name}"
    output_dir = Path(
        args.output_dir
        or (
            REPO_ROOT
            / "embeddings"
            / "binarycorp_bench"
            / "joint_distill"
            / run_label
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.device_id}")
    model = TaskConditionedJointStudent.from_pretrained(checkpoint).to(device)
    model.eval()

    _, nova_tokenizer = load_ebm_tokenizer()
    condition_on_target = args.condition_on_target
    if condition_on_target is None:
        condition_on_target = bool(
            resolve_training_setting(checkpoint, "condition_on_target", False)
        )
    collator = JointDistillCollator(
        nova_tokenizer,
        max_length=model.config.max_length,
        condition_on_target=condition_on_target,
    )

    samples = load_jsonl(args.data)
    pairs = make_pairs(samples, args.seed)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise RuntimeError("no evaluation pairs were built")
    dataset = JointPairDataset(pairs)
    common_loader = dict(
        dataset=dataset,
        batch_size=args.pair_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    if args.task in ("retrieval", "both"):
        retrieval_loader = DataLoader(
            collate_fn=PredOnlyCollator(collator), **common_loader
        )
        result, report = evaluate_retrieval(model, retrieval_loader, pairs, device)
        result["experiment"] = {
            "checkpoint": str(checkpoint),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "attention": "shared_causal_lm_or_bidirectional_pred",
            "condition_on_target": condition_on_target,
        }
        report["model"] = "joint_distill_student"
        report["data"] = str(Path(args.data).resolve())
        report["experiment"] = result["experiment"]
        report["inference"] = result["stats"]
        embeddings_path = output_dir / "retrieval_embeddings.pt"
        report_path = output_dir / "retrieval_report.json"
        torch.save(result, embeddings_path)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print_report_summary(report)
        print(f"Retrieval report: {report_path}")

    if args.task in ("translation", "both"):
        lm_loader = DataLoader(collate_fn=LMOnlyCollator(collator), **common_loader)
        translation = evaluate_translation_teacher_forced(model, lm_loader, device)
        translation["greedy"] = evaluate_greedy_exact(
            model,
            pairs,
            collator,
            device,
            limit=args.generation_pairs,
            max_new_tokens=args.generation_max_new_tokens,
        )
        translation["checkpoint"] = str(checkpoint)
        translation["data"] = str(Path(args.data).resolve())
        translation["condition_on_target"] = condition_on_target
        translation_path = output_dir / "translation_report.json"
        with translation_path.open("w", encoding="utf-8") as handle:
            json.dump(translation, handle, indent=2)
        print(json.dumps(translation, indent=2))
        print(f"Translation report: {translation_path}")


if __name__ == "__main__":
    main()
