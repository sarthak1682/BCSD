#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from joint_scratch.config import ScratchModelConfig, estimate_parameter_count  # noqa: E402
from joint_scratch.data import (  # noqa: E402
    JointPairDataset,
    JointScratchCollator,
    load_variant_pairs,
    split_group,
    split_pairs,
)
from joint_scratch.losses import (  # noqa: E402
    retrieval_recall_at_1,
    shifted_causal_cross_entropy,
    shifted_token_accuracy,
)
from joint_scratch.model import ScratchJointModel  # noqa: E402
from shared.data_utils import set_seed  # noqa: E402
from shared.losses import contrastive_loss_positive_aware  # noqa: E402


def default_clap_bench_train_path() -> Path:
    candidates = [
        BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_train_clap_strict.jsonl",
        BCSD_ROOT / "output_benchset_rebalanced_train_clap_strict.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


@dataclass(frozen=True)
class LossWeights:
    lm: float = 1.0
    mlm: float = 1.0
    retrieval: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    data: str
    val_data: Optional[str]
    no_validation: bool
    output_dir: str
    split_unit: str
    seed: int
    epochs: int
    pair_batch_size: int
    grad_accum: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    contrastive_temperature: float
    val_fraction: float
    val_batches: int
    num_workers: int
    device: str
    amp: bool
    gradient_checkpointing: bool
    training_objectives: str
    training_mode: str
    lm_task_probability: Optional[float]
    lm_target_fraction: float
    mlm_probability: float
    raw_chars_per_token: int
    max_train_pairs: Optional[int]
    max_val_pairs: Optional[int]
    model: Dict
    loss_weights: Dict
    init_checkpoint: Optional[str] = None
    initialization: str = "random"
    teacher: None = None
    coupled_every_step: bool = True
    pairing: str = "all_clap_strict_bench_variants_conditioned_on_target"
    freeze_backbone: bool = False
    train_task_embedding: bool = False


@dataclass
class BatchResult:
    loss: torch.Tensor
    metrics: Dict[str, float]
    correct_tokens: int = 0
    target_tokens: int = 0
    mlm_correct_tokens: int = 0
    mlm_target_tokens: int = 0
    query_embeddings: Optional[torch.Tensor] = None
    target_embeddings: Optional[torch.Tensor] = None
    query_ids: Optional[torch.Tensor] = None
    target_ids: Optional[torch.Tensor] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(default_clap_bench_train_path()),
    )
    parser.add_argument(
        "--val-data",
        default=None,
        help=(
            "Optional explicit validation/test JSONL. When set, --data is used "
            "entirely for training and no internal train/val split is carved out."
        ),
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Train only on --data and skip the validation phase entirely.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split-unit", choices=("binary", "function"), default="binary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--contrastive-temperature", type=float, default=0.05)
    parser.add_argument("--lambda-lm", type=float, default=1.0)
    parser.add_argument("--lambda-mlm", type=float, default=1.0)
    parser.add_argument("--lambda-retrieval", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--val-batches",
        type=int,
        default=50,
        help="Validation batches per epoch; zero evaluates the complete split.",
    )
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional ScratchJointModel checkpoint directory to initialize from.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Freeze all parameters except the retrieval head. Requires "
            "--init-checkpoint and --training-objectives retrieval."
        ),
    )
    parser.add_argument(
        "--train-task-embedding",
        action="store_true",
        help=(
            "When --freeze-backbone is set, also keep the task embeddings "
            "trainable instead of freezing them with the backbone."
        ),
    )
    parser.add_argument(
        "--training-objectives",
        choices=("lm_retrieval", "lm_mlm", "retrieval", "lm", "mlm", "all"),
        default="lm_retrieval",
        help=(
            "Which objective set to optimize. Use lm_mlm for causal+MLM "
            "pretraining and retrieval for contrastive fine-tuning."
        ),
    )
    parser.add_argument(
        "--training-mode",
        choices=("joint", "alternate"),
        default="joint",
        help=(
            "joint sums LM and retrieval losses every batch; alternate samples "
            "one task per batch, Hyformer-style."
        ),
    )
    parser.add_argument(
        "--lm-task-prob",
        type=float,
        default=0.5,
        help=(
            "LM task probability for two-task alternating objective sets "
            "(lm_retrieval or lm_mlm)."
        ),
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use bfloat16 autocast on CUDA.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--ff-dim", type=int, default=3_072)
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1_024)
    parser.add_argument(
        "--lm-target-fraction",
        type=float,
        default=0.5,
        help="Target-token reservation when an LM example exceeds max-length.",
    )
    parser.add_argument(
        "--mlm-probability",
        type=float,
        default=0.15,
        help="Token masking probability for bidirectional MLM/MNTP batches.",
    )
    parser.add_argument(
        "--raw-chars-per-token",
        type=int,
        default=8,
        help=(
            "Pre-tokenization assembly character budget multiplier. Raw text is "
            "clipped before tokenization, then token-truncated to max-length."
        ),
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def log(message: str, path: Path) -> None:
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def append_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def move_tensor_dict(values: Dict, device: torch.device) -> Dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in values.items()
    }


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must agree on batch/sequence shape")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must agree on batch/sequence shape")
    mask = labels.ne(-100)
    total = int(mask.sum().item())
    if not total:
        return 0, 0
    predictions = logits.argmax(dim=-1)
    correct = int(predictions.eq(labels).masked_select(mask).sum().item())
    return correct, total


def validate_objective_set(name: str) -> None:
    if name not in {"lm_retrieval", "lm_mlm", "retrieval", "lm", "mlm", "all"}:
        raise ValueError(f"unknown training objective set: {name}")


def task_for_step(args: argparse.Namespace, rng: random.Random) -> str:
    validate_objective_set(args.training_objectives)
    if args.training_objectives == "lm_retrieval":
        if args.training_mode == "joint":
            return "joint"
        return "lm" if rng.random() < args.lm_task_prob else "retrieval"
    if args.training_objectives == "lm_mlm":
        if args.training_mode == "joint":
            return "lm_mlm"
        return "lm" if rng.random() < args.lm_task_prob else "mlm"
    if args.training_objectives == "all":
        if args.training_mode == "joint":
            return "all"
        draw = rng.random()
        if draw < args.lm_task_prob:
            return "lm"
        return "mlm" if draw < (1.0 + args.lm_task_prob) / 2.0 else "retrieval"
    return args.training_objectives


def validation_task_for_objectives(name: str) -> str:
    validate_objective_set(name)
    return {
        "lm_retrieval": "joint",
        "lm_mlm": "lm_mlm",
        "retrieval": "retrieval",
        "lm": "lm",
        "mlm": "mlm",
        "all": "all",
    }[name]


def compute_batch(
    model: ScratchJointModel,
    cpu_batch: Dict,
    device: torch.device,
    *,
    weights: LossWeights,
    contrastive_temperature: float,
    amp: bool,
    task: str = "joint",
    collect_retrieval: bool = False,
) -> BatchResult:
    if task not in {"joint", "lm_mlm", "all", "lm", "mlm", "retrieval"}:
        raise ValueError(
            "task must be one of: joint, lm_mlm, all, lm, mlm, retrieval"
        )
    losses: Dict[str, torch.Tensor] = {}
    correct = 0
    target_tokens = 0
    mlm_correct = 0
    mlm_target_tokens = 0
    retrieval_output = None
    retrieval_batch = None

    with autocast_context(device, amp):
        if task in {"joint", "lm_mlm", "all", "lm"}:
            lm = move_tensor_dict(cpu_batch["lm"], device)
            lm_output = model.forward_lm(lm["input_ids"], lm["attention_mask"])
            losses["lm_ce"] = shifted_causal_cross_entropy(
                lm_output.logits, lm["labels"]
            )
            correct, target_tokens = shifted_token_accuracy(
                lm_output.logits, lm["labels"]
            )

        if task in {"lm_mlm", "all", "mlm"}:
            mlm = move_tensor_dict(cpu_batch["mlm"], device)
            mlm_output = model.forward_mlm(mlm["input_ids"], mlm["attention_mask"])
            losses["mlm_ce"] = masked_cross_entropy(mlm_output.logits, mlm["labels"])
            mlm_correct, mlm_target_tokens = masked_token_accuracy(
                mlm_output.logits, mlm["labels"]
            )

        if task in {"joint", "all", "retrieval"}:
            retrieval_batch = move_tensor_dict(cpu_batch["retrieval"], device)
            retrieval_output = model.forward_retrieval(
                retrieval_batch["input_ids"],
                retrieval_batch["attention_mask"],
                retrieval_batch["pool_mask"],
            )
            losses["retrieval_contrastive"] = contrastive_loss_positive_aware(
                retrieval_output.embeddings.float(),
                retrieval_batch["func_ids"],
                temperature=contrastive_temperature,
            )

    total = None
    if "lm_ce" in losses:
        total = weights.lm * losses["lm_ce"]
    if "mlm_ce" in losses:
        mlm_loss = weights.mlm * losses["mlm_ce"]
        total = mlm_loss if total is None else total + mlm_loss
    if "retrieval_contrastive" in losses:
        retrieval_loss = weights.retrieval * losses["retrieval_contrastive"]
        total = retrieval_loss if total is None else total + retrieval_loss
    if total is None:
        raise RuntimeError("no task loss was computed")

    metrics = {name: float(value.detach().item()) for name, value in losses.items()}
    metrics["loss"] = float(total.detach().item())
    result = BatchResult(
        loss=total,
        metrics=metrics,
        correct_tokens=correct,
        target_tokens=target_tokens,
        mlm_correct_tokens=mlm_correct,
        mlm_target_tokens=mlm_target_tokens,
    )
    if collect_retrieval and retrieval_output is not None and retrieval_batch is not None:
        embeddings = retrieval_output.embeddings.detach().float().cpu()
        ids = retrieval_batch["func_ids"].detach().cpu()
        result.query_embeddings = embeddings[0::2]
        result.target_embeddings = embeddings[1::2]
        result.query_ids = ids[0::2]
        result.target_ids = ids[1::2]
    return result


@torch.no_grad()
def validate(
    model: ScratchJointModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    weights: LossWeights,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    steps = 0
    correct = 0
    target_tokens = 0
    mlm_correct = 0
    mlm_target_tokens = 0
    queries = []
    targets = []
    query_ids = []
    target_ids = []
    validation_task = validation_task_for_objectives(
        getattr(args, "training_objectives", "lm_retrieval")
    )

    for cpu_batch in tqdm(loader, desc="validation", leave=False):
        result = compute_batch(
            model,
            cpu_batch,
            device,
            weights=weights,
            contrastive_temperature=args.contrastive_temperature,
            amp=args.amp,
            task=validation_task,
            collect_retrieval=True,
        )
        for key, value in result.metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        correct += result.correct_tokens
        target_tokens += result.target_tokens
        mlm_correct += result.mlm_correct_tokens
        mlm_target_tokens += result.mlm_target_tokens
        if result.query_embeddings is not None:
            queries.append(result.query_embeddings)
            targets.append(result.target_embeddings)
            query_ids.append(result.query_ids)
            target_ids.append(result.target_ids)
        steps += 1
        if args.val_batches and steps >= args.val_batches:
            break

    metrics = {key: value / max(steps, 1) for key, value in totals.items()}
    if target_tokens:
        metrics["token_accuracy"] = correct / target_tokens
        metrics["target_tokens"] = float(target_tokens)
        metrics["perplexity"] = math.exp(min(metrics["lm_ce"], 20.0))
    if mlm_target_tokens:
        metrics["mlm_token_accuracy"] = mlm_correct / mlm_target_tokens
        metrics["mlm_target_tokens"] = float(mlm_target_tokens)
    if queries:
        metrics["retrieval_recall_at_1"] = retrieval_recall_at_1(
            torch.cat(queries),
            torch.cat(targets),
            torch.cat(query_ids),
            torch.cat(target_ids),
            compute_device=getattr(args, "retrieval_recall_device", None),
        )
        metrics["retrieval_queries"] = float(sum(value.shape[0] for value in queries))
    model.train()
    return metrics


def cosine_schedule(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


def validate_args(args: argparse.Namespace) -> None:
    validate_objective_set(args.training_objectives)
    if args.epochs < 1 or args.grad_accum < 1 or args.pair_batch_size < 2:
        raise ValueError("epochs, grad-accum, and pair-batch-size must be positive")
    if args.learning_rate <= 0 or args.contrastive_temperature <= 0:
        raise ValueError("learning rate and temperature must be positive")
    if args.lambda_lm < 0 or args.lambda_mlm < 0 or args.lambda_retrieval < 0:
        raise ValueError("loss weights must be non-negative")
    required_weights = {
        "lm_retrieval": ("lambda_lm", "lambda_retrieval"),
        "lm_mlm": ("lambda_lm", "lambda_mlm"),
        "retrieval": ("lambda_retrieval",),
        "lm": ("lambda_lm",),
        "mlm": ("lambda_mlm",),
        "all": ("lambda_lm", "lambda_mlm", "lambda_retrieval"),
    }[args.training_objectives]
    for field in required_weights:
        if getattr(args, field) == 0:
            raise ValueError(f"{field.replace('_', '-')} must be nonzero")
    if args.no_validation and args.val_data:
        raise ValueError("--no-validation and --val-data are mutually exclusive")
    if not args.no_validation and args.val_data is None and not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be in (0,1) without --val-data")
    if not args.no_validation and args.val_data is not None and not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("val-fraction must be in [0,1)")
    if not 0.0 < args.lm_task_prob < 1.0:
        raise ValueError("lm-task-prob must be in (0,1)")
    if not 0.0 < args.mlm_probability <= 1.0:
        raise ValueError("mlm-probability must be in (0,1]")
    if args.raw_chars_per_token <= 0:
        raise ValueError("raw-chars-per-token must be positive")
    if args.log_every < 1:
        raise ValueError("log-every must be positive")
    if args.freeze_backbone and not args.init_checkpoint:
        raise ValueError("--freeze-backbone requires --init-checkpoint")
    if args.freeze_backbone and args.training_objectives != "retrieval":
        raise ValueError("--freeze-backbone requires --training-objectives retrieval")
    if args.train_task_embedding and not args.freeze_backbone:
        raise ValueError("--train-task-embedding requires --freeze-backbone")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    set_seed(args.seed)

    # Keep model/data/objective imports usable in lightweight test environments
    # that do not install Hugging Face Transformers.
    from joint_scratch.tokenizer import load_nova_tokenizer

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = REPO_ROOT / "model_checkpoints" / "joint_scratch" / run_id
    output_dir = Path(args.output_dir or default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    metrics_path = output_dir / "metrics.jsonl"
    log_path.write_text("", encoding="utf-8")
    metrics_path.write_text("", encoding="utf-8")

    base_tokenizer, nova_tokenizer = load_nova_tokenizer()
    model_config = ScratchModelConfig(
        vocab_size=len(base_tokenizer),
        pad_id=base_tokenizer.pad_token_id or 0,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_length=args.max_length,
        embedding_dim=args.embedding_dim,
    )
    model_config.validate()
    model = None
    initialization = "random"
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint).resolve()
        model = ScratchJointModel.from_pretrained(init_path, map_location="cpu")
        model_config = model.config
        initialization = f"checkpoint:{init_path}"
    weights = LossWeights(
        lm=args.lambda_lm,
        mlm=args.lambda_mlm,
        retrieval=args.lambda_retrieval,
    )
    run_config = RunConfig(
        data=str(Path(args.data).resolve()),
        val_data=str(Path(args.val_data).resolve()) if args.val_data else None,
        no_validation=bool(args.no_validation),
        output_dir=str(output_dir),
        split_unit=args.split_unit,
        seed=args.seed,
        epochs=args.epochs,
        pair_batch_size=args.pair_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        contrastive_temperature=args.contrastive_temperature,
        val_fraction=args.val_fraction,
        val_batches=args.val_batches,
        num_workers=args.num_workers,
        device=str(device),
        amp=bool(args.amp and device.type == "cuda"),
        gradient_checkpointing=args.gradient_checkpointing,
        training_objectives=args.training_objectives,
        training_mode=args.training_mode,
        lm_task_probability=(
            args.lm_task_prob
            if args.training_mode == "alternate"
            and args.training_objectives in {"lm_retrieval", "lm_mlm", "all"}
            else None
        ),
        lm_target_fraction=args.lm_target_fraction,
        mlm_probability=args.mlm_probability,
        raw_chars_per_token=args.raw_chars_per_token,
        max_train_pairs=args.max_train_pairs,
        max_val_pairs=args.max_val_pairs,
        model=model_config.to_dict(),
        loss_weights=asdict(weights),
        init_checkpoint=(
            str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None
        ),
        initialization=initialization,
        coupled_every_step=args.training_mode == "joint",
        freeze_backbone=args.freeze_backbone,
        train_task_embedding=args.train_task_embedding,
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(run_config), handle, indent=2)

    log(
        f"Loading CLAP-strict conditioned all-variant bench train pairs from {args.data}",
        log_path,
    )
    pairs = load_variant_pairs(args.data, seed=args.seed)
    val_source_pairs = None
    if args.no_validation:
        train_pairs = list(pairs)
        val_pairs = []
    elif args.val_data:
        log(
            f"Loading explicit CLAP-strict validation/test pairs from {args.val_data}",
            log_path,
        )
        val_source_pairs = load_variant_pairs(args.val_data, seed=args.seed)
        train_pairs = list(pairs)
        val_pairs = list(val_source_pairs)
    else:
        train_pairs, val_pairs = split_pairs(
            pairs,
            val_fraction=args.val_fraction,
            seed=args.seed,
            unit=args.split_unit,
        )
    if args.max_train_pairs is not None:
        train_pairs = train_pairs[: args.max_train_pairs]
    if args.max_val_pairs is not None and val_pairs:
        val_pairs = val_pairs[: args.max_val_pairs]
    if not train_pairs:
        raise RuntimeError("training requires at least one pair")
    if not args.no_validation and not val_pairs:
        raise RuntimeError("validation requires at least one pair")
    if args.no_validation:
        log(
            f"Training-only split; train eligible={len(pairs):,}; "
            f"train={len(train_pairs):,}; validation skipped",
            log_path,
        )
    elif args.val_data:
        train_functions = {pair.function_id for pair in train_pairs}
        val_functions = {pair.function_id for pair in val_pairs}
        overlap = train_functions.intersection(val_functions)
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise RuntimeError(
                "explicit train/validation function leakage detected; "
                f"examples: {examples}"
            )
        log(
            "Explicit train/test split; "
            f"train eligible={len(pairs):,}; val eligible={len(val_source_pairs or []):,}; "
            f"train/val={len(train_pairs):,}/{len(val_pairs):,}",
            log_path,
        )
    else:
        train_groups = {
            split_group(pair.function_id, args.split_unit) for pair in train_pairs
        }
        val_groups = {split_group(pair.function_id, args.split_unit) for pair in val_pairs}
        if train_groups.intersection(val_groups):
            raise RuntimeError("train/validation group leakage detected")
        log(
            f"Eligible pairs={len(pairs):,}; {args.split_unit}-disjoint "
            f"train/val={len(train_pairs):,}/{len(val_pairs):,}",
            log_path,
        )

    collator = JointScratchCollator(
        nova_tokenizer,
        max_length=model_config.max_length,
        lm_target_fraction=args.lm_target_fraction,
        mlm_probability=args.mlm_probability,
        raw_chars_per_token=args.raw_chars_per_token,
    )
    loader_args = dict(
        batch_size=args.pair_batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        JointPairDataset(train_pairs),
        shuffle=True,
        drop_last=True,
        **loader_args,
    )
    val_loader = None
    if not args.no_validation:
        val_loader = DataLoader(
            JointPairDataset(val_pairs), shuffle=False, drop_last=False, **loader_args
        )

    if model is None:
        model = ScratchJointModel(model_config)
    model = model.to(device)
    if args.freeze_backbone:
        model.freeze_backbone(train_task_embedding=args.train_task_embedding)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    estimated_parameters = estimate_parameter_count(model_config)
    if actual_parameters != estimated_parameters:
        raise RuntimeError(
            f"parameter estimator drifted: actual={actual_parameters:,}, "
            f"estimated={estimated_parameters:,}"
        )
    log(
        f"Initialization={initialization}; parameters={actual_parameters:,} "
        f"({actual_parameters / 1e6:.1f}M); trainable={trainable_parameters:,} "
        f"({trainable_parameters / 1e6:.2f}M); device={device}",
        log_path,
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = cosine_schedule(
        optimizer,
        warmup_steps=int(args.warmup_ratio * total_updates),
        total_steps=total_updates,
    )

    best_validation = float("inf")
    final_validation = None
    global_update = 0
    task_rng = random.Random(args.seed)
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(1, args.epochs + 1):
        running: Dict[str, float] = {}
        running_steps = 0
        correct = 0
        target_tokens = 0
        mlm_correct = 0
        mlm_target_tokens = 0
        for step, cpu_batch in enumerate(
            tqdm(train_loader, desc=f"joint epoch {epoch}/{args.epochs}"), start=1
        ):
            result = compute_batch(
                model,
                cpu_batch,
                device,
                weights=weights,
                contrastive_temperature=args.contrastive_temperature,
                amp=args.amp,
                task=task_for_step(args, task_rng),
            )
            (result.loss / args.grad_accum).backward()
            for key, value in result.metrics.items():
                running[key] = running.get(key, 0.0) + value
            running_steps += 1
            correct += result.correct_tokens
            target_tokens += result.target_tokens
            mlm_correct += result.mlm_correct_tokens
            mlm_target_tokens += result.mlm_target_tokens

            should_update = step % args.grad_accum == 0 or step == len(train_loader)
            if not should_update:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            if global_update % args.log_every == 0:
                record = {
                    "event": "train",
                    "epoch": epoch,
                    "update": global_update,
                    **{
                        key: value / max(running_steps, 1)
                        for key, value in running.items()
                    },
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                if target_tokens:
                    record["token_accuracy"] = correct / target_tokens
                if mlm_target_tokens:
                    record["mlm_token_accuracy"] = mlm_correct / mlm_target_tokens
                append_jsonl(metrics_path, record)
                metric_text = " ".join(
                    f"{key}={value:.4f}"
                    for key, value in record.items()
                    if key
                    in (
                        "loss",
                        "lm_ce",
                        "mlm_ce",
                        "retrieval_contrastive",
                        "token_accuracy",
                        "mlm_token_accuracy",
                    )
                )
                log(
                    f"update={global_update}/{total_updates} {metric_text}", log_path
                )
                running = {}
                running_steps = 0
                correct = 0
                target_tokens = 0
                mlm_correct = 0
                mlm_target_tokens = 0

        if val_loader is not None:
            validation = validate(model, val_loader, device, args, weights)
            final_validation = validation
            validation_record = {
                "event": "validation",
                "epoch": epoch,
                "update": global_update,
                **validation,
            }
            append_jsonl(metrics_path, validation_record)
            metric_text = " ".join(
                f"{key}={value:.4f}"
                for key, value in validation.items()
                if key
                in (
                    "loss",
                    "lm_ce",
                    "mlm_ce",
                    "perplexity",
                    "token_accuracy",
                    "mlm_token_accuracy",
                    "retrieval_contrastive",
                    "retrieval_recall_at_1",
                )
            )
            log(f"validation epoch={epoch} {metric_text}", log_path)
            if validation["loss"] < best_validation:
                best_validation = validation["loss"]
                model.save_pretrained(output_dir / "best")

        # Persist the latest epoch's weights on every epoch so an interrupted
        # run can always resume from the most recent checkpoint.
        model.save_pretrained(output_dir / "latest")

    model.save_pretrained(output_dir / "final")
    summary = {
        "training": "joint_scratch",
        "parameters": actual_parameters,
        "trainable_parameters": trainable_parameters,
        "eligible_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "best_validation_loss": None if args.no_validation else best_validation,
        "final_validation": final_validation,
        "updates": global_update,
        "checkpoints": ["final"] if args.no_validation else ["best", "latest", "final"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(f"Training complete: {output_dir}", log_path)


if __name__ == "__main__":
    main()
