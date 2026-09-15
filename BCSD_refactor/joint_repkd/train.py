#!/usr/bin/env python3
"""Two-stage representation-distillation pretraining for the joint student.

Implements the supervisor's "Joint-Scratch Distillation Objectives":

* Stage 1 (causal): both teacher and student consume the clean assembly under a
  causal attention mask; the student is trained with next-token prediction
  (NTP) to generate the target compiled variant, while its final-layer hidden
  states are distilled toward the frozen teacher's.

* Stage 2 (bidirectional): the student consumes token-masked assembly under
  bidirectional attention and is trained with masked-language modeling (MLM);
  the teacher consumes the clean assembly bidirectionally and its hidden states
  are distilled at the masked positions.

Each mode weights its distillation term by ``lambda(t) * s_m(t)``, where
``s_m(t)`` is a gradient-norm EMA and ``lambda(t)`` is a teacher-first anneal.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from joint_repkd.balance import GradientBalance, teacher_lambda  # noqa: E402
from joint_repkd.data import RepKDCollator  # noqa: E402
from joint_repkd.losses import hidden_state_distillation_loss  # noqa: E402
from joint_distill.data import (  # noqa: E402
    JointPairDataset,
    build_joint_pairs,
    split_function_ids,
)
from joint_distill.teacher import FullEBMTeacher  # noqa: E402
from joint_scratch.config import ScratchModelConfig, estimate_parameter_count  # noqa: E402
from joint_scratch.losses import shifted_causal_cross_entropy, shifted_token_accuracy  # noqa: E402
from joint_scratch.model import ScratchJointModel  # noqa: E402
from joint_scratch.tokenizer import load_nova_tokenizer  # noqa: E402
from shared.data_utils import load_jsonl, set_seed  # noqa: E402


def default_train_path() -> str:
    candidates = [
        "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_train_nova.jsonl",
        str(BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_train_nova.jsonl"),
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return candidates[0]


def default_teacher_path() -> str:
    candidates = [
        "/home/ra72yeq/model_checkpoints/nova_ebm_bench/s3_final",
        str(BCSD_ROOT / "model_checkpoints" / "nova_ebm_bench" / "s3_final"),
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return candidates[0]


@dataclass
class RunConfig:
    data: str
    val_data: Optional[str]
    teacher_dir: str
    output_dir: str
    seed: int
    epochs: int
    pair_batch_size: int
    grad_accum: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    val_fraction: float
    val_batches: int
    num_workers: int
    device: str
    amp: bool
    gradient_checkpointing: bool
    mlm_probability: float
    lambda_high: float
    lambda_low: float
    anneal_fraction: float
    beta: float
    kd_epsilon: float
    max_train_pairs: Optional[int]
    max_val_pairs: Optional[int]
    model: Dict
    student_parameters: int
    teacher_hidden_dim: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=default_train_path())
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--teacher-dir", default=default_teacher_path())
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:1", help="e.g. cuda:1, cuda:0, cpu")
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--ff-dim", type=int, default=3_072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1_024)
    parser.add_argument("--mlm-probability", type=float, default=0.15)

    parser.add_argument("--lambda-high", type=float, default=2.0)
    parser.add_argument("--lambda-low", type=float, default=0.1)
    parser.add_argument("--anneal-fraction", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--kd-epsilon", type=float, default=1e-6)

    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use bfloat16 autocast on CUDA.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


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


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


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


def cosine_schedule(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


@dataclass
class StageResult:
    gt_loss: torch.Tensor
    kd_loss: torch.Tensor
    metrics: Dict[str, float]
    correct: int = 0
    target_tokens: int = 0


def compute_stage1(
    student: ScratchJointModel,
    teacher: FullEBMTeacher,
    projection: nn.Module,
    cpu_batch: Dict,
    device: torch.device,
    *,
    amp: bool,
) -> StageResult:
    batch = move_tensor_dict(cpu_batch["stage1"], device)
    with autocast_context(device, amp):
        output = student.forward_lm(batch["input_ids"], batch["attention_mask"])
    with torch.no_grad():
        teacher_hidden = teacher.hidden_states(
            batch["input_ids"], batch["attention_mask"], batch["nova_attention_mask"]
        )
    gt_loss = shifted_causal_cross_entropy(output.logits, batch["labels"])
    kd_loss = hidden_state_distillation_loss(
        output.hidden_states, teacher_hidden, batch["assembly_mask"], projection
    )
    correct, target_tokens = shifted_token_accuracy(output.logits, batch["labels"])
    metrics = {
        "lm_ce": float(gt_loss.detach().item()),
        "kd_c": float(kd_loss.detach().item()),
    }
    return StageResult(gt_loss, kd_loss, metrics, correct, target_tokens)


def compute_stage2(
    student: ScratchJointModel,
    teacher: FullEBMTeacher,
    projection: nn.Module,
    cpu_batch: Dict,
    device: torch.device,
    *,
    amp: bool,
) -> StageResult:
    batch = move_tensor_dict(cpu_batch["stage2"], device)
    with autocast_context(device, amp):
        output = student.forward_mlm(batch["input_ids"], batch["attention_mask"])
    with torch.no_grad():
        teacher_hidden = teacher.hidden_states(
            batch["teacher_input_ids"],
            batch["attention_mask"],
            batch["teacher_nova_attention_mask"],
        )
    gt_loss = masked_cross_entropy(output.logits, batch["labels"])
    kd_loss = hidden_state_distillation_loss(
        output.hidden_states, teacher_hidden, batch["kd_mask"], projection
    )
    correct, target_tokens = masked_token_accuracy(output.logits, batch["labels"])
    metrics = {
        "mlm_ce": float(gt_loss.detach().item()),
        "kd_b": float(kd_loss.detach().item()),
    }
    return StageResult(gt_loss, kd_loss, metrics, correct, target_tokens)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_seed(args.seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = REPO_ROOT / "model_checkpoints" / "joint_repkd" / run_id
    output_dir = Path(args.output_dir or default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "train.log"
    metrics_path = output_dir / "metrics.jsonl"

    base_tokenizer, nova_tokenizer = load_nova_tokenizer()
    student_config = ScratchModelConfig(
        vocab_size=len(base_tokenizer),
        pad_id=base_tokenizer.pad_token_id or 0,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_length=args.max_length,
        embedding_dim=args.d_model,
    )
    student_config.validate()

    log(f"Loading frozen EBM-full teacher from {args.teacher_dir}", log_path)
    device_id = device.index if device.type == "cuda" and device.index is not None else 0
    teacher = FullEBMTeacher.load(args.teacher_dir, device_id=device_id)
    teacher_hidden_dim = teacher.embedding_weight.shape[1]
    if teacher.embedding_weight.shape[0] + 1 != student_config.vocab_size:
        raise ValueError(
            "student/teacher vocabularies differ: expected student vocab "
            f"{teacher.embedding_weight.shape[0] + 1} (Nova + [MASK]) but got "
            f"{student_config.vocab_size}"
        )

    log(f"Loading paired Nova data: {args.data}", log_path)
    samples = load_jsonl(args.data)
    train_ids, val_ids = split_function_ids(
        samples, val_fraction=args.val_fraction, seed=args.seed
    )
    train_pairs = build_joint_pairs(samples, train_ids, seed=args.seed)
    val_pairs = build_joint_pairs(samples, val_ids, seed=args.seed + 1)
    if args.max_train_pairs is not None:
        train_pairs = train_pairs[: args.max_train_pairs]
    if args.max_val_pairs is not None:
        val_pairs = val_pairs[: args.max_val_pairs]
    if not train_pairs or not val_pairs:
        raise RuntimeError("training and validation each require at least one pair")
    log(
        f"Function-disjoint train/val: {len(train_ids):,}/{len(val_ids):,}; "
        f"pairs: {len(train_pairs):,}/{len(val_pairs):,}",
        log_path,
    )

    collator = RepKDCollator(
        nova_tokenizer, max_length=args.max_length, mlm_probability=args.mlm_probability
    )
    loader_args = dict(
        batch_size=args.pair_batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(JointPairDataset(train_pairs), shuffle=True, **loader_args)
    val_loader = DataLoader(JointPairDataset(val_pairs), shuffle=False, **loader_args)

    student = ScratchJointModel(student_config).to(device)
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
    actual_parameters = sum(p.numel() for p in student.parameters())
    if actual_parameters != estimate_parameter_count(student_config):
        raise RuntimeError("parameter estimator drifted from the built model")

    proj_c = nn.Linear(args.d_model, teacher_hidden_dim).to(device)
    proj_b = nn.Linear(args.d_model, teacher_hidden_dim).to(device)

    run_config = RunConfig(
        data=str(Path(args.data).resolve()),
        val_data=str(Path(args.val_data).resolve()) if args.val_data else None,
        teacher_dir=str(Path(args.teacher_dir).resolve()),
        output_dir=str(output_dir),
        seed=args.seed,
        epochs=args.epochs,
        pair_batch_size=args.pair_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        val_fraction=args.val_fraction,
        val_batches=args.val_batches,
        num_workers=args.num_workers,
        device=str(device),
        amp=bool(args.amp and device.type == "cuda"),
        gradient_checkpointing=args.gradient_checkpointing,
        mlm_probability=args.mlm_probability,
        lambda_high=args.lambda_high,
        lambda_low=args.lambda_low,
        anneal_fraction=args.anneal_fraction,
        beta=args.beta,
        kd_epsilon=args.kd_epsilon,
        max_train_pairs=args.max_train_pairs,
        max_val_pairs=args.max_val_pairs,
        model=student_config.to_dict(),
        student_parameters=actual_parameters,
        teacher_hidden_dim=teacher_hidden_dim,
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(run_config), handle, indent=2)

    log(
        f"Student parameters={actual_parameters:,} ({actual_parameters / 1e6:.1f}M); "
        f"teacher hidden dim={teacher_hidden_dim}; device={device}",
        log_path,
    )

    optimizer = AdamW(
        list(student.parameters()) + list(proj_c.parameters()) + list(proj_b.parameters()),
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

    balance = GradientBalance(beta=args.beta, epsilon=args.kd_epsilon)
    backbone_params = list(student.parameters())
    best_validation = float("inf")
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    student.train()
    proj_c.train()
    proj_b.train()

    for epoch in range(1, args.epochs + 1):
        running: Dict[str, float] = {}
        running_steps = 0
        correct = 0
        target_tokens = 0
        stage1_turns = 0
        stage2_turns = 0
        for step, cpu_batch in enumerate(
            tqdm(train_loader, desc=f"repkd epoch {epoch}/{args.epochs}"), start=1
        ):
            if step % 2 == 1:
                result = compute_stage1(
                    student, teacher, proj_c, cpu_batch, device, amp=args.amp
                )
                mode = "C"
                stage1_turns += 1
            else:
                result = compute_stage2(
                    student, teacher, proj_b, cpu_batch, device, amp=args.amp
                )
                mode = "B"
                stage2_turns += 1

            s_m = balance.update(mode, result.gt_loss, result.kd_loss, backbone_params)
            lam = teacher_lambda(
                global_update,
                total_updates,
                lambda_high=args.lambda_high,
                lambda_low=args.lambda_low,
                anneal_fraction=args.anneal_fraction,
            )
            total = result.gt_loss + lam * s_m * result.kd_loss
            (total / args.grad_accum).backward()

            for key, value in result.metrics.items():
                running[key] = running.get(key, 0.0) + value
            running_steps += 1
            correct += result.correct
            target_tokens += result.target_tokens

            should_update = step % args.grad_accum == 0 or step == len(train_loader)
            if not should_update:
                continue
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(proj_c.parameters()) + list(proj_b.parameters()),
                1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1

            if global_update % args.log_every == 0:
                record = {
                    "event": "train",
                    "epoch": epoch,
                    "update": global_update,
                    **{key: value / max(running_steps, 1) for key, value in running.items()},
                    "learning_rate": scheduler.get_last_lr()[0],
                    "lambda": lam,
                    "s_C": balance.values["C"],
                    "s_B": balance.values["B"],
                }
                if target_tokens:
                    record["token_accuracy"] = correct / target_tokens
                append_jsonl(metrics_path, record)
                metric_text = " ".join(
                    f"{key}={value:.4f}"
                    for key, value in record.items()
                    if key in ("lm_ce", "kd_c", "mlm_ce", "kd_b", "token_accuracy")
                )
                log(
                    f"update={global_update}/{total_updates} lam={lam:.3f} "
                    f"s_C={balance.values['C']:.3f} s_B={balance.values['B']:.3f} "
                    f"{metric_text}",
                    log_path,
                )
                running = {}
                running_steps = 0
                correct = 0
                target_tokens = 0

        # Validation: report raw GT and KD losses, no balancing update.
        student.eval()
        proj_c.eval()
        proj_b.eval()
        val_totals: Dict[str, float] = {}
        val_steps = 0
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="validation", leave=False):
                if val_steps % 2 == 0:
                    result = compute_stage1(
                        student, teacher, proj_c, val_batch, device, amp=args.amp
                    )
                else:
                    result = compute_stage2(
                        student, teacher, proj_b, val_batch, device, amp=args.amp
                    )
                for key, value in result.metrics.items():
                    val_totals[key] = val_totals.get(key, 0.0) + value
                val_totals["gt"] = val_totals.get("gt", 0.0) + result.gt_loss.detach().item()
                val_totals["kd"] = val_totals.get("kd", 0.0) + result.kd_loss.detach().item()
                val_steps += 1
                if args.val_batches and val_steps >= args.val_batches:
                    break
        student.train()
        proj_c.train()
        proj_b.train()

        validation = {key: value / max(val_steps, 1) for key, value in val_totals.items()}
        validation["loss"] = validation.get("gt", 0.0) + validation.get("kd", 0.0)
        validation_record = {"event": "validation", "epoch": epoch, "update": global_update, **validation}
        append_jsonl(metrics_path, validation_record)
        metric_text = " ".join(
            f"{key}={value:.4f}"
            for key, value in validation.items()
            if key in ("loss", "lm_ce", "kd_c", "mlm_ce", "kd_b")
        )
        log(f"validation epoch={epoch} {metric_text}", log_path)
        if validation["loss"] < best_validation:
            best_validation = validation["loss"]
            student.save_pretrained(output_dir / "best")
            torch.save(
                {"proj_c": proj_c.state_dict(), "proj_b": proj_b.state_dict()},
                output_dir / "best" / "kd_projections.pt",
            )

    student.save_pretrained(output_dir / "final")
    torch.save(
        {"proj_c": proj_c.state_dict(), "proj_b": proj_b.state_dict()},
        output_dir / "final" / "kd_projections.pt",
    )
    summary = {
        "parameters": actual_parameters,
        "best_validation_loss": best_validation,
        "updates": global_update,
        "checkpoints": ["best", "final"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(f"Training complete: {output_dir}", log_path)


if __name__ == "__main__":
    main()
