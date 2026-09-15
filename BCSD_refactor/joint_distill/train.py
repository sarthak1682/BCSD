#!/usr/bin/env python3
"""Coupled distillation of EBM-full into a CLAP-sized joint student.

Every training batch supplies two views of the same function pairs:

* LM: causal source-to-target assembly translation;
* PRED: bidirectional query/positive retrieval.

The frozen merged EBM-full checkpoint supplies top-k causal logits and pooled
retrieval embeddings.  Ground-truth LM cross entropy and supervised
contrastive loss keep the student from merely imitating teacher errors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from joint_distill.config import StudentConfig, estimate_parameter_count  # noqa: E402
from joint_distill.data import (  # noqa: E402
    JointDistillCollator,
    JointPairDataset,
    build_joint_pairs,
    split_function_ids,
)
from joint_distill.losses import (  # noqa: E402
    embedding_distillation_loss,
    shifted_causal_cross_entropy,
    shifted_token_accuracy,
    topk_teacher_kl,
)
from joint_distill.model import TaskConditionedJointStudent  # noqa: E402
from joint_distill.teacher import FullEBMTeacher  # noqa: E402
from shared.data_utils import load_jsonl, set_seed  # noqa: E402
from shared.losses import contrastive_loss_positive_aware  # noqa: E402
from joint_distill.tokenizer import load_ebm_tokenizer  # noqa: E402


@dataclass
class LossWeights:
    lm_ce: float
    lm_kd: float
    embedding_kd: float
    contrastive: float


@dataclass
class RunConfig:
    data: str
    teacher_dir: str
    output_dir: str
    seed: int
    epochs: int
    pair_batch_size: int
    grad_accum: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    temperature: float
    kd_temperature: float
    teacher_top_k: int
    val_fraction: float
    val_batches: int
    num_workers: int
    device_id: int
    condition_on_target: bool
    max_train_pairs: Optional[int]
    max_val_pairs: Optional[int]
    student: Dict
    loss_weights: Dict
    teacher: str = "merged_nova_ebm_full_s3"
    coupled_every_step: bool = True
    function_disjoint_validation: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_train_nova.jsonl"),
    )
    parser.add_argument(
        "--teacher-dir",
        default=str(BCSD_ROOT / "model_checkpoints" / "nova_ebm_bench" / "s3_final"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=8,
        help="Functions per coupled batch; PRED sees twice this many sequences.",
    )
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-top-k", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--val-batches",
        type=int,
        default=50,
        help="Validation batches per epoch; 0 evaluates the entire split.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument(
        "--condition-on-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Prefix LM sources with the requested target compilation setting. "
            "Off by default to reproduce EBM-full's causal training format."
        ),
    )

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--student-layers", type=int, default=10)
    parser.add_argument("--student-heads", type=int, default=12)
    parser.add_argument("--student-ff-dim", type=int, default=3_072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1_024)

    parser.add_argument("--lambda-lm-ce", type=float, default=1.0)
    parser.add_argument("--lambda-lm-kd", type=float, default=0.5)
    parser.add_argument("--lambda-embedding-kd", type=float, default=1.0)
    parser.add_argument("--lambda-contrastive", type=float, default=1.0)
    parser.add_argument(
        "--skip-teacher-embedding-init",
        action="store_true",
        help="Use random token embeddings instead of projected Nova embeddings.",
    )
    return parser.parse_args()


def log(message: str, log_path: Path) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def append_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def move_tensor_dict(values: Dict, device: torch.device) -> Dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in values.items()
    }


def compute_coupled_losses(
    student: TaskConditionedJointStudent,
    teacher: FullEBMTeacher,
    cpu_batch: Dict,
    device: torch.device,
    weights: LossWeights,
    *,
    teacher_top_k: int,
    kd_temperature: float,
    contrastive_temperature: float,
) -> tuple[torch.Tensor, Dict[str, float], tuple[int, int]]:
    lm = move_tensor_dict(cpu_batch["lm"], device)
    pred = move_tensor_dict(cpu_batch["pred"], device)

    with torch.no_grad():
        lm_targets = teacher.lm_targets(lm, top_k=teacher_top_k)
        teacher_embeddings = teacher.pred_embeddings(pred, pred["label_positions"])

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        lm_output = student.forward_lm(lm["input_ids"], lm["attention_mask"])
        pred_output = student.forward_pred(
            pred["input_ids"], pred["attention_mask"], pred["pool_mask"]
        )

    loss_lm_ce = shifted_causal_cross_entropy(lm_output.logits, lm["labels"])
    loss_lm_kd = topk_teacher_kl(
        lm_output.logits,
        lm["labels"],
        lm_targets.indices,
        lm_targets.logits,
        temperature=kd_temperature,
    )
    loss_embedding = embedding_distillation_loss(
        pred_output.embeddings, teacher_embeddings
    )
    loss_contrastive = contrastive_loss_positive_aware(
        pred_output.embeddings.float(),
        pred["func_ids"],
        temperature=contrastive_temperature,
    )
    total = (
        weights.lm_ce * loss_lm_ce
        + weights.lm_kd * loss_lm_kd
        + weights.embedding_kd * loss_embedding
        + weights.contrastive * loss_contrastive
    )
    correct, token_count = shifted_token_accuracy(lm_output.logits, lm["labels"])
    metrics = {
        "loss": float(total.detach().item()),
        "lm_ce": float(loss_lm_ce.detach().item()),
        "lm_kd": float(loss_lm_kd.detach().item()),
        "embedding_kd": float(loss_embedding.detach().item()),
        "contrastive": float(loss_contrastive.detach().item()),
    }
    return total, metrics, (correct, token_count)


@torch.no_grad()
def validate(
    student: TaskConditionedJointStudent,
    teacher: FullEBMTeacher,
    loader: DataLoader,
    device: torch.device,
    weights: LossWeights,
    args: argparse.Namespace,
) -> Dict[str, float]:
    student.eval()
    totals = {
        "loss": 0.0,
        "lm_ce": 0.0,
        "lm_kd": 0.0,
        "embedding_kd": 0.0,
        "contrastive": 0.0,
    }
    correct = 0
    tokens = 0
    steps = 0
    for cpu_batch in tqdm(loader, desc="validation", leave=False):
        _, metrics, accuracy = compute_coupled_losses(
            student,
            teacher,
            cpu_batch,
            device,
            weights,
            teacher_top_k=args.teacher_top_k,
            kd_temperature=args.kd_temperature,
            contrastive_temperature=args.temperature,
        )
        for key in totals:
            totals[key] += metrics[key]
        correct += accuracy[0]
        tokens += accuracy[1]
        steps += 1
        if args.val_batches and steps >= args.val_batches:
            break
    student.train()
    result = {key: value / max(steps, 1) for key, value in totals.items()}
    result["token_accuracy"] = correct / max(tokens, 1)
    result["tokens"] = tokens
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Joint distillation requires CUDA for the 1.3B teacher")
    if args.epochs < 1 or args.grad_accum < 1:
        raise ValueError("epochs and grad-accum must be positive")
    if args.pair_batch_size < 2:
        raise ValueError(
            "pair-batch-size must be at least 2 so contrastive batches have negatives"
        )
    if any(
        weight < 0
        for weight in (
            args.lambda_lm_ce,
            args.lambda_lm_kd,
            args.lambda_embedding_kd,
            args.lambda_contrastive,
        )
    ):
        raise ValueError("loss weights must be non-negative")

    set_seed(args.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = REPO_ROOT / "model_checkpoints" / "joint_distill" / run_id
    output_dir = Path(args.output_dir or default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "train.log"
    metrics_path = output_dir / "metrics.jsonl"

    _, nova_tokenizer = load_ebm_tokenizer()
    student_config = StudentConfig(
        vocab_size=len(nova_tokenizer.tokenizer),
        pad_id=nova_tokenizer.tokenizer.pad_token_id or 0,
        d_model=args.d_model,
        num_layers=args.student_layers,
        num_heads=args.student_heads,
        ff_dim=args.student_ff_dim,
        dropout=args.dropout,
        max_length=args.max_length,
        teacher_embedding_dim=2_048,
    )
    student_config.validate()
    weights = LossWeights(
        lm_ce=args.lambda_lm_ce,
        lm_kd=args.lambda_lm_kd,
        embedding_kd=args.lambda_embedding_kd,
        contrastive=args.lambda_contrastive,
    )
    run_config = RunConfig(
        data=str(Path(args.data).resolve()),
        teacher_dir=str(Path(args.teacher_dir).resolve()),
        output_dir=str(output_dir),
        seed=args.seed,
        epochs=args.epochs,
        pair_batch_size=args.pair_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        temperature=args.temperature,
        kd_temperature=args.kd_temperature,
        teacher_top_k=args.teacher_top_k,
        val_fraction=args.val_fraction,
        val_batches=args.val_batches,
        num_workers=args.num_workers,
        device_id=args.device_id,
        condition_on_target=args.condition_on_target,
        max_train_pairs=args.max_train_pairs,
        max_val_pairs=args.max_val_pairs,
        student=student_config.to_dict(),
        loss_weights=asdict(weights),
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(run_config), handle, indent=2)

    log(f"Loading paired data: {args.data}", log_path)
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

    collator = JointDistillCollator(
        nova_tokenizer,
        max_length=args.max_length,
        condition_on_target=args.condition_on_target,
    )
    loader_args = dict(
        batch_size=args.pair_batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(JointPairDataset(train_pairs), shuffle=True, **loader_args)
    val_loader = DataLoader(JointPairDataset(val_pairs), shuffle=False, **loader_args)

    device = torch.device(f"cuda:{args.device_id}")
    log(f"Loading frozen EBM-full teacher: {args.teacher_dir}", log_path)
    teacher = FullEBMTeacher.load(args.teacher_dir, device_id=args.device_id)
    if teacher.embedding_weight.shape[0] != student_config.vocab_size:
        raise ValueError(
            "teacher/student vocabularies differ: "
            f"{teacher.embedding_weight.shape[0]} vs {student_config.vocab_size}"
        )

    student = TaskConditionedJointStudent(student_config)
    if not args.skip_teacher_embedding_init:
        log("Initializing compact token table by random projection of Nova embeddings", log_path)
        student.initialize_token_embeddings_from_teacher(
            teacher.embedding_weight, seed=args.seed
        )
    student = student.to(device)
    student.gradient_checkpointing_enable()
    actual_parameters = sum(parameter.numel() for parameter in student.parameters())
    estimated_parameters = estimate_parameter_count(student_config)
    if actual_parameters != estimated_parameters:
        raise RuntimeError(
            f"parameter estimator drifted: actual={actual_parameters:,}, "
            f"estimated={estimated_parameters:,}"
        )
    log(
        f"Joint student parameters: {actual_parameters:,} "
        f"({actual_parameters / 1e6:.1f}M; CLAP-size target)",
        log_path,
    )

    optimizer = AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_updates),
        num_training_steps=total_updates,
    )

    best_joint = float("inf")
    pareto_lm = float("inf")
    pareto_pred = float("inf")
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    student.train()

    for epoch in range(1, args.epochs + 1):
        running = {key: 0.0 for key in ("loss", "lm_ce", "lm_kd", "embedding_kd", "contrastive")}
        running_steps = 0
        correct = 0
        tokens = 0

        for step, cpu_batch in enumerate(
            tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"), start=1
        ):
            total, metrics, accuracy = compute_coupled_losses(
                student,
                teacher,
                cpu_batch,
                device,
                weights,
                teacher_top_k=args.teacher_top_k,
                kd_temperature=args.kd_temperature,
                contrastive_temperature=args.temperature,
            )
            (total / args.grad_accum).backward()
            for key in running:
                running[key] += metrics[key]
            running_steps += 1
            correct += accuracy[0]
            tokens += accuracy[1]

            should_update = step % args.grad_accum == 0 or step == len(train_loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if global_update % 20 == 0:
                    record = {
                        "event": "train",
                        "epoch": epoch,
                        "update": global_update,
                        **{key: value / max(running_steps, 1) for key, value in running.items()},
                        "token_accuracy": correct / max(tokens, 1),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    append_jsonl(metrics_path, record)
                    log(
                        f"update={global_update}/{total_updates} "
                        f"loss={record['loss']:.4f} lm_ce={record['lm_ce']:.4f} "
                        f"lm_kd={record['lm_kd']:.4f} emb={record['embedding_kd']:.4f} "
                        f"cont={record['contrastive']:.4f} "
                        f"tok_acc={record['token_accuracy']:.4f}",
                        log_path,
                    )
                    running = {key: 0.0 for key in running}
                    running_steps = 0
                    correct = 0
                    tokens = 0

        validation = validate(
            student, teacher, val_loader, device, weights, args
        )
        validation_record = {
            "event": "validation",
            "epoch": epoch,
            "update": global_update,
            **validation,
        }
        append_jsonl(metrics_path, validation_record)
        log(
            f"validation epoch={epoch} loss={validation['loss']:.4f} "
            f"lm_ce={validation['lm_ce']:.4f} lm_kd={validation['lm_kd']:.4f} "
            f"emb={validation['embedding_kd']:.4f} "
            f"cont={validation['contrastive']:.4f} "
            f"tok_acc={validation['token_accuracy']:.4f}",
            log_path,
        )

        if validation["loss"] < best_joint:
            best_joint = validation["loss"]
            student.save_pretrained(output_dir / "best_joint")
        pred_validation = validation["embedding_kd"] + validation["contrastive"]
        improves_both = (
            validation["lm_ce"] <= pareto_lm
            and pred_validation <= pareto_pred
            and (validation["lm_ce"] < pareto_lm or pred_validation < pareto_pred)
        )
        if improves_both:
            pareto_lm = validation["lm_ce"]
            pareto_pred = pred_validation
            student.save_pretrained(output_dir / "best_dual_canary")

    student.save_pretrained(output_dir / "final")
    summary = {
        "parameters": actual_parameters,
        "best_joint_validation_loss": best_joint,
        "best_dual_lm_ce": pareto_lm,
        "best_dual_pred_loss": pareto_pred,
        "updates": global_update,
        "checkpoints": ["best_joint", "best_dual_canary", "final"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(f"Training complete: {output_dir}", log_path)


if __name__ == "__main__":
    main()
