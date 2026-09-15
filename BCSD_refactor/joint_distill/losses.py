"""Losses for coupled language-model and retrieval distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def shifted_causal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Next-token cross entropy over non-ignored target tokens."""
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must have matching [B,L] dimensions")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(-100)
    if not torch.any(valid):
        return shifted_logits.sum() * 0.0
    return F.cross_entropy(shifted_logits[valid].float(), shifted_labels[valid])


def shifted_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    if not torch.any(valid):
        return 0, 0
    predictions = shifted_logits.argmax(dim=-1)
    correct = predictions[valid].eq(shifted_labels[valid]).sum().item()
    return int(correct), int(valid.sum().item())


def topk_teacher_kl(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Renormalized KL on the teacher's top-k next-token support."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_shifted = student_logits[:, :-1, :]
    if teacher_indices.shape != teacher_logits.shape:
        raise ValueError("teacher top-k indices/logits shapes differ")
    if teacher_indices.shape[:2] != student_shifted.shape[:2]:
        raise ValueError("teacher and student sequence dimensions differ")

    valid = labels[:, 1:].ne(-100)
    if not torch.any(valid):
        return student_shifted.sum() * 0.0
    selected_student = student_shifted.gather(-1, teacher_indices)
    teacher_prob = torch.softmax(teacher_logits.float() / temperature, dim=-1)
    student_log_prob = torch.log_softmax(selected_student.float() / temperature, dim=-1)
    teacher_log_prob = torch.log(teacher_prob.clamp_min(1e-9))
    per_token = torch.sum(teacher_prob * (teacher_log_prob - student_log_prob), dim=-1)
    return per_token[valid].mean() * (temperature * temperature)


def embedding_distillation_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    if student_embeddings.shape != teacher_embeddings.shape:
        raise ValueError("student and teacher embedding shapes differ")
    student = F.normalize(student_embeddings.float(), p=2, dim=1)
    teacher = F.normalize(teacher_embeddings.float(), p=2, dim=1)
    return (1.0 - torch.sum(student * teacher, dim=1)).mean()
