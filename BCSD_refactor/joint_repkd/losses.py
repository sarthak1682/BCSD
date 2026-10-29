from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def hidden_state_distillation_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    mask: torch.Tensor,
    projection: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Masked MSE, normalized by the selected token count and teacher hidden dimension.

    student_hidden: [B, L, d_S]; teacher_hidden: [B, L, d_T].
    mask: [B, L], True for positions to distill.
    projection maps d_S to d_T and may be omitted when they match."""
    if student_hidden.shape[:2] != teacher_hidden.shape[:2]:
        raise ValueError("student/teacher hidden sequence shapes differ")
    if student_hidden.shape[:2] != mask.shape:
        raise ValueError("mask must match the hidden sequence shape")

    student = student_hidden.float()
    if projection is not None:
        student = projection(student)
    teacher = teacher_hidden.float()

    diff = (student - teacher) ** 2
    weight = mask.unsqueeze(-1).to(diff.dtype)
    denominator = (weight.sum() * teacher.shape[-1]).clamp_min(1.0)
    return (diff * weight).sum() / denominator
