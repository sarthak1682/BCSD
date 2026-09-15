"""Representation-distillation losses for the two-stage pretraining."""

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
    """Masked mean-squared error between projected student and teacher states.

    Matches the supervisor's definition

    .. math::
        \\mathcal{L}_{\\mathrm{KD}}
        = \\frac{1}{N d_T}\\sum_{i} \\lVert P h_i^S - h_i^T \\rVert_2^2,

    where ``N`` is the number of positions selected by ``mask`` and ``d_T`` is
    the teacher hidden dimension.  ``projection`` maps the student hidden dim to
    the teacher hidden dim and is omitted when they already match.

    Args:
        student_hidden: ``[B, L, d_S]`` final-layer student states.
        teacher_hidden: ``[B, L, d_T]`` final-layer teacher states.
        mask: ``[B, L]`` bool; True for positions to distill.
        projection: optional ``Linear(d_S, d_T)``.
    """
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
