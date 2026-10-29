from __future__ import annotations

from typing import Iterable

import torch


def grad_norm_wrt(loss: torch.Tensor, parameters: Iterable[torch.Tensor]) -> float:
    """Full-parameter L2 gradient norm of ``loss`` w.r.t. ``parameters``.

    Uses ``torch.autograd.grad`` with ``retain_graph=True`` so the caller can
    still backward the combined objective afterwards.  Parameters unused by
    ``loss`` (for example the retrieval head during pretraining) contribute
    zero.  The returned value is a detached Python float, which makes the
    scaling factor stop-grad by construction.
    """
    grads = torch.autograd.grad(loss, list(parameters), retain_graph=True, allow_unused=True)
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().float().pow(2).sum().item())
    return total ** 0.5


class GradientBalance:
    """EMA of the GT/KD gradient-norm ratio, one state per mode.

    For mode ``m`` the update is

    .. math::
        s_m(t) = \\beta s_m(t-1) + (1-\\beta)\\, \\frac{\\|\\nabla_\\theta
        \\mathcal{L}_{\\mathrm{GT}}^m\\|_2}
        {\\|\\nabla_\\theta \\mathcal{L}_{\\mathrm{KD}}^m\\|_2 + \\epsilon}.
    """

    def __init__(self, beta: float = 0.9, epsilon: float = 1e-6, init_value: float = 1.0) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self.beta = beta
        self.epsilon = epsilon
        self.values = {"C": float(init_value), "B": float(init_value)}

    def update(
        self,
        mode: str,
        gt_loss: torch.Tensor,
        kd_loss: torch.Tensor,
        parameters: Iterable[torch.Tensor],
    ) -> float:
        """Update and return the scaling factor for ``mode`` (``'C'`` or ``'B'``)."""
        if mode not in self.values:
            raise ValueError(f"unknown mode: {mode}")
        gt_norm = grad_norm_wrt(gt_loss, parameters)
        kd_norm = grad_norm_wrt(kd_loss, parameters)
        ratio = gt_norm / (kd_norm + self.epsilon)
        self.values[mode] = self.beta * self.values[mode] + (1.0 - self.beta) * ratio
        return self.values[mode]


def teacher_lambda(
    step: int,
    total_steps: int,
    *,
    lambda_high: float = 2.0,
    lambda_low: float = 0.1,
    anneal_fraction: float = 0.3,
) -> float:
    """Teacher-first schedule: anneal ``lambda`` linearly from high to low.

    Distillation initially dominates (``lambda_high > 1``) and is annealed over
    the first ``anneal_fraction`` of training until it reaches the non-zero
    ``lambda_low < 1`` regularizer strength.
    """
    if lambda_high <= 0 or lambda_low <= 0:
        raise ValueError("lambda bounds must be positive")
    if not 0.0 < anneal_fraction < 1.0:
        raise ValueError("anneal_fraction must be in (0, 1)")
    boundary = max(1, int(total_steps * anneal_fraction))
    if step >= boundary:
        return lambda_low
    frac = step / boundary
    return lambda_high + (lambda_low - lambda_high) * frac
