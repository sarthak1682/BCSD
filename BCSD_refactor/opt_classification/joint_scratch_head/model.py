"""Frozen pre-CL joint-scratch backbone with a trainable opt-class head.

This is the opt-classification analog of the BCSD frozen-head experiment: the
pre-CL :class:`~joint_scratch.model.ScratchJointModel` backbone is kept fixed and
only an attention-pooling + linear head is fit on top, so the same representation
space is probed by a BCSD head (contrastive) and an optimization head
(cross-entropy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from joint_scratch.model import ScratchJointModel, Task


@dataclass(frozen=True)
class ParameterCounts:
    backbone: int
    pool_attention: int
    classifier: int

    @property
    def total_trainable(self) -> int:
        return self.pool_attention + self.classifier

    def as_dict(self) -> dict[str, int]:
        return {
            "backbone": self.backbone,
            "pool_attention": self.pool_attention,
            "classifier": self.classifier,
            "total_trainable": self.total_trainable,
        }


def _count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class JointScratchOptClassifier(nn.Module):
    """Attention-pool frozen retrieval states and classify optimization level."""

    def __init__(self, backbone: ScratchJointModel, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.opt_classifier = nn.Linear(backbone.config.d_model, num_classes)
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.pool_attention.parameters():
            parameter.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key_padding_mask is None:
            key_padding_mask = input_ids == self.backbone.pad_id
        attention_mask = ~key_padding_mask
        hidden = self.backbone._encode(input_ids, attention_mask, Task.RETRIEVAL)
        pooled = self.backbone.pool_hidden(hidden, attention_mask)
        return self.opt_classifier(pooled)

    def parameter_counts(self) -> ParameterCounts:
        return ParameterCounts(
            backbone=_count(self.backbone),
            pool_attention=_count(self.backbone.pool_attention),
            classifier=_count(self.opt_classifier),
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        num_classes: int = 4,
        map_location: str | torch.device = "cpu",
    ) -> "JointScratchOptClassifier":
        backbone = ScratchJointModel.from_pretrained(
            checkpoint_dir, map_location=map_location
        )
        return cls(backbone, num_classes=num_classes)
