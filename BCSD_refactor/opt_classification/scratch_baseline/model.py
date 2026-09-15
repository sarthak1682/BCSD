"""Student+LAL optimization-classification model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from shared.student_model import LatentAttentionLayer, StudentDistillationModule


@dataclass(frozen=True)
class ParameterCounts:
    """Parameter counts for the scratch baseline components."""

    backbone: int
    latent_attention: int
    classifier: int

    @property
    def total(self) -> int:
        return self.backbone + self.latent_attention + self.classifier

    def as_dict(self) -> dict[str, int]:
        return {
            "backbone": self.backbone,
            "latent_attention": self.latent_attention,
            "classifier": self.classifier,
            "total": self.total,
        }


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class StudentOptClassifier(nn.Module):
    """Randomly initialized student encoder, LAL pooling, and linear head."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int = 2048,
        num_layers: int = 2,
        num_latents: int = 512,
        num_classes: int = 4,
        pad_id: int = 0,
        normalize_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.normalize_embeddings = normalize_embeddings
        self.backbone = StudentDistillationModule(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pad_id=pad_id,
        )
        self.latent_attention = LatentAttentionLayer(
            hidden_dim=hidden_dim,
            num_latents=num_latents,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key_padding_mask is None:
            key_padding_mask = input_ids == self.pad_id
        hidden = self.backbone(input_ids, key_padding_mask=key_padding_mask)
        embeddings = self.latent_attention(
            hidden,
            key_padding_mask=key_padding_mask,
            pool_mask=key_padding_mask,
        )
        if self.normalize_embeddings:
            embeddings = functional.normalize(embeddings.float(), p=2, dim=1).to(
                embeddings.dtype
            )
        return self.classifier(embeddings)

    def parameter_counts(self) -> ParameterCounts:
        return ParameterCounts(
            backbone=count_trainable_parameters(self.backbone),
            latent_attention=count_trainable_parameters(self.latent_attention),
            classifier=count_trainable_parameters(self.classifier),
        )


def build_model_from_config(
    config: dict[str, Any],
    *,
    pad_id: int,
    num_latents: int = 512,
    num_classes: int = 4,
    normalize_embeddings: bool = False,
) -> StudentOptClassifier:
    return StudentOptClassifier(
        vocab_size=int(config["vocab_size"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_latents=num_latents,
        num_classes=num_classes,
        pad_id=pad_id,
        normalize_embeddings=normalize_embeddings,
    )

