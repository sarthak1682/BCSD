"""Configuration for the compact joint model trained from scratch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ScratchModelConfig:
    """Architecture defaults for the approximately CLAP-sized model."""

    vocab_size: int = 32_280
    pad_id: int = 0
    d_model: int = 768
    num_layers: int = 10
    num_heads: int = 12
    ff_dim: int = 3_072
    dropout: float = 0.1
    max_length: int = 1_024
    embedding_dim: int = 768

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("pad_id must be inside the vocabulary")
        if min(self.d_model, self.num_layers, self.num_heads, self.ff_dim) <= 0:
            raise ValueError("model dimensions and num_layers must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.max_length <= 1 or self.embedding_dim <= 0:
            raise ValueError("max_length must exceed one and embedding_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ScratchModelConfig":
        config = cls(**values)
        config.validate()
        return config


def estimate_parameter_count(config: ScratchModelConfig) -> int:
    """Return the exact count implied by :mod:`joint_scratch.model`."""

    config.validate()
    d = config.d_model
    ff = config.ff_dim
    token_embedding = config.vocab_size * d
    positional_embedding = (config.max_length + 1) * d
    task_embeddings = 2 * d
    encoder_layer = 4 * d * d + 2 * d * ff + 9 * d + ff
    encoder = config.num_layers * encoder_layer
    final_norm = 2 * d
    attention_pool = d * d + 2 * d
    retrieval_projection = d * config.embedding_dim + config.embedding_dim
    return (
        token_embedding
        + positional_embedding
        + task_embeddings
        + encoder
        + final_norm
        + attention_pool
        + retrieval_projection
    )
