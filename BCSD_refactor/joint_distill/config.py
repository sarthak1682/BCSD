"""Configuration and parameter-budget helpers for the compact joint student."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class StudentConfig:
    """Architecture of the decoder/encoder student.

    The defaults intentionally land in CLAP's parameter class.  The language
    modeling head is tied to the token embedding, so it does not add another
    vocabulary-sized matrix.
    """

    vocab_size: int = 32_280
    pad_id: int = 0
    d_model: int = 768
    num_layers: int = 10
    num_heads: int = 12
    ff_dim: int = 3_072
    dropout: float = 0.1
    max_length: int = 1_024
    teacher_embedding_dim: int = 2_048

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("pad_id must be inside the vocabulary")
        if self.d_model <= 0 or self.num_layers <= 0 or self.ff_dim <= 0:
            raise ValueError("model dimensions and num_layers must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.max_length <= 0 or self.teacher_embedding_dim <= 0:
            raise ValueError("max_length and teacher_embedding_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "StudentConfig":
        config = cls(**values)
        config.validate()
        return config


def estimate_parameter_count(config: StudentConfig) -> int:
    """Return the exact parameter count implied by ``model.py``.

    This mirrors PyTorch's TransformerEncoderLayer layout, including biases and
    layer norms.  It is useful before allocating the roughly 100M-parameter
    default model.
    """

    config.validate()
    d = config.d_model
    ff = config.ff_dim

    token_embedding = config.vocab_size * d
    positional_embedding = (config.max_length + 1) * d
    task_embeddings = 2 * d

    # MHA: qkv weight+bias, output weight+bias. FFN: two affine layers.
    # Two LayerNorms each contribute a scale and bias.
    encoder_layer = 4 * d * d + 2 * d * ff + 9 * d + ff
    encoder = config.num_layers * encoder_layer
    final_norm = 2 * d

    # Scalar attention pool and retrieval projection into teacher space.
    attention_pool = d * d + 2 * d
    retrieval_projection = d * config.teacher_embedding_dim + config.teacher_embedding_dim

    # lm_head is tied to token_embedding and therefore contributes zero.
    return (
        token_embedding
        + positional_embedding
        + task_embeddings
        + encoder
        + final_norm
        + attention_pool
        + retrieval_projection
    )
