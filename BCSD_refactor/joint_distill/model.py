from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from joint_distill.config import StudentConfig


class Task(str, Enum):
    LM = "lm"
    PRED = "pred"


@dataclass
class JointStudentOutput:
    hidden_states: torch.Tensor
    logits: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None


class TaskConditionedJointStudent(nn.Module):
    """One shared transformer for causal translation and BCSD retrieval.

    A learned task vector is prepended internally.  Public token-aligned outputs
    exclude that vector, keeping LM labels and retrieval masks aligned with the
    original unprefixed token sequence.
    """

    CONFIG_NAME = "student_config.json"
    WEIGHTS_NAME = "student_model.pt"

    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.pad_id = config.pad_id

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.position_embedding = nn.Embedding(config.max_length + 1, config.d_model)
        self.task_embeddings = nn.Parameter(torch.empty(2, config.d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )

        self.pool_attention = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Linear(config.d_model, 1, bias=False),
        )
        self.retrieval_projection = nn.Linear(
            config.d_model, config.teacher_embedding_dim
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.gradient_checkpointing = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].zero_()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.task_embeddings, mean=0.0, std=0.02)

    @staticmethod
    def _task_index(task: Task) -> int:
        return 0 if task == Task.LM else 1

    @staticmethod
    def causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
        """Boolean mask where True entries are forbidden future-token edges."""
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        task: Task,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.max_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_length "
                f"{self.config.max_length}"
            )
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_id)
        attention_mask = attention_mask.bool()
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")

        token_states = self.token_embedding(input_ids)
        task_state = self.task_embeddings[self._task_index(task)].view(1, 1, -1)
        task_state = task_state.expand(batch_size, -1, -1)
        states = torch.cat((task_state, token_states), dim=1)

        positions = torch.arange(
            sequence_length + 1, device=input_ids.device, dtype=torch.long
        )
        states = states + self.position_embedding(positions).unsqueeze(0)

        valid = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=torch.bool, device=input_ids.device),
                attention_mask,
            ),
            dim=1,
        )
        key_padding_mask = ~valid
        sequence_mask = None
        if task == Task.LM:
            sequence_mask = self.causal_attention_mask(sequence_length + 1, input_ids.device)

        if self.gradient_checkpointing and self.training:
            encoded = states
            for layer in self.encoder.layers:
                def layer_forward(value, module=layer):
                    return module(
                        value,
                        src_mask=sequence_mask,
                        src_key_padding_mask=key_padding_mask,
                    )

                encoded = checkpoint(layer_forward, encoded, use_reentrant=False)
            if self.encoder.norm is not None:
                encoded = self.encoder.norm(encoded)
        else:
            encoded = self.encoder(
                states,
                mask=sequence_mask,
                src_key_padding_mask=key_padding_mask,
            )
        # Hide the implementation-only task position from every caller.  This
        # is the central offset invariant: external masks/labels remain [B,L].
        return encoded[:, 1:, :]

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward_lm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> JointStudentOutput:
        hidden = self._encode(input_ids, attention_mask, Task.LM)
        return JointStudentOutput(hidden_states=hidden, logits=self.lm_head(hidden))

    def pool(
        self,
        hidden_states: torch.Tensor,
        pool_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or pool_mask.ndim != 2:
            raise ValueError("expected hidden_states [B,L,D] and pool_mask [B,L]")
        if hidden_states.shape[:2] != pool_mask.shape:
            raise ValueError("pool_mask must match the token dimensions")
        pool_mask = pool_mask.bool()
        if not torch.all(pool_mask.any(dim=1)):
            raise ValueError("every sequence needs at least one pooling position")

        scores = self.pool_attention(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~pool_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=1).to(hidden_states.dtype)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return self.retrieval_projection(pooled)

    def forward_pred(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        pool_mask: torch.Tensor,
        *,
        normalize: bool = False,
    ) -> JointStudentOutput:
        hidden = self._encode(input_ids, attention_mask, Task.PRED)
        embeddings = self.pool(hidden, pool_mask)
        if normalize:
            embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        return JointStudentOutput(hidden_states=hidden, embeddings=embeddings)

    @torch.no_grad()
    def initialize_token_embeddings_from_teacher(
        self,
        teacher_embeddings: torch.Tensor,
        *,
        seed: int = 42,
        chunk_size: int = 2_048,
    ) -> None:
        """Random-project Nova token embeddings into the compact hidden space.

        This transfers token-level information without retaining a frozen 66M
        parameter Nova embedding table in the deployed student.
        """
        if teacher_embeddings.ndim != 2:
            raise ValueError("teacher_embeddings must be a matrix")
        rows = min(teacher_embeddings.shape[0], self.config.vocab_size)
        source_dim = teacher_embeddings.shape[1]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.randn(
            source_dim,
            self.config.d_model,
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(source_dim)
        projection = projection.to(teacher_embeddings.device)

        target = self.token_embedding.weight
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            projected = teacher_embeddings[start:end].float() @ projection
            target[start:end].copy_(projected.to(device=target.device, dtype=target.dtype))
        if self.token_embedding.padding_idx is not None:
            target[self.token_embedding.padding_idx].zero_()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedy generation for translation evaluation (currently batch size one)."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("generate currently supports a single sequence")
        generated = input_ids
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.config.max_length:
                break
            attention = torch.ones_like(generated, dtype=torch.bool)
            logits = self.forward_lm(generated, attention).logits
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break
        return generated

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / self.CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(self.config.to_dict(), handle, indent=2)
        torch.save(self.state_dict(), output / self.WEIGHTS_NAME)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "TaskConditionedJointStudent":
        checkpoint = Path(checkpoint_dir)
        with (checkpoint / cls.CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = StudentConfig.from_dict(json.load(handle))
        model = cls(config)
        state = torch.load(
            checkpoint / cls.WEIGHTS_NAME,
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(state)
        return model
