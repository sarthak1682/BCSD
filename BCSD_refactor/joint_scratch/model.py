"""One randomly initialized transformer for causal LM and BCSD retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from joint_scratch.config import ScratchModelConfig


class Task(str, Enum):
    LM = "lm"
    RETRIEVAL = "retrieval"


@dataclass
class ScratchJointOutput:
    hidden_states: torch.Tensor
    logits: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None


class ScratchJointModel(nn.Module):
    """Shared backbone with an explicit task token and task attention contract.

    LM mode uses a strict causal mask and the tied vocabulary head. Retrieval
    mode uses full bidirectional attention and an attention-pooling head. No
    pretrained or teacher parameters are accepted by this class.
    """

    CONFIG_NAME = "model_config.json"
    WEIGHTS_NAME = "model.pt"

    def __init__(self, config: ScratchModelConfig) -> None:
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
        self.retrieval_projection = nn.Linear(config.d_model, config.embedding_dim)
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
        """Return a boolean upper triangle where ``True`` forbids attention."""

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
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        attention_mask = attention_mask.bool()

        token_states = self.token_embedding(input_ids)
        task_state = self.task_embeddings[self._task_index(task)].view(1, 1, -1)
        states = torch.cat((task_state.expand(batch_size, -1, -1), token_states), dim=1)
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
        return encoded[:, 1:, :]

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def freeze_backbone(self, *, train_task_embedding: bool = False) -> None:
        """Freeze all parameters except the retrieval head.

        Keeps ``pool_attention`` and ``retrieval_projection`` trainable so a
        contrastive BCSD head can be fit on top of a frozen representation.
        ``lm_head`` is weight-tied to ``token_embedding`` and is therefore
        frozen with it. When ``train_task_embedding`` is false, the task
        embeddings are frozen too; their retrieval row was already trained
        through the bidirectional MLM path during pre-CL.
        """
        for param in self.parameters():
            param.requires_grad = False
        for param in self.pool_attention.parameters():
            param.requires_grad = True
        for param in self.retrieval_projection.parameters():
            param.requires_grad = True
        if train_task_embedding:
            self.task_embeddings.requires_grad = True

    def forward_lm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ScratchJointOutput:
        hidden = self._encode(input_ids, attention_mask, Task.LM)
        return ScratchJointOutput(hidden_states=hidden, logits=self.lm_head(hidden))

    def forward_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ScratchJointOutput:
        """Bidirectional masked-token prediction on the encoder/retrieval path."""

        hidden = self._encode(input_ids, attention_mask, Task.RETRIEVAL)
        return ScratchJointOutput(hidden_states=hidden, logits=self.lm_head(hidden))

    def pool_hidden(self, hidden_states: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        """Attention-pool token states into one ``d_model`` vector per sequence."""
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
        return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

    def pool(self, hidden_states: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pool_hidden(hidden_states, pool_mask)
        return self.retrieval_projection(pooled)

    def forward_retrieval(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        pool_mask: torch.Tensor,
        *,
        normalize: bool = False,
    ) -> ScratchJointOutput:
        hidden = self._encode(input_ids, attention_mask, Task.RETRIEVAL)
        embeddings = self.pool(hidden, pool_mask)
        if normalize:
            embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        return ScratchJointOutput(hidden_states=hidden, embeddings=embeddings)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
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
    ) -> "ScratchJointModel":
        checkpoint = Path(checkpoint_dir)
        with (checkpoint / cls.CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = ScratchModelConfig.from_dict(json.load(handle))
        model = cls(config)
        state = torch.load(
            checkpoint / cls.WEIGHTS_NAME,
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(state)
        return model
