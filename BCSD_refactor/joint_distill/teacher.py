"""Frozen adapter-free interface to the merged full-EBM teacher checkpoint."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from shared.nova_utils import NOVA_CACHE_DIR
from shared.pooling import AttentionPooling


@dataclass
class TeacherLMTargets:
    indices: torch.Tensor
    logits: torch.Tensor


class FullEBMTeacher:
    """Serve causal logits and BCSD embeddings from EBM-full ``s3_final``."""

    def __init__(self, model, pooling_head: AttentionPooling) -> None:
        self.model = model
        self.pooling_head = pooling_head
        self.model.requires_grad_(False).eval()
        self.model.config.use_cache = False
        self.pooling_head.requires_grad_(False).eval()

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        *,
        device_id: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "FullEBMTeacher":
        checkpoint = Path(checkpoint_dir).resolve()
        model_path = checkpoint / "model.safetensors"
        head_path = checkpoint / "pooling_head.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"missing merged EBM model: {model_path}")
        if not head_path.is_file():
            raise FileNotFoundError(f"missing EBM pooling head: {head_path}")

        if NOVA_CACHE_DIR not in sys.path:
            sys.path.insert(0, NOVA_CACHE_DIR)
        from modeling_nova import NovaForCausalLM

        model = NovaForCausalLM.from_pretrained(
            str(checkpoint),
            torch_dtype=dtype,
            device_map={"": device_id},
            local_files_only=True,
        )
        head = AttentionPooling(model.config.hidden_size)
        head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        device = next(model.parameters()).device
        head = head.to(device=device, dtype=dtype)
        return cls(model, head)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def embedding_weight(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight

    @torch.no_grad()
    def lm_targets(self, batch: Dict[str, torch.Tensor], *, top_k: int) -> TeacherLMTargets:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            nova_attention_mask=batch["nova_attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        shifted = outputs.logits[:, :-1, :]
        values, indices = torch.topk(shifted, min(top_k, shifted.shape[-1]), dim=-1)
        return TeacherLMTargets(indices=indices.detach(), logits=values.detach())

    @torch.no_grad()
    def pred_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
        label_positions: List[List[int]],
    ) -> torch.Tensor:
        outputs = self.model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            nova_attention_mask=batch["nova_attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        embeddings = self.pooling_head(outputs.last_hidden_state, label_positions)
        return torch.nn.functional.normalize(embeddings.float(), p=2, dim=1).detach()

    @torch.no_grad()
    def hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        nova_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the final-layer hidden states for representation distillation.

        The caller controls causal versus bidirectional behavior through
        ``nova_attention_mask``: pass the native hierarchical mask for a causal
        view, or its ``make_bidirectional_nova_mask`` symmetrization for a
        bidirectional view.  Outputs are token-aligned with ``input_ids``.
        """
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            nova_attention_mask=nova_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state.detach()
