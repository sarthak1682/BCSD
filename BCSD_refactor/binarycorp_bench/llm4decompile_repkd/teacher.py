from __future__ import annotations

from types import MethodType
from typing import Optional

import torch
import torch.nn as nn

from binarycorp_bench.llm4decompile_causal_pooling.core import (
    FrozenCausalDecompiler,
    load_llm4decompile,
)
from shared.qwen_utils import make_bidirectional_4d_mask


def _decoder_body(model: nn.Module) -> nn.Module:
    body = model
    for _ in range(4):
        if hasattr(body, "_update_causal_mask"):
            return body
        nxt = getattr(body, "model", None)
        if nxt is None or nxt is body:
            break
        body = nxt
    return getattr(model, "model", model)


def patch_decoder_honor_4d_mask(model: nn.Module) -> nn.Module:
    body = _decoder_body(model)
    if getattr(body, "_l4d_honor_4d_mask", False):
        return model
    original = getattr(body, "_update_causal_mask", None)
    if original is None:
        return model

    def _update_causal_mask(self, attention_mask, input_tensor, *args, **kwargs):
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        return original(attention_mask, input_tensor, *args, **kwargs)

    body._update_causal_mask = MethodType(_update_causal_mask, body)
    body._l4d_honor_4d_mask = True
    return model


def attention_mask_4d(
    input_ids: torch.Tensor,
    pad_id: int,
    *,
    bidirectional: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = make_bidirectional_4d_mask(input_ids, pad_id, dtype=dtype)
    if bidirectional:
        return mask
    length = input_ids.shape[1]
    future = torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=input_ids.device),
        diagonal=1,
    )
    return mask.masked_fill(future.view(1, 1, length, length), torch.finfo(dtype).min)


class FrozenL4DTeacher(nn.Module):
    def __init__(self, tokenizer, causal_lm: nn.Module) -> None:
        super().__init__()
        patch_decoder_honor_4d_mask(causal_lm)
        self.tokenizer = tokenizer
        self.decompiler = FrozenCausalDecompiler(causal_lm)
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("LLM4Decompile tokenizer must define pad_token_id")

    @classmethod
    def load(cls, model_id: str, device: torch.device) -> "FrozenL4DTeacher":
        tokenizer, model = load_llm4decompile(
            model_id, device, attn_implementation="eager"
        )
        return cls(tokenizer, model)

    @property
    def hidden_dim(self) -> int:
        return self.decompiler.hidden_dim

    @property
    def honors_4d_mask(self) -> bool:
        body = _decoder_body(self.decompiler.causal_lm)
        return bool(getattr(body, "_l4d_honor_4d_mask", False))

    @torch.no_grad()
    def assert_causal_and_bidirectional_differ(self, device: torch.device) -> float:
        text = "# This is the assembly code:\n<func0>:\nmov eax, ebx\nret\n# What is the source code?\n"
        encoded = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=True, padding=False
        )
        input_ids = encoded["input_ids"].to(device)
        attention = torch.ones_like(input_ids, dtype=torch.bool)
        causal = self.hidden_states(input_ids, attention, bidirectional=False)
        bidir = self.hidden_states(input_ids, attention, bidirectional=True)
        delta = float((causal.float() - bidir.float()).abs().max().item())
        if delta <= 0.0:
            raise RuntimeError(
                "teacher attention views are identical; "
                "4D attention masks are not taking effect"
            )
        return delta

    @torch.no_grad()
    def hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        bidirectional: bool,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        del attention_mask
        if dtype is None:
            dtype = next(self.decompiler.causal_lm.parameters()).dtype
        mask_4d = attention_mask_4d(
            input_ids, int(self.pad_id), bidirectional=bidirectional, dtype=dtype
        )
        return self.decompiler(input_ids, mask_4d)


class FrozenGenericCoderTeacher(FrozenL4DTeacher):
    """Input formatting is supplied by the collator; no decompile prompt is added."""

    @classmethod
    def load(cls, model_id: str, device: torch.device) -> "FrozenGenericCoderTeacher":
        tokenizer, model = load_llm4decompile(
            model_id, device, attn_implementation="eager"
        )
        return cls(tokenizer, model)

    @torch.no_grad()
    def assert_causal_and_bidirectional_differ(self, device: torch.device) -> float:
        text = "push rbp\nmov rbp, rsp\nmov eax, edi\npop rbp\nret"
        encoded = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=True, padding=False
        )
        input_ids = encoded["input_ids"].to(device)
        attention = torch.ones_like(input_ids, dtype=torch.bool)
        causal = self.hidden_states(input_ids, attention, bidirectional=False)
        bidir = self.hidden_states(input_ids, attention, bidirectional=True)
        delta = float((causal.float() - bidir.float()).abs().max().item())
        if delta <= 0.0:
            raise RuntimeError(
                "generic-coder attention views are identical; "
                "4D attention masks are not taking effect"
            )
        return delta
