from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from binarycorp_bench.nova_causal_pooling.core import (  # noqa: F401
    LinearMeanProbe,
    MaskedAttentionPooling,
    build_bench_pairs,
    masked_mean_pool,
)
from shared.data_utils import asm_to_text


MODEL_ID = "LLM4Binary/llm4decompile-1.3b-v1.5"
PROMPT_PREFIX = "# This is the assembly code:\n"
FUNC_HEADER = "<func0>:\n"
PROMPT_SUFFIX = "\n# What is the source code?\n"


def default_jsonl(split: str) -> str:
    name = f"output_benchset_rebalanced_{split}_clap_strict.jsonl"
    candidates = [
        Path(__file__).resolve().parents[3] / "nvemb" / name,
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return str(candidates[0])


def build_decompile_prompt(assembly: str) -> str:
    body = asm_to_text(assembly).strip()
    if not body:
        raise ValueError("Cannot encode an empty assembly function")
    return f"{PROMPT_PREFIX}{FUNC_HEADER}{body}{PROMPT_SUFFIX}"


def load_llm4decompile(
    model_id: str,
    device: torch.device,
    *,
    attn_implementation: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"{model_id} has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    load_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    return tokenizer, model


def _encode_without_specials(tokenizer, text: str) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in ids]


class CausalDecompilePairCollator:
    """Use the decompile prompt on both sides; pool only the function header and assembly."""

    def __init__(
        self,
        tokenizer,
        *,
        max_length: int = 1024,
        prompt_prefix: str = PROMPT_PREFIX,
        func_header: str = FUNC_HEADER,
        prompt_suffix: str = PROMPT_SUFFIX,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_prefix = prompt_prefix
        self.func_header = func_header
        self.prompt_suffix = prompt_suffix
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id")

        self.prefix_ids = _encode_without_specials(tokenizer, prompt_prefix)
        if getattr(tokenizer, "bos_token_id", None) is not None:
            self.prefix_ids = [int(tokenizer.bos_token_id)] + self.prefix_ids
        self.suffix_ids = _encode_without_specials(tokenizer, prompt_suffix)
        reserved = len(self.prefix_ids) + len(self.suffix_ids)
        if reserved >= max_length:
            raise ValueError(
                f"max_length={max_length} leaves no room for assembly "
                f"(prefix={len(self.prefix_ids)}, suffix={len(self.suffix_ids)})"
            )
        self._body_budget = max_length - reserved

    def _encode(self, assembly: str) -> Tuple[np.ndarray, np.ndarray]:
        body = asm_to_text(assembly).strip()
        if not body:
            raise ValueError("Cannot encode an empty assembly function")
        body_ids = _encode_without_specials(self.tokenizer, self.func_header + body)
        if not body_ids:
            raise ValueError("Tokenizer produced no assembly tokens")
        body_ids = body_ids[: self._body_budget]
        input_ids = np.asarray(
            self.prefix_ids + body_ids + self.suffix_ids, dtype=np.int64
        )
        pool_mask = np.zeros(len(input_ids), dtype=np.bool_)
        start = len(self.prefix_ids)
        pool_mask[start : start + len(body_ids)] = True
        return input_ids, pool_mask

    def __call__(self, batch: Sequence[Tuple[str, str, int]]) -> Dict[str, torch.Tensor]:
        encoded_rows = []
        function_ids: List[int] = []
        for query, positive, function_id in batch:
            encoded_rows.append(self._encode(query))
            encoded_rows.append(self._encode(positive))
            function_ids.extend((function_id, function_id))

        max_length = max(len(row[0]) for row in encoded_rows)
        batch_size = len(encoded_rows)
        input_ids = np.full((batch_size, max_length), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_length), dtype=np.bool_)
        pool_mask = np.zeros((batch_size, max_length), dtype=np.bool_)

        for row_index, (ids, pool) in enumerate(encoded_rows):
            length = len(ids)
            input_ids[row_index, :length] = ids
            attention_mask[row_index, :length] = True
            pool_mask[row_index, :length] = pool

        if not np.all(pool_mask.any(axis=1)):
            raise RuntimeError("Every encoded function must have at least one pooling token")

        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "pool_mask": torch.from_numpy(pool_mask),
            "func_ids": torch.tensor(function_ids, dtype=torch.long),
        }


class FrozenCausalDecompiler(nn.Module):
    def __init__(self, causal_lm: nn.Module) -> None:
        super().__init__()
        self.causal_lm = causal_lm
        self.causal_lm.requires_grad_(False)
        self.causal_lm.eval()
        self.causal_lm.config.use_cache = False

    @property
    def hidden_dim(self) -> int:
        return int(self.causal_lm.config.hidden_size)

    def train(self, mode: bool = True):
        del mode
        super().train(False)
        self.causal_lm.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        transformer = getattr(self.causal_lm, "model", None)
        if transformer is not None:
            outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
        else:
            outputs = self.causal_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
        return hidden.detach()
