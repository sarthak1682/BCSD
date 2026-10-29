from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from binarycorp_bench.llm4decompile_causal_pooling.core import (
    FUNC_HEADER,
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    CausalDecompilePairCollator,
    _encode_without_specials,
)


class L4DTranslationCollator:
    def __init__(self, tokenizer, max_length: int = 1024) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id")

        self.bos_ids: List[int] = []
        if getattr(tokenizer, "bos_token_id", None) is not None:
            self.bos_ids = [int(tokenizer.bos_token_id)]
        self.prompt_prefix_ids = _encode_without_specials(
            tokenizer, PROMPT_PREFIX + FUNC_HEADER
        )
        self.prompt_suffix_ids = _encode_without_specials(tokenizer, PROMPT_SUFFIX)
        self.eos_marker_ids = _encode_without_specials(tokenizer, "\n<EOS>")

    def _encode_pair(
        self, item: Sequence[str]
    ) -> Tuple[List[int], List[int]]:
        if len(item) == 3:
            source, target, target_settings = item
            transition = f"<SEP> {target_settings}\n"
        elif len(item) == 2:
            source, target = item
            transition = "<SEP>\n"
        else:
            raise ValueError("Translation items must contain 2 or 3 strings")

        if not str(source).strip() or not str(target).strip():
            raise ValueError("Translation source and target must be non-empty")

        source_ids = _encode_without_specials(self.tokenizer, str(source).rstrip())
        transition_ids = _encode_without_specials(self.tokenizer, transition)
        target_ids = _encode_without_specials(self.tokenizer, str(target).strip())
        if not source_ids or not target_ids:
            raise ValueError("Tokenizer produced no source or target assembly tokens")

        fixed_prefix = self.bos_ids + self.prompt_prefix_ids
        fixed_middle = self.prompt_suffix_ids + transition_ids
        fixed_size = len(fixed_prefix) + len(fixed_middle) + len(self.eos_marker_ids)
        variable_budget = self.max_length - fixed_size
        if variable_budget < 2:
            raise ValueError(
                f"max_length={self.max_length} leaves no room for source and target "
                f"assembly (fixed prompt tokens={fixed_size})"
            )

        target_take = min(len(target_ids), max(1, variable_budget // 2))
        source_take = min(len(source_ids), max(1, variable_budget - target_take))
        spare = variable_budget - source_take - target_take
        if spare > 0:
            extra_target = min(len(target_ids) - target_take, spare)
            target_take += extra_target
            spare -= extra_target
        if spare > 0:
            source_take += min(len(source_ids) - source_take, spare)

        # Preserve the source setting block under truncation.
        prefix_ids = fixed_prefix + source_ids[-source_take:] + fixed_middle
        supervised_ids = target_ids[:target_take] + self.eos_marker_ids
        input_ids = prefix_ids + supervised_ids
        labels = [-100] * len(prefix_ids) + supervised_ids
        if len(input_ids) > self.max_length:
            raise RuntimeError("Translation token budgeting exceeded max_length")
        return input_ids, labels

    def __call__(self, batch: Sequence[Sequence[str]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Cannot collate an empty translation batch")
        encoded = [self._encode_pair(item) for item in batch]
        width = max(len(ids) for ids, _ in encoded)
        input_ids = torch.full((len(encoded), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(encoded), width), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), width), dtype=torch.bool)
        for row, (ids, row_labels) in enumerate(encoded):
            length = len(ids)
            input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
            labels[row, :length] = torch.tensor(row_labels, dtype=torch.long)
            attention_mask[row, :length] = True
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class L4DAssemblyCollator(CausalDecompilePairCollator):
    def __call__(self, assemblies: Sequence[str]) -> Dict[str, torch.Tensor]:
        if not assemblies:
            raise ValueError("Cannot collate an empty assembly batch")
        encoded = [self._encode(assembly) for assembly in assemblies]
        width = max(len(ids) for ids, _ in encoded)
        input_ids = np.full((len(encoded), width), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(encoded), width), dtype=np.bool_)
        pool_mask = np.zeros((len(encoded), width), dtype=np.bool_)
        for row, (ids, pool) in enumerate(encoded):
            length = len(ids)
            input_ids[row, :length] = ids
            attention_mask[row, :length] = True
            pool_mask[row, :length] = pool
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "pool_mask": torch.from_numpy(pool_mask),
        }


TranslationCollator = L4DTranslationCollator
L4DPairCollator = CausalDecompilePairCollator
PairCollator = CausalDecompilePairCollator


__all__ = [
    "L4DTranslationCollator",
    "L4DPairCollator",
    "L4DAssemblyCollator",
    "TranslationCollator",
    "PairCollator",
]
