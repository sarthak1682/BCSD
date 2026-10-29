from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from shared.data_utils import asm_to_text, parse_bench_opt


INSTRUCT_TEMPLATE = (
    "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
)


def build_bench_pairs(
    samples: Sequence[Dict],
    function_ids: Iterable[str],
    *,
    seed: int,
) -> List[Tuple[str, str, int]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for sample in samples:
        parse_bench_opt(sample["opt"])
        grouped[sample["id"]].append(sample)

    selected = list(function_ids)
    id_to_int = {fid: index for index, fid in enumerate(sorted(grouped))}
    rng = random.Random(seed)
    pairs: List[Tuple[str, str, int]] = []
    for fid in selected:
        variants = grouped[fid]
        if len(variants) < 2:
            continue
        ordered = variants[:]
        rng.shuffle(ordered)
        for index, query in enumerate(ordered):
            positive = ordered[(index + 1) % len(ordered)]
            pairs.append(
                (
                    asm_to_text(query["asm"]),
                    asm_to_text(positive["asm"]),
                    id_to_int[fid],
                )
            )
    rng.shuffle(pairs)
    return pairs


class CausalNovaPairCollator:
    """Tokenize query/positive pairs using Nova's untouched native masks.

    Queries retain the instruction format used by the existing BCSD teacher;
    positives contain assembly only.  ``pool_mask`` is True only for assembly
    tokens, so all pooling variants consume the same positions and exclude the
    query instruction and batch padding.
    """

    def __init__(
        self,
        nova_tokenizer,
        *,
        max_length: int = 1024,
        instruct_template: str = INSTRUCT_TEMPLATE,
        assert_causal: bool = True,
    ) -> None:
        self.tokenizer = nova_tokenizer
        self.max_length = max_length
        self.instruct_template = instruct_template
        self.assert_causal = assert_causal
        self.pad_id = nova_tokenizer.tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Nova's base tokenizer must define pad_token_id")

        prefix = self.tokenizer.encode(
            "",
            instruct_template,
            "0" * len(instruct_template),
        )
        self.instruction_tokens = len(prefix["input_ids"])

    def _encode(self, assembly: str, *, is_query: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assembly = asm_to_text(assembly)
        if not assembly.strip():
            raise ValueError("Cannot encode an empty assembly function")

        if is_query:
            text = self.instruct_template + assembly
            char_types = "0" * len(self.instruct_template) + "1" * len(assembly)
            pool_start = self.instruction_tokens
        else:
            text = assembly
            char_types = "1" * len(assembly)
            pool_start = 0

        encoded = self.tokenizer.encode("", text, char_types)
        length = min(len(encoded["input_ids"]), self.max_length)
        if length <= pool_start:
            raise ValueError(
                "max_length leaves no assembly tokens to pool "
                f"(length={length}, prefix_tokens={pool_start})"
            )

        input_ids = np.asarray(encoded["input_ids"][:length], dtype=np.int64)
        native_mask = np.asarray(
            encoded["nova_attention_mask"][:length, :length], dtype=np.bool_
        )

        # Nova's official native mask is causal even though it contains extra
        # hierarchical summary edges.  Guard against accidentally passing a
        # symmetrized/encoder mask or changing tokenizer semantics unnoticed.
        if self.assert_causal and np.any(np.triu(native_mask, k=1)):
            raise ValueError(
                "Nova native mask contains a future-token attention edge; "
                "refusing to run the decoder-only experiment"
            )

        pool_mask = np.zeros(length, dtype=np.bool_)
        pool_mask[pool_start:length] = True
        return input_ids, native_mask, pool_mask

    def __call__(self, batch: Sequence[Tuple[str, str, int]]) -> Dict[str, torch.Tensor]:
        encoded_rows = []
        function_ids: List[int] = []
        for query, positive, function_id in batch:
            encoded_rows.append(self._encode(query, is_query=True))
            encoded_rows.append(self._encode(positive, is_query=False))
            function_ids.extend((function_id, function_id))

        max_length = max(len(row[0]) for row in encoded_rows)
        batch_size = len(encoded_rows)
        input_ids = np.full((batch_size, max_length), self.pad_id, dtype=np.int64)
        native_masks = np.zeros(
            (batch_size, max_length, max_length), dtype=np.bool_
        )
        attention_mask = np.zeros((batch_size, max_length), dtype=np.bool_)
        pool_mask = np.zeros((batch_size, max_length), dtype=np.bool_)

        for row_index, (ids, native, pool) in enumerate(encoded_rows):
            length = len(ids)
            input_ids[row_index, :length] = ids
            native_masks[row_index, :length, :length] = native
            attention_mask[row_index, :length] = True
            pool_mask[row_index, :length] = pool

        if not np.all(pool_mask.any(axis=1)):
            raise RuntimeError("Every encoded function must have at least one pooling token")

        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "nova_attention_mask": torch.from_numpy(native_masks),
            "pool_mask": torch.from_numpy(pool_mask),
            "func_ids": torch.tensor(function_ids, dtype=torch.long),
        }


def masked_mean_pool(hidden_states: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 3 or pool_mask.ndim != 2:
        raise ValueError("Expected hidden_states [B,L,D] and pool_mask [B,L]")
    if hidden_states.shape[:2] != pool_mask.shape:
        raise ValueError("hidden_states and pool_mask sequence shapes do not match")
    weights = pool_mask.unsqueeze(-1).to(hidden_states.dtype)
    denominator = weights.sum(dim=1).clamp_min(1)
    return (hidden_states * weights).sum(dim=1) / denominator


class LinearMeanProbe(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.projection.weight)

    def forward(self, mean_embedding: torch.Tensor) -> torch.Tensor:
        return self.projection(mean_embedding)


class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[:2] != pool_mask.shape:
            raise ValueError("hidden_states and pool_mask sequence shapes do not match")
        if not torch.all(pool_mask.any(dim=1)):
            raise ValueError("Every sample needs at least one attention-pooling token")
        scores = self.attention(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~pool_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=1).to(hidden_states.dtype)
        return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)


class FrozenCausalNova(nn.Module):
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
        # Heads may enter train mode, but the frozen backbone must retain eval
        # semantics (especially zero dropout) throughout head training.
        super().train(False)
        self.causal_lm.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        nova_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            nova_attention_mask=nova_attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.last_hidden_state.detach()
