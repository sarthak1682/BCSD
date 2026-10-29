from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from shared.data_utils import asm_to_text, parse_bench_opt


Pair = Tuple[str, str, int]


def build_o0_o3_pairs(
    samples: Sequence[Dict],
    function_ids: Iterable[str],
    *,
    seed: int,
    pairs_per_function: int = 1,
) -> Tuple[List[Pair], Dict[str, int]]:
    """Build paper-style O0->O3 positive pairs.

    The original Nova BCSD fine-tuning setup uses O0 functions as queries and
    O3 functions as candidates.  Our local bench has multiple compilers,
    architectures, and obfuscation variants per function, so this builder
    samples O0/O3 variants per function while keeping one positive function per
    row by default.
    """
    if pairs_per_function < 1:
        raise ValueError("pairs_per_function must be positive")

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for sample in samples:
        parse_bench_opt(sample["opt"])
        grouped[sample["id"]].append(sample)

    id_to_int = {fid: index for index, fid in enumerate(sorted(grouped))}
    rng = random.Random(seed)
    selected_ids = list(function_ids)
    pairs: List[Pair] = []
    stats = Counter()

    for fid in selected_ids:
        variants = grouped[fid]
        by_opt: Dict[str, List[Dict]] = defaultdict(list)
        for row in variants:
            by_opt[parse_bench_opt(row["opt"])["optimization"]].append(row)

        if not by_opt["O0"] or not by_opt["O3"]:
            stats["skipped_missing_o0_o3"] += 1
            continue

        stats["functions_with_o0_o3"] += 1
        for _ in range(pairs_per_function):
            query = rng.choice(by_opt["O0"])
            target = rng.choice(by_opt["O3"])
            pairs.append(
                (
                    asm_to_text(query["asm"]),
                    asm_to_text(target["asm"]),
                    id_to_int[fid],
                )
            )

    rng.shuffle(pairs)
    stats["pairs"] = len(pairs)
    stats["functions_seen"] = len(selected_ids)
    return pairs, dict(stats)


def mean_label_pool(hidden_states: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 3 or label_mask.ndim != 2:
        raise ValueError("Expected hidden_states [B,L,D] and label_mask [B,L]")
    if hidden_states.shape[:2] != label_mask.shape:
        raise ValueError("hidden_states and label_mask sequence shapes do not match")
    if not torch.all(label_mask.any(dim=1)):
        raise ValueError("Every sequence must contain at least one Nova label token")

    weights = label_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def bcsd_l2_contrastive_loss(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if query_embeddings.shape != target_embeddings.shape:
        raise ValueError("query and target embedding shapes must match")
    if query_embeddings.ndim != 2:
        raise ValueError("Expected [B,D] query and target embeddings")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    distances = torch.cdist(query_embeddings.float(), target_embeddings.float(), p=2)
    logits = -distances / temperature
    labels = torch.arange(query_embeddings.shape[0], device=query_embeddings.device)
    return F.cross_entropy(logits, labels)


class PaperBCSDCollator:
    def __init__(
        self,
        nova_tokenizer,
        *,
        max_length: int = 2048,
        assert_causal: bool = True,
    ) -> None:
        self.tokenizer = nova_tokenizer
        self.max_length = max_length
        self.assert_causal = assert_causal
        self.label_ids = set(nova_tokenizer.labels)
        self.pad_id = nova_tokenizer.tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Nova's base tokenizer must define pad_token_id")

    def encode_assembly(self, assembly: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        text = asm_to_text(assembly)
        if not text.strip():
            raise ValueError("Cannot encode an empty assembly function")

        encoded = self.tokenizer.encode("", text, "1" * len(text))
        length = min(len(encoded["input_ids"]), self.max_length)
        input_ids = np.asarray(encoded["input_ids"][:length], dtype=np.int64)
        native_mask = np.asarray(
            encoded["nova_attention_mask"][:length, :length], dtype=np.bool_
        )

        if self.assert_causal and np.any(np.triu(native_mask, k=1)):
            raise ValueError(
                "Nova native mask contains a future-token attention edge; "
                "refusing to run the decoder-only BCSD baseline"
            )

        label_mask = np.asarray(
            [token_id in self.label_ids for token_id in input_ids], dtype=np.bool_
        )
        if not label_mask.any():
            raise ValueError(
                "Encoded assembly has no Nova <label-*> instruction-summary tokens"
            )
        return input_ids, native_mask, label_mask

    def batch_encoded(
        self,
        encoded_rows: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> Dict[str, torch.Tensor]:
        max_length = max(len(row[0]) for row in encoded_rows)
        batch_size = len(encoded_rows)
        input_ids = np.full((batch_size, max_length), self.pad_id, dtype=np.int64)
        native_masks = np.zeros(
            (batch_size, max_length, max_length), dtype=np.bool_
        )
        attention_mask = np.zeros((batch_size, max_length), dtype=np.bool_)
        label_mask = np.zeros((batch_size, max_length), dtype=np.bool_)

        for row_index, (ids, native, labels) in enumerate(encoded_rows):
            length = len(ids)
            input_ids[row_index, :length] = ids
            native_masks[row_index, :length, :length] = native
            attention_mask[row_index, :length] = True
            label_mask[row_index, :length] = labels

        if not np.all(label_mask.any(axis=1)):
            raise RuntimeError("Every encoded function must have at least one label token")

        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "nova_attention_mask": torch.from_numpy(native_masks),
            "label_mask": torch.from_numpy(label_mask),
        }

    def __call__(self, batch: Sequence[Pair]) -> Dict[str, torch.Tensor]:
        encoded_rows = []
        function_ids: List[int] = []
        for query, target, function_id in batch:
            encoded_rows.append(self.encode_assembly(query))
            encoded_rows.append(self.encode_assembly(target))
            function_ids.extend((function_id, function_id))

        result = self.batch_encoded(encoded_rows)
        result["func_ids"] = torch.tensor(function_ids, dtype=torch.long)
        return result


class PaperBCSDTextCollator(PaperBCSDCollator):
    def __call__(self, batch: Sequence[str]) -> Dict[str, torch.Tensor]:
        return self.batch_encoded([self.encode_assembly(item) for item in batch])

