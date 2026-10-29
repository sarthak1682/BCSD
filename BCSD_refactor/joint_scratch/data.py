from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from shared.data_utils import asm_to_text

_NOVA_LABEL_RE = re.compile(r"<label-\d+>")


@dataclass(frozen=True)
class JointPair:
    function_id: str
    numeric_id: int
    source_asm: str
    target_asm: str
    source_opt: str
    target_opt: str


def _validate_record(record: Dict) -> None:
    missing = {"id", "opt", "asm"}.difference(record)
    if missing:
        raise ValueError(f"sample is missing fields: {sorted(missing)}")
    assembly = asm_to_text(record["asm"])
    if _NOVA_LABEL_RE.search(assembly):
        raise ValueError(
            "joint_scratch expects CLAP-strict/raw bench assembly, but found "
            "Nova-style <label-N> tokens"
        )


def build_variant_pairs(
    samples: Iterable[Dict],
    *,
    seed: int = 42,
) -> List[JointPair]:
    """Use every bench variant once with another same-function variant.

    The target configuration is retained in :class:`JointPair` and placed in
    the LM prompt by the collator, so outputs remain well-defined even when a
    function has variants from many compilers, optimizations, or architectures.
    """

    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for sample in samples:
        _validate_record(sample)
        function_id = str(sample["id"])
        grouped.setdefault(function_id, []).append(
            (str(sample["opt"]), asm_to_text(sample["asm"]))
        )

    rng = random.Random(seed)
    numeric_ids = {
        function_id: index for index, function_id in enumerate(sorted(grouped))
    }
    pairs: List[JointPair] = []
    for function_id in sorted(grouped):
        variants = sorted(grouped[function_id], key=lambda value: (value[0], value[1]))
        if len(variants) < 2:
            continue
        rng.shuffle(variants)
        for index, (source_opt, source_asm) in enumerate(variants):
            target_opt, target_asm = variants[(index + 1) % len(variants)]
            pairs.append(
                JointPair(
                    function_id=function_id,
                    numeric_id=numeric_ids[function_id],
                    source_asm=source_asm,
                    target_asm=target_asm,
                    source_opt=source_opt,
                    target_opt=target_opt,
                )
            )
    rng.shuffle(pairs)
    return pairs


def load_variant_pairs(path: str | Path, *, seed: int = 42) -> List[JointPair]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    def records():
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {source}:{line_number}") from exc

    return build_variant_pairs(records(), seed=seed)


def split_group(function_id: str, unit: str) -> str:
    if unit == "function":
        return function_id
    if unit == "binary":
        return function_id.rsplit("__", 1)[0]
    raise ValueError("split unit must be 'function' or 'binary'")


def split_pairs(
    pairs: Sequence[JointPair],
    *,
    val_fraction: float,
    seed: int,
    unit: str = "binary",
) -> Tuple[List[JointPair], List[JointPair]]:
    """Split whole functions or whole binaries; no pair can cross the boundary."""

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0,1)")
    buckets: Dict[str, List[JointPair]] = {}
    for pair in pairs:
        buckets.setdefault(split_group(pair.function_id, unit), []).append(pair)
    group_ids = sorted(buckets)
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    target_val_pairs = max(1, int(len(pairs) * val_fraction)) if val_fraction else 0
    val_groups = set()
    selected_pairs = 0
    for group_id in group_ids:
        if selected_pairs >= target_val_pairs:
            break
        val_groups.add(group_id)
        selected_pairs += len(buckets[group_id])
    train = [pair for group in group_ids if group not in val_groups for pair in buckets[group]]
    val = [pair for group in group_ids if group in val_groups for pair in buckets[group]]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class JointPairDataset(Dataset):
    def __init__(self, pairs: Sequence[JointPair]) -> None:
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> JointPair:
        return self.pairs[index]


class JointScratchCollator:
    def __init__(
        self,
        nova_tokenizer,
        *,
        max_length: int = 1_024,
        lm_target_fraction: float = 0.5,
        mlm_probability: float = 0.15,
        raw_chars_per_token: int = 8,
    ) -> None:
        if max_length <= 1:
            raise ValueError("max_length must exceed one")
        if not 0.0 < lm_target_fraction < 1.0:
            raise ValueError("lm_target_fraction must be in (0,1)")
        if not 0.0 < mlm_probability <= 1.0:
            raise ValueError("mlm_probability must be in (0,1]")
        if raw_chars_per_token <= 0:
            raise ValueError("raw_chars_per_token must be positive")
        self.tokenizer = nova_tokenizer
        self.max_length = max_length
        self.lm_target_fraction = lm_target_fraction
        self.mlm_probability = mlm_probability
        self.raw_chars_per_token = raw_chars_per_token
        self.pad_id = nova_tokenizer.tokenizer.pad_token_id
        self.mask_id = nova_tokenizer.tokenizer.mask_token_id
        if self.pad_id is None:
            raise ValueError("Nova tokenizer must define pad_token_id")
        if self.mask_id is None:
            raise ValueError("Nova tokenizer must define mask_token_id for MLM")

    @staticmethod
    def format_lm_source(pair: JointPair) -> Tuple[str, str]:
        prefix = (
            f"Translate assembly from {pair.source_opt} to {pair.target_opt}.\n"
            "Source:\n"
        )
        suffix = "\nTarget:\n"
        text = prefix + pair.source_asm + suffix
        char_types = (
            "0" * len(prefix) + "1" * len(pair.source_asm) + "0" * len(suffix)
        )
        return text, char_types

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        clipped = text[:max_chars]
        if "\n" in clipped:
            clipped = clipped.rsplit("\n", 1)[0] or clipped
        return clipped.rstrip() or text[:max_chars]

    def _clip_for_token_budget(self, assembly: str, token_budget: int) -> str:
        text = asm_to_text(assembly)
        max_chars = max(1, token_budget * self.raw_chars_per_token)
        return self._clip_text(text, max_chars)

    def _truncate_lm(
        self,
        input_ids: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_positions = np.flatnonzero(labels != -100)
        if not len(target_positions):
            raise ValueError("LM example contains no target tokens")
        target_start = int(target_positions[0])
        source_ids = input_ids[:target_start]
        target_ids = input_ids[target_start:]
        target_labels = labels[target_start:]
        if len(input_ids) <= self.max_length:
            return input_ids.astype(np.int64), labels.astype(np.int64)

        reserved_target = min(
            len(target_ids), max(1, int(self.max_length * self.lm_target_fraction))
        )
        source_budget = self.max_length - reserved_target
        kept_source = source_ids[:source_budget]
        remaining = self.max_length - len(kept_source)
        kept_target = target_ids[:remaining]
        kept_labels = target_labels[:remaining]
        truncated_ids = np.concatenate((kept_source, kept_target))
        truncated_labels = np.concatenate(
            (np.full(len(kept_source), -100, dtype=np.int64), kept_labels)
        )
        return truncated_ids.astype(np.int64), truncated_labels.astype(np.int64)

    def _encode_lm(self, pair: JointPair) -> Tuple[np.ndarray, np.ndarray]:
        reserved_target = max(1, int(self.max_length * self.lm_target_fraction))
        source_budget = max(1, self.max_length - reserved_target)
        source_asm = self._clip_for_token_budget(pair.source_asm, source_budget)
        target_asm = self._clip_for_token_budget(pair.target_asm, reserved_target)
        clipped_pair = JointPair(
            function_id=pair.function_id,
            numeric_id=pair.numeric_id,
            source_asm=source_asm,
            target_asm=target_asm,
            source_opt=pair.source_opt,
            target_opt=pair.target_opt,
        )
        input_text, input_types = self.format_lm_source(clipped_pair)
        encoded = self.tokenizer.encode(
            input_text,
            target_asm,
            input_types + "1" * len(target_asm),
        )
        return self._truncate_lm(
            np.asarray(encoded["input_ids"]), np.asarray(encoded["labels"])
        )

    def _encode_retrieval(self, assembly: str) -> np.ndarray:
        text = self._clip_for_token_budget(assembly, self.max_length)
        if not text:
            raise ValueError("retrieval assembly must not be empty")
        encoded = self.tokenizer.encode("", text, "1" * len(text))
        ids = np.asarray(encoded["input_ids"][: self.max_length], dtype=np.int64)
        if not len(ids):
            raise ValueError("retrieval example produced no tokens")
        return ids

    def _encode_mlm(self, assembly: str) -> Tuple[np.ndarray, np.ndarray]:
        ids = self._encode_retrieval(assembly)
        labels = np.full(len(ids), -100, dtype=np.int64)
        mask = np.random.random(len(ids)) < self.mlm_probability
        if not mask.any():
            mask[np.random.randint(0, len(ids))] = True
        labels[mask] = ids[mask]
        masked_ids = ids.copy()
        masked_ids[mask] = self.mask_id
        return masked_ids, labels

    def collate_lm(self, batch: Sequence[JointPair]) -> Dict[str, torch.Tensor]:
        rows = [self._encode_lm(pair) for pair in batch]
        length = max(len(ids) for ids, _ in rows)
        input_ids = np.full((len(rows), length), self.pad_id, dtype=np.int64)
        labels = np.full((len(rows), length), -100, dtype=np.int64)
        attention = np.zeros((len(rows), length), dtype=np.bool_)
        for index, (ids, row_labels) in enumerate(rows):
            row_length = len(ids)
            input_ids[index, :row_length] = ids
            labels[index, :row_length] = row_labels
            attention[index, :row_length] = True
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention),
            "labels": torch.from_numpy(labels),
        }

    def collate_retrieval(self, batch: Sequence[JointPair]) -> Dict[str, torch.Tensor]:
        rows: List[np.ndarray] = []
        function_ids: List[int] = []
        for pair in batch:
            rows.append(self._encode_retrieval(pair.source_asm))
            rows.append(self._encode_retrieval(pair.target_asm))
            function_ids.extend((pair.numeric_id, pair.numeric_id))
        length = max(map(len, rows))
        input_ids = np.full((len(rows), length), self.pad_id, dtype=np.int64)
        attention = np.zeros((len(rows), length), dtype=np.bool_)
        for index, ids in enumerate(rows):
            row_length = len(ids)
            input_ids[index, :row_length] = ids
            attention[index, :row_length] = True
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention),
            "pool_mask": torch.from_numpy(attention.copy()),
            "func_ids": torch.tensor(function_ids, dtype=torch.long),
        }

    def collate_mlm(self, batch: Sequence[JointPair]) -> Dict[str, torch.Tensor]:
        rows: List[Tuple[np.ndarray, np.ndarray]] = []
        for pair in batch:
            rows.append(self._encode_mlm(pair.source_asm))
            rows.append(self._encode_mlm(pair.target_asm))
        length = max(len(ids) for ids, _ in rows)
        input_ids = np.full((len(rows), length), self.pad_id, dtype=np.int64)
        labels = np.full((len(rows), length), -100, dtype=np.int64)
        attention = np.zeros((len(rows), length), dtype=np.bool_)
        for index, (ids, row_labels) in enumerate(rows):
            row_length = len(ids)
            input_ids[index, :row_length] = ids
            labels[index, :row_length] = row_labels
            attention[index, :row_length] = True
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention),
            "labels": torch.from_numpy(labels),
        }

    def __call__(self, batch: Sequence[JointPair]) -> Dict:
        result: Dict = {
            "metadata": {
                "function_ids": [pair.function_id for pair in batch],
                "source_opts": [pair.source_opt for pair in batch],
                "target_opts": [pair.target_opt for pair in batch],
            }
        }
        result["lm"] = self.collate_lm(batch)
        result["mlm"] = self.collate_mlm(batch)
        result["retrieval"] = self.collate_retrieval(batch)
        return result
