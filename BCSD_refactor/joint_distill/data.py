from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from shared.data_utils import asm_to_text
from shared.nova_utils import make_bidirectional_nova_mask


INSTRUCT_TEMPLATE = (
    "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
)


@dataclass(frozen=True)
class JointPair:
    function_id: str
    numeric_id: int
    source_asm: str
    target_asm: str
    source_opt: str
    target_opt: str


def group_variants(samples: Sequence[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for sample in samples:
        missing = {"id", "opt", "asm"}.difference(sample)
        if missing:
            raise ValueError(f"sample is missing fields: {sorted(missing)}")
        grouped.setdefault(str(sample["id"]), []).append(sample)
    return grouped


def split_function_ids(
    samples: Sequence[Dict],
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0,1)")
    grouped = group_variants(samples)
    function_ids = sorted(fid for fid, rows in grouped.items() if len(rows) >= 2)
    random.Random(seed).shuffle(function_ids)
    if not function_ids:
        return [], []
    val_count = max(1, int(len(function_ids) * val_fraction)) if val_fraction else 0
    return function_ids[val_count:], function_ids[:val_count]


def build_joint_pairs(
    samples: Sequence[Dict],
    function_ids: Iterable[str],
    *,
    seed: int,
) -> List[JointPair]:
    grouped = group_variants(samples)
    numeric_ids = {fid: index for index, fid in enumerate(sorted(grouped))}
    rng = random.Random(seed)
    pairs: List[JointPair] = []
    for fid in function_ids:
        variants = list(grouped.get(fid, ()))
        if len(variants) < 2:
            continue
        rng.shuffle(variants)
        for index, source in enumerate(variants):
            target = variants[(index + 1) % len(variants)]
            pairs.append(
                JointPair(
                    function_id=fid,
                    numeric_id=numeric_ids[fid],
                    source_asm=asm_to_text(source["asm"]),
                    target_asm=asm_to_text(target["asm"]),
                    source_opt=str(source["opt"]),
                    target_opt=str(target["opt"]),
                )
            )
    rng.shuffle(pairs)
    return pairs


class JointPairDataset(Dataset):
    def __init__(self, pairs: Sequence[JointPair]) -> None:
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> JointPair:
        return self.pairs[index]


class JointDistillCollator:
    def __init__(
        self,
        nova_tokenizer,
        *,
        max_length: int = 1_024,
        condition_on_target: bool = False,
    ) -> None:
        self.tokenizer = nova_tokenizer
        self.max_length = max_length
        self.condition_on_target = condition_on_target
        self.pad_id = nova_tokenizer.tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Nova tokenizer must define pad_token_id")
        self.label_ids = nova_tokenizer.labels
        prefix = nova_tokenizer.encode(
            "", INSTRUCT_TEMPLATE, "0" * len(INSTRUCT_TEMPLATE)
        )
        self.instruction_token_count = len(prefix["input_ids"])

    def _lm_text(self, pair: JointPair) -> Tuple[str, str]:
        if not self.condition_on_target:
            return pair.source_asm, "1" * len(pair.source_asm)
        prefix = f"Target configuration: {pair.target_opt}\nSource:\n"
        suffix = "\nTarget:\n"
        text = prefix + pair.source_asm + suffix
        char_types = "0" * len(prefix) + "1" * len(pair.source_asm) + "0" * len(suffix)
        return text, char_types

    def _encode_lm(self, pair: JointPair):
        input_text, input_types = self._lm_text(pair)
        encoded = self.tokenizer.encode(
            input_text,
            pair.target_asm,
            input_types + "1" * len(pair.target_asm),
        )
        length = min(len(encoded["input_ids"]), self.max_length)
        return (
            np.asarray(encoded["input_ids"][:length], dtype=np.int64),
            np.asarray(encoded["labels"][:length], dtype=np.int64),
            np.asarray(
                encoded["nova_attention_mask"][:length, :length], dtype=np.bool_
            ),
        )

    def _encode_pred(self, assembly: str, *, query: bool):
        assembly = asm_to_text(assembly)
        if query:
            text = INSTRUCT_TEMPLATE + assembly
            types = "0" * len(INSTRUCT_TEMPLATE) + "1" * len(assembly)
            pool_start = self.instruction_token_count
        else:
            text = assembly
            types = "1" * len(assembly)
            pool_start = 0
        encoded = self.tokenizer.encode("", text, types)
        length = min(len(encoded["input_ids"]), self.max_length)
        if length <= pool_start:
            raise ValueError("max_length leaves no assembly tokens for retrieval")
        ids = np.asarray(encoded["input_ids"][:length], dtype=np.int64)
        native = np.asarray(
            encoded["nova_attention_mask"][:length, :length], dtype=np.bool_
        )
        bidirectional = np.asarray(
            make_bidirectional_nova_mask(native), dtype=np.bool_
        )
        pool_mask = np.zeros(length, dtype=np.bool_)
        pool_mask[pool_start:length] = True
        label_positions = [
            index for index, token_id in enumerate(ids) if token_id in self.label_ids
        ]
        return ids, bidirectional, pool_mask, label_positions

    def collate_lm(self, batch: Sequence[JointPair]):
        lm_rows = [self._encode_lm(pair) for pair in batch]
        lm_length = max(len(row[0]) for row in lm_rows)
        lm_ids = np.full((len(batch), lm_length), self.pad_id, dtype=np.int64)
        lm_labels = np.full((len(batch), lm_length), -100, dtype=np.int64)
        lm_nova = np.zeros((len(batch), lm_length, lm_length), dtype=np.bool_)
        lm_attention = np.zeros((len(batch), lm_length), dtype=np.bool_)
        for index, (ids, labels, native) in enumerate(lm_rows):
            length = len(ids)
            lm_ids[index, :length] = ids
            lm_labels[index, :length] = labels
            lm_nova[index, :length, :length] = native
            lm_attention[index, :length] = True
        return {
            "input_ids": torch.from_numpy(lm_ids),
            "attention_mask": torch.from_numpy(lm_attention),
            "nova_attention_mask": torch.from_numpy(lm_nova),
            "labels": torch.from_numpy(lm_labels),
        }

    def collate_pred(self, batch: Sequence[JointPair]):
        pred_rows = []
        function_ids: List[int] = []
        for pair in batch:
            pred_rows.append(self._encode_pred(pair.source_asm, query=True))
            pred_rows.append(self._encode_pred(pair.target_asm, query=False))
            function_ids.extend((pair.numeric_id, pair.numeric_id))
        pred_length = max(len(row[0]) for row in pred_rows)
        pred_ids = np.full((len(pred_rows), pred_length), self.pad_id, dtype=np.int64)
        pred_nova = np.zeros((len(pred_rows), pred_length, pred_length), dtype=np.bool_)
        pred_attention = np.zeros((len(pred_rows), pred_length), dtype=np.bool_)
        pred_pool = np.zeros((len(pred_rows), pred_length), dtype=np.bool_)
        label_positions = []
        for index, (ids, mask, pool, positions) in enumerate(pred_rows):
            length = len(ids)
            pred_ids[index, :length] = ids
            pred_nova[index, :length, :length] = mask
            pred_attention[index, :length] = True
            pred_pool[index, :length] = pool
            label_positions.append(positions)
        return {
            "input_ids": torch.from_numpy(pred_ids),
            "attention_mask": torch.from_numpy(pred_attention),
            "nova_attention_mask": torch.from_numpy(pred_nova),
            "pool_mask": torch.from_numpy(pred_pool),
            "func_ids": torch.tensor(function_ids, dtype=torch.long),
            "label_positions": label_positions,
        }

    def __call__(self, batch: Sequence[JointPair]):
        return {
            "lm": self.collate_lm(batch),
            "pred": self.collate_pred(batch),
            "metadata": {
                "function_ids": [pair.function_id for pair in batch],
                "source_opts": [pair.source_opt for pair in batch],
                "target_opts": [pair.target_opt for pair in batch],
            },
        }
