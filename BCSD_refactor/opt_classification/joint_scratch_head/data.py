"""Manifest-aligned assembly data for the frozen-head opt classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from joint_scratch.data import JointScratchCollator
from joint_scratch.tokenizer import load_nova_tokenizer

from opt_classification.scratch_baseline.data import LABEL_TO_INDEX


def build_collator(*, max_length: int) -> JointScratchCollator:
    _, nova_tokenizer = load_nova_tokenizer()
    return JointScratchCollator(nova_tokenizer, max_length=max_length)


class JointScratchAssemblyDataset(Dataset):
    """Random-access split JSONL whose rows are encoded via the joint collator."""

    def __init__(self, path: str | Path, collator: JointScratchCollator) -> None:
        self.path = Path(path)
        self.collator = collator
        self.offsets = self._build_offsets()
        self._handle = None

    def _build_offsets(self) -> list[int]:
        offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    offsets.append(offset)
                offset += len(line)
        if not offsets:
            raise ValueError(f"Dataset is empty: {self.path}")
        return offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    @property
    def handle(self):
        if self._handle is None:
            self._handle = self.path.open("r", encoding="utf-8")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.handle.seek(self.offsets[index])
        row = json.loads(self.handle.readline())
        label_name = str(row["optimization"])
        ids = self.collator._encode_retrieval(row["asm"])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(LABEL_TO_INDEX[label_name], dtype=torch.long),
            "label_name": label_name,
            "id": str(row["id"]),
            "opt": str(row["opt"]),
        }


def collate_batch(batch: Sequence[dict[str, Any]], *, pad_id: int) -> dict[str, Any]:
    width = max(int(item["input_ids"].numel()) for item in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    ids: list[str] = []
    opts: list[str] = []
    label_names: list[str] = []
    for row, item in enumerate(batch):
        token_ids = item["input_ids"]
        input_ids[row, : token_ids.numel()] = token_ids
        labels[row] = item["label"]
        ids.append(item["id"])
        opts.append(item["opt"])
        label_names.append(item["label_name"])
    return {
        "input_ids": input_ids,
        "key_padding_mask": input_ids == pad_id,
        "labels": labels,
        "ids": ids,
        "opts": opts,
        "label_names": label_names,
    }
