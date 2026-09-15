"""Two-stage representation-distillation collation on Nova-format data.

The teacher is the EBM-full Nova model, which consumes Nova-format assembly
(``<label-N>`` summary tokens plus a hierarchical ``nova_attention_mask``).
Both student and teacher therefore tokenize the same Nova text; the student is
a standard transformer that treats ``<label-N>`` as ordinary tokens, while the
teacher uses its hierarchical attention.  Stage 1 builds a causal
source->target translation view with a per-token assembly mask; Stage 2 builds
a token-masked MLM view together with the clean teacher view and a KD mask over
the masked positions.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from shared.data_utils import asm_to_text
from shared.nova_utils import make_bidirectional_nova_mask

from joint_distill.data import JointPair

INSTRUCT_PREFIX_TEMPLATE = (
    "Translate assembly from {source_opt} to {target_opt}.\nSource:\n"
)
INSTRUCT_SUFFIX = "\nTarget:\n"


class RepKDCollator:
    """Collate each pair into a Stage-1 (NTP) and Stage-2 (MLM) view."""

    def __init__(
        self,
        nova_tokenizer,
        *,
        max_length: int = 1_024,
        mlm_probability: float = 0.15,
    ) -> None:
        if max_length <= 1:
            raise ValueError("max_length must exceed one")
        if not 0.0 < mlm_probability <= 1.0:
            raise ValueError("mlm_probability must be in (0, 1]")
        self.tokenizer = nova_tokenizer
        self.base = nova_tokenizer.tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.pad_id = self.base.pad_token_id
        self.mask_id = self.base.mask_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        if self.mask_id is None:
            raise ValueError("tokenizer must define mask_token_id for MLM")

    def _segment_token_count(self, text: str) -> int:
        """Token count NovaTokenizer assigns to one single-type text segment."""
        if not text:
            return 0
        return len(self.base.encode(text)[1:])

    def _encode_stage1(self, pair: JointPair) -> Tuple[np.ndarray, ...]:
        prefix = INSTRUCT_PREFIX_TEMPLATE.format(
            source_opt=pair.source_opt, target_opt=pair.target_opt
        )
        suffix = INSTRUCT_SUFFIX
        input_text = prefix + pair.source_asm + suffix
        input_types = "0" * len(prefix) + "1" * len(pair.source_asm) + "0" * len(suffix)
        output_types = "1" * len(pair.target_asm)
        encoded = self.tokenizer.encode(input_text, pair.target_asm, input_types + output_types)

        length = min(len(encoded["input_ids"]), self.max_length)
        ids = np.asarray(encoded["input_ids"][:length], dtype=np.int64)
        labels = np.asarray(encoded["labels"][:length], dtype=np.int64)
        native = np.asarray(
            encoded["nova_attention_mask"][:length, :length], dtype=np.bool_
        )

        # Assembly positions = source body + target body, excluding the
        # instruction prefix and the "\nTarget:\n" separator.
        prefix_tokens = self._segment_token_count(prefix)
        source_tokens = self._segment_token_count(pair.source_asm)
        separator_tokens = self._segment_token_count(suffix)
        target_start = prefix_tokens + source_tokens + separator_tokens
        assembly_mask = np.zeros(length, dtype=np.bool_)
        assembly_mask[prefix_tokens : prefix_tokens + source_tokens] = True
        assembly_mask[target_start:length] = True
        return ids, labels, native, assembly_mask

    def _encode_stage2(self, assembly: str) -> Tuple[np.ndarray, ...]:
        text = asm_to_text(assembly)
        encoded = self.tokenizer.encode("", text, "1" * len(text))
        length = min(len(encoded["input_ids"]), self.max_length)
        ids = np.asarray(encoded["input_ids"][:length], dtype=np.int64)
        native = np.asarray(
            encoded["nova_attention_mask"][:length, :length], dtype=np.bool_
        )

        labels = np.full(length, -100, dtype=np.int64)
        mask = np.random.random(length) < self.mlm_probability
        if not mask.any():
            mask[np.random.randint(0, length)] = True
        labels[mask] = ids[mask]
        masked_ids = ids.copy()
        masked_ids[mask] = self.mask_id
        bidirectional = np.asarray(
            make_bidirectional_nova_mask(native), dtype=np.bool_
        )
        return masked_ids, ids, labels, mask, bidirectional

    def _collate_stage1(self, batch: Sequence[JointPair]) -> Dict[str, torch.Tensor]:
        rows = [self._encode_stage1(pair) for pair in batch]
        length = max(len(row[0]) for row in rows)
        input_ids = np.full((len(rows), length), self.pad_id, dtype=np.int64)
        labels = np.full((len(rows), length), -100, dtype=np.int64)
        attention = np.zeros((len(rows), length), dtype=np.bool_)
        native = np.zeros((len(rows), length, length), dtype=np.bool_)
        assembly = np.zeros((len(rows), length), dtype=np.bool_)
        for index, (row_ids, row_labels, row_native, row_assembly) in enumerate(rows):
            row_length = len(row_ids)
            input_ids[index, :row_length] = row_ids
            labels[index, :row_length] = row_labels
            attention[index, :row_length] = True
            native[index, :row_length, :row_length] = row_native
            assembly[index, :row_length] = row_assembly
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention),
            "labels": torch.from_numpy(labels),
            "nova_attention_mask": torch.from_numpy(native),
            "assembly_mask": torch.from_numpy(assembly),
        }

    def _collate_stage2(self, batch: Sequence[JointPair]) -> Dict[str, torch.Tensor]:
        rows: List[Tuple[np.ndarray, ...]] = []
        for pair in batch:
            rows.append(self._encode_stage2(pair.source_asm))
            rows.append(self._encode_stage2(pair.target_asm))
        length = max(len(row[0]) for row in rows)
        masked = np.full((len(rows), length), self.pad_id, dtype=np.int64)
        clean = np.full((len(rows), length), self.pad_id, dtype=np.int64)
        labels = np.full((len(rows), length), -100, dtype=np.int64)
        kd_mask = np.zeros((len(rows), length), dtype=np.bool_)
        attention = np.zeros((len(rows), length), dtype=np.bool_)
        bidirectional = np.zeros((len(rows), length, length), dtype=np.bool_)
        for index, (row_masked, row_clean, row_labels, row_mask, row_bidir) in enumerate(rows):
            row_length = len(row_masked)
            masked[index, :row_length] = row_masked
            clean[index, :row_length] = row_clean
            labels[index, :row_length] = row_labels
            kd_mask[index, :row_length] = row_mask
            attention[index, :row_length] = True
            bidirectional[index, :row_length, :row_length] = row_bidir
        return {
            "input_ids": torch.from_numpy(masked),
            "attention_mask": torch.from_numpy(attention),
            "labels": torch.from_numpy(labels),
            "kd_mask": torch.from_numpy(kd_mask),
            "teacher_input_ids": torch.from_numpy(clean),
            "teacher_nova_attention_mask": torch.from_numpy(bidirectional),
        }

    def __call__(self, batch: Sequence[JointPair]) -> Dict:
        return {
            "stage1": self._collate_stage1(batch),
            "stage2": self._collate_stage2(batch),
            "metadata": {
                "function_ids": [pair.function_id for pair in batch],
                "source_opts": [pair.source_opt for pair in batch],
                "target_opts": [pair.target_opt for pair in batch],
            },
        }


class NovaRetrievalCollator:
    """Contrastive retrieval view over Nova-format assembly pairs.

    Encodes each pair's source and target assembly into a symmetric raw-assembly
    view (no instruction prefix) and interleaves them ``[src_0, tgt_0, src_1,
    tgt_1, ...]`` with their numeric function ids, matching the contract expected
    by ``shared.losses.contrastive_loss_positive_aware``.  Tokenization matches
    the frozen-representation probe in ``joint_repkd/evaluate.py``: the full text
    is tokenized and then truncated to ``max_length`` tokens.
    """

    def __init__(self, nova_tokenizer, *, max_length: int = 1_024) -> None:
        if max_length <= 1:
            raise ValueError("max_length must exceed one")
        self.tokenizer = nova_tokenizer
        self.base = nova_tokenizer.tokenizer
        self.max_length = max_length
        self.pad_id = self.base.pad_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer must define pad_token_id")

    def _encode(self, assembly: str) -> np.ndarray:
        text = asm_to_text(assembly)
        encoded = self.tokenizer.encode("", text, "1" * len(text))
        ids = np.asarray(encoded["input_ids"][: self.max_length], dtype=np.int64)
        if not len(ids):
            raise ValueError("retrieval example produced no tokens")
        return ids

    def __call__(self, batch: Sequence[JointPair]) -> Dict[str, torch.Tensor]:
        rows: List[np.ndarray] = []
        function_ids: List[int] = []
        for pair in batch:
            rows.append(self._encode(pair.source_asm))
            rows.append(self._encode(pair.target_asm))
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
