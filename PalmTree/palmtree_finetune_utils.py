import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PALMTREE_ROOT = Path(__file__).resolve().parent
PRETRAINED_DIR = PALMTREE_ROOT / "pre-trained_model"
SRC_DIR = PALMTREE_ROOT / "src"


def _ensure_import_paths() -> None:
    for path in (PRETRAINED_DIR, SRC_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_palmtree_pretrained(
    model_path: Optional[Path] = None,
    vocab_path: Optional[Path] = None,
    device: str = "cpu",
):
    _ensure_import_paths()

    model_path = model_path or (PRETRAINED_DIR / "palmtree" / "transformer.ep19")
    vocab_path = vocab_path or (PRETRAINED_DIR / "palmtree" / "vocab")

    with vocab_path.open("rb") as fh:
        vocab = pickle.load(fh)

    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=device)

    model.to(device)
    return model, vocab


def flatten_function_instructions(record: Dict[str, object]) -> List[str]:
    instructions = record.get("instructions")
    if isinstance(instructions, list) and instructions:
        return [str(item) for item in instructions]

    flattened: List[str] = []
    blocks = record.get("blocks", [])
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, list):
                flattened.extend(str(item) for item in block)
    return flattened


class PalmTreeFunctionEncoder(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        vocab,
        device: str = "cpu",
        seq_len: int = 20,
        instruction_batch_size: int = 256,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling != "mean":
            raise ValueError("Only mean pooling is currently supported.")

        self.model = model
        self.vocab = vocab
        self.device = device
        self.seq_len = seq_len
        self.instruction_batch_size = instruction_batch_size
        self.pooling = pooling
        self.hidden_size = int(getattr(model, "hidden", 128))

    def _encode_instruction_chunk(self, texts: Sequence[str]) -> torch.Tensor:
        sequence_rows: List[List[int]] = []
        segment_rows: List[List[int]] = []

        for text in texts:
            token_ids = [self.vocab.sos_index] + self.vocab.to_seq(text) + [self.vocab.eos_index]
            segment = [1] * len(token_ids)

            if len(token_ids) > self.seq_len:
                token_ids = token_ids[: self.seq_len]
                segment = segment[: self.seq_len]
            else:
                pad_len = self.seq_len - len(token_ids)
                token_ids = token_ids + [self.vocab.pad_index] * pad_len
                segment = segment + [0] * pad_len

            sequence_rows.append(token_ids)
            segment_rows.append(segment)

        sequence = torch.tensor(sequence_rows, dtype=torch.long, device=self.device)
        segment_label = torch.tensor(segment_rows, dtype=torch.long, device=self.device)
        hidden_states = self.model(sequence, segment_label)
        return hidden_states.mean(dim=1)

    def encode_instruction_texts(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.hidden_size), device=self.device)

        outputs: List[torch.Tensor] = []
        for start in range(0, len(texts), self.instruction_batch_size):
            chunk = texts[start : start + self.instruction_batch_size]
            outputs.append(self._encode_instruction_chunk(chunk))
        return torch.cat(outputs, dim=0)

    def forward(self, records: Sequence[Dict[str, object]]) -> torch.Tensor:
        if not records:
            return torch.empty((0, self.hidden_size), device=self.device)

        all_texts: List[str] = []
        spans: List[Tuple[int, int]] = []

        for record in records:
            instructions = flatten_function_instructions(record)
            start = len(all_texts)
            all_texts.extend(instructions)
            spans.append((start, len(instructions)))

        instruction_embeddings = self.encode_instruction_texts(all_texts)
        function_embeddings: List[torch.Tensor] = []

        for start, length in spans:
            if length == 0:
                function_embeddings.append(
                    torch.zeros(self.hidden_size, device=self.device, dtype=instruction_embeddings.dtype)
                )
                continue
            function_embeddings.append(instruction_embeddings[start : start + length].mean(dim=0))

        return torch.stack(function_embeddings, dim=0)


def load_finetuned_palmtree_encoder(
    checkpoint_path: Optional[Path],
    device: str = "cpu",
    instruction_batch_size: int = 256,
):
    model, vocab = load_palmtree_pretrained(device=device)
    encoder = PalmTreeFunctionEncoder(
        model=model,
        vocab=vocab,
        device=device,
        instruction_batch_size=instruction_batch_size,
        pooling="mean",
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.model.load_state_dict(checkpoint["model_state_dict"])
    return encoder


def build_positive_pairs(
    records: Sequence[Dict[str, object]],
    prefer_opt_pair: Tuple[str, str] = ("O0", "O3"),
) -> List[Tuple[Dict[str, object], Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["id"])].append(record)

    pairs: List[Tuple[Dict[str, object], Dict[str, object]]] = []
    preferred_left, preferred_right = prefer_opt_pair

    for variants in grouped.values():
        by_opt: Dict[str, Dict[str, object]] = {}
        for variant in variants:
            by_opt[str(variant.get("opt", "unknown"))] = variant

        if preferred_left in by_opt and preferred_right in by_opt:
            pairs.append((by_opt[preferred_left], by_opt[preferred_right]))
            continue

        sorted_variants = sorted(variants, key=lambda item: str(item.get("opt", "")))
        for i in range(len(sorted_variants)):
            for j in range(i + 1, len(sorted_variants)):
                pairs.append((sorted_variants[i], sorted_variants[j]))

    return pairs


def iter_batches(items: Sequence, batch_size: int) -> Iterator[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def contrastive_loss(anchors: torch.Tensor, positives: torch.Tensor, temperature: float) -> torch.Tensor:
    if anchors.shape[0] != positives.shape[0]:
        raise ValueError("Anchors and positives must have the same batch size.")

    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    logits = anchors @ positives.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def save_palmtree_checkpoint(
    output_path: Path,
    encoder: PalmTreeFunctionEncoder,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": encoder.model.state_dict(),
        "seq_len": encoder.seq_len,
        "pooling": encoder.pooling,
        "instruction_batch_size": encoder.instruction_batch_size,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, output_path)
