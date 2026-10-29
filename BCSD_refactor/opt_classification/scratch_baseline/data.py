from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from opt_classification.build_manifest import assert_split_disjointness
from opt_classification.embedders import assert_no_explicit_settings
from opt_classification.extract import load_manifest as load_extract_manifest
from opt_classification.extract import sha256_file
from shared.data_utils import asm_to_text
from shared.nova_utils import setup_nova_tokenizer


CLASS_LABELS = ("O0", "O1", "O2", "O3")
LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


def read_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    rows = load_extract_manifest(str(manifest_path))
    assert_split_disjointness(rows)
    missing = {"id", "opt", "optimization", "split", "binary_group"}
    for row in rows:
        row_missing = missing.difference(row)
        if row_missing:
            raise ValueError(f"Manifest row misses {sorted(row_missing)}: {row}")
        if row["optimization"] not in LABEL_TO_INDEX:
            raise ValueError(f"Unknown optimization label: {row['optimization']!r}")
    return rows


def split_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if str(row["split"]) == split]
        counts[split] = {
            "rows": len(selected),
            "functions": len({str(row["id"]) for row in selected}),
            "binary_groups": len({str(row["binary_group"]) for row in selected}),
        }
    return counts


def _metadata_matches(
    metadata_path: Path,
    *,
    manifest_hash: str,
    train_source: Path,
    test_source: Path,
    counts: Mapping[str, Mapping[str, int]],
) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected = {
        "manifest_sha256": manifest_hash,
        "train_source": str(train_source),
        "test_source": str(test_source),
        "counts": counts,
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def prepare_split_jsonls(
    *,
    manifest_path: str | Path,
    train_source: str | Path,
    test_source: str | Path,
    output_dir: str | Path,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Stream source JSONLs and materialize only manifest-selected rows.

    The cache keeps training lazy without repeatedly scanning the much larger
    Nova sources. Rows are written split-wise under ``output_dir/data``.
    """

    manifest_path = Path(manifest_path)
    train_source = Path(train_source)
    test_source = Path(test_source)
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {split: data_dir / f"{split}.jsonl" for split in ("train", "val", "test")}
    metadata_path = data_dir / "cache_metadata.json"

    manifest_rows = read_manifest(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    counts = split_counts(manifest_rows)
    if (
        not rebuild
        and all(path.is_file() for path in output_paths.values())
        and _metadata_matches(
            metadata_path,
            manifest_hash=manifest_hash,
            train_source=train_source,
            test_source=test_source,
            counts=counts,
        )
    ):
        return {
            "paths": {split: str(path) for split, path in output_paths.items()},
            "metadata_path": str(metadata_path),
            "manifest_sha256": manifest_hash,
            "counts": counts,
            "reused": True,
        }

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest_rows:
        key = (str(row["id"]), str(row["opt"]))
        if key in by_key:
            raise ValueError(f"Duplicate manifest key: {key}")
        by_key[key] = dict(row)

    wanted_by_source = {
        train_source: {
            key
            for key, row in by_key.items()
            if str(row["split"]) in {"train", "val"}
        },
        test_source: {
            key for key, row in by_key.items() if str(row["split"]) == "test"
        },
    }
    found: set[tuple[str, str]] = set()
    temp_paths = {split: path.with_suffix(".jsonl.tmp") for split, path in output_paths.items()}

    handles = {
        split: temp_paths[split].open("w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    try:
        for source_path, wanted in wanted_by_source.items():
            with source_path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    source_row = json.loads(line)
                    key = (str(source_row.get("id")), str(source_row.get("opt")))
                    if key not in wanted:
                        continue
                    if key in found:
                        raise ValueError(
                            f"Duplicate source row for {key} in {source_path}:{line_number}"
                        )
                    asm = source_row.get("asm")
                    if asm is None:
                        raise ValueError(f"{source_path}:{line_number} has no asm field")
                    assert_no_explicit_settings(asm, key[1])
                    manifest_row = by_key[key]
                    split = str(manifest_row["split"])
                    output_row = dict(manifest_row)
                    output_row["asm"] = asm
                    handles[split].write(json.dumps(output_row, sort_keys=True) + "\n")
                    found.add(key)
    finally:
        for handle in handles.values():
            handle.close()

    missing = sorted(set(by_key).difference(found))
    if missing:
        preview = ", ".join(f"{function_id}/{opt}" for function_id, opt in missing[:5])
        raise ValueError(f"Source JSONLs are missing {len(missing)} manifest rows: {preview}")

    for split, temp_path in temp_paths.items():
        temp_path.replace(output_paths[split])

    metadata = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "train_source": str(train_source),
        "test_source": str(test_source),
        "counts": counts,
        "paths": {split: str(path) for split, path in output_paths.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "paths": {split: str(path) for split, path in output_paths.items()},
        "metadata_path": str(metadata_path),
        "manifest_sha256": manifest_hash,
        "counts": counts,
        "reused": False,
    }


class NovaAssemblyDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        max_length: int = 1024,
        pad_id: int = 0,
    ) -> None:
        self.path = Path(path)
        self.max_length = max_length
        self.pad_id = pad_id
        self.offsets = self._build_offsets()
        self._handle = None
        self._tokenizer = None

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
        state["_tokenizer"] = None
        return state

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            _, self._tokenizer, _ = setup_nova_tokenizer()
        return self._tokenizer

    @property
    def handle(self):
        if self._handle is None:
            self._handle = self.path.open("r", encoding="utf-8")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.handle.seek(self.offsets[index])
        row = json.loads(self.handle.readline())
        text = asm_to_text(row["asm"])
        if not text.strip():
            raise ValueError(f"Empty assembly in {self.path}:{index}")
        encoded = self.tokenizer.encode("", text, "1" * len(text))
        raw_input_ids = encoded["input_ids"][: self.max_length]
        input_ids = (
            raw_input_ids.tolist()
            if hasattr(raw_input_ids, "tolist")
            else list(raw_input_ids)
        )
        if len(input_ids) == 0:
            raise ValueError(f"Nova tokenizer produced no input ids in {self.path}:{index}")
        label_name = str(row["optimization"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(LABEL_TO_INDEX[label_name], dtype=torch.long),
            "label_name": label_name,
            "id": str(row["id"]),
            "opt": str(row["opt"]),
        }


def collate_batch(batch: Sequence[Mapping[str, Any]], *, pad_id: int) -> dict[str, Any]:
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
        ids.append(str(item["id"]))
        opts.append(str(item["opt"]))
        label_names.append(str(item["label_name"]))
    return {
        "input_ids": input_ids,
        "key_padding_mask": input_ids == pad_id,
        "labels": labels,
        "ids": ids,
        "opts": opts,
        "label_names": label_names,
    }
