"""Extract per-instance embeddings for an optimization-probe manifest.

The cache schema is deliberately small and shared with ``probe.py``::

    {
        "ids": list[str],
        "opts": list[str],
        "embeddings": torch.FloatTensor,  # [N, D], CPU
        "metadata": dict,
    }

Input assembly is looked up by ``(id, opt)`` in model-specific JSONL files.
Optimization/compiler/architecture fields are retained for cache alignment only
and are never passed to a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if __package__:
    from .embedders import (
        MODEL_REGISTRY,
        assert_no_explicit_settings,
        create_embedder,
        list_models,
    )
else:  # Allow help/syntax checks without importing the package's probe module.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from embedders import (  # type: ignore[no-redef]
        MODEL_REGISTRY,
        assert_no_explicit_settings,
        create_embedder,
        list_models,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DATA_ROOT = Path("/home/ra72yeq/projects/NovaXLLM2Vec/nvemb")

DEFAULT_INPUT_JSONLS: Dict[str, Tuple[str, str]] = {
    "nova": (
        str(BENCH_DATA_ROOT / "output_benchset_rebalanced_train_nova.jsonl"),
        str(BENCH_DATA_ROOT / "output_benchset_rebalanced_test_nova.jsonl"),
    ),
    "clap": (
        str(BENCH_DATA_ROOT / "output_benchset_rebalanced_train_clap_strict.jsonl"),
        str(BENCH_DATA_ROOT / "output_benchset_rebalanced_test_clap_strict.jsonl"),
    ),
    "jtrans": (
        str(
            BENCH_DATA_ROOT
            / "output_benchset_rebalanced_train_jtrans_strict.jsonl"
        ),
        str(
            BENCH_DATA_ROOT
            / "output_benchset_rebalanced_test_jtrans_strict.jsonl"
        ),
    ),
}

DEFAULT_BATCH_SIZES = {
    "clap": 128,
    "jtrans": 64,
    "student": 32,
    "nova": 16,
    "asm2vec": 2048,
    "joint_scratch": 64,
}
DEFAULT_MAX_LENGTHS = {
    "clap": 256,
    "jtrans": 512,
    "student": 1024,
    "nova": 1024,
    "asm2vec": 1024,  # recorded for provenance; asm2vec parses complete functions
    "joint_scratch": 512,
}

Key = Tuple[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract assembly-only per-(id,opt) embeddings for an optimization "
            "classification manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Manifest JSONL path.")
    parser.add_argument("--model", required=True, choices=list_models())
    parser.add_argument("--output", required=True, help="Output torch .pt cache.")
    parser.add_argument(
        "--input-jsonl",
        action="append",
        default=None,
        help=(
            "Model-format source JSONL. Repeat for train/test files. When omitted, "
            "the registry's Nova, CLAP-strict, or jTrans-strict defaults are used."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override the registered checkpoint/model ID.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or a concrete device such as 'cuda:1'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (model-specific deployed default when omitted).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum tokens/instructions (model-specific default when omitted).",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val", "test"),
        help="Extract only this manifest split; repeat to select several.",
    )
    parser.add_argument(
        "--nova-mask",
        choices=("native", "bidirectional"),
        default=None,
        help="Override the registered Nova mask contract.",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "attention", "auto"),
        default=None,
        help=(
            "Override Nova pooling. Attention is refused unless pooling_head.pt "
            "exists in the checkpoint."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional jTrans tokenizer directory containing vocab.txt.",
    )
    parser.add_argument("--asm2vec-epochs", type=int, default=10)
    parser.add_argument(
        "--temp-root",
        default=None,
        help="Parent for a dedicated asm2vec temporary child directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help=(
            "Checkpoint resumable cache every N ordinary model batches. "
            "Set 0 to save only the final cache; this is much faster for "
            "large embedding dimensions."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an exactly matching partial cache. Existing output is otherwise "
            "never overwritten."
        ),
    )
    parser.add_argument(
        "--provenance-mode",
        choices=("hash", "stat"),
        default="hash",
        help=(
            "Use full SHA-256 content hashes for provenance, or faster size/mtime "
            "fingerprints. Use 'stat' for large iterative extraction runs."
        ),
    )
    return parser


def load_manifest(path: str, splits: Optional[Sequence[str]] = None) -> List[dict]:
    selected_splits = set(splits or ())
    rows: List[dict] = []
    seen: set[Key] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            missing = {"id", "opt", "split"} - row.keys()
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing manifest fields {sorted(missing)}"
                )
            if selected_splits and row["split"] not in selected_splits:
                continue
            key = (str(row["id"]), str(row["opt"]))
            if key in seen:
                raise ValueError(f"Duplicate manifest key {key!r}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("No manifest rows selected")
    return rows


def load_assembly_index(
    paths: Sequence[str], requested_keys: Optional[set[Key]] = None
) -> Dict[Key, Any]:
    """Load JSONL assembly keyed by ``(id,opt)`` and reject conflicts."""

    index: Dict[Key, Any] = {}
    origins: Dict[Key, str] = {}
    for path in paths:
        print(f"Scanning assembly JSONL: {path}", flush=True)
        matched_before = len(index)
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                missing = {"id", "opt", "asm"} - row.keys()
                if missing:
                    raise ValueError(
                        f"{path}:{line_number}: missing source fields {sorted(missing)}"
                    )
                key = (str(row["id"]), str(row["opt"]))
                if line_number % 50_000 == 0:
                    print(
                        f"  scanned {line_number:,} lines; matched "
                        f"{len(index):,} requested assemblies",
                        flush=True,
                    )
                if requested_keys is not None and key not in requested_keys:
                    continue
                assembly = row["asm"]
                if key in index and index[key] != assembly:
                    raise ValueError(
                        f"Conflicting assembly for {key!r} in {origins[key]} and "
                        f"{path}:{line_number}"
                    )
                index[key] = assembly
                origins[key] = f"{path}:{line_number}"
        print(
            f"Finished {path}; matched {len(index) - matched_before:,} new "
            f"requested assemblies ({len(index):,} total)",
            flush=True,
        )
    return index


def select_assemblies(
    manifest_rows: Sequence[Mapping[str, Any]],
    assembly_index: Mapping[Key, Any],
) -> Tuple[List[Key], List[Any]]:
    """Align rows and enforce the no-explicit-settings leakage contract."""

    keys: List[Key] = []
    assemblies: List[Any] = []
    missing: List[Key] = []
    for row in manifest_rows:
        key = (str(row["id"]), str(row["opt"]))
        if key not in assembly_index:
            missing.append(key)
            continue
        assembly = assembly_index[key]
        assert_no_explicit_settings(assembly, key[1])
        keys.append(key)
        assemblies.append(assembly)
    if missing:
        examples = ", ".join(repr(key) for key in missing[:5])
        raise KeyError(
            f"{len(missing)} manifest rows are absent from model input JSONL; "
            f"examples: {examples}"
        )
    return keys, assemblies


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size} B"


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
    *,
    label: Optional[str] = None,
) -> str:
    if label is not None:
        print(
            f"Hashing {label}: {path} ({_format_bytes(path.stat().st_size)})",
            flush=True,
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    value = digest.hexdigest()
    if label is not None:
        print(f"Finished hashing {label}: {value[:12]}...", flush=True)
    return value


def _stat_digest(path: Path, *, root: Optional[Path] = None) -> str:
    stat = path.stat()
    name = path.relative_to(root).as_posix() if root is not None else str(path.resolve())
    payload = {
        "path": name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_fingerprint(
    path: Path,
    *,
    mode: str = "hash",
    label: Optional[str] = None,
) -> Tuple[str, str]:
    if mode == "hash":
        return sha256_file(path, label=label), "sha256-file"
    if mode == "stat":
        if label is not None:
            print(f"Fingerprinting {label} with size/mtime: {path}", flush=True)
        return _stat_digest(path), "stat-file"
    raise ValueError(f"Unknown provenance mode: {mode!r}")


def checkpoint_hash(
    checkpoint: str,
    *,
    mode: str = "hash",
    verbose: bool = False,
) -> Tuple[str, str]:
    """Return a content hash for local artifacts, or an identifier hash."""

    path = Path(checkpoint).expanduser()
    if path.is_file():
        return file_fingerprint(
            path, mode=mode, label="checkpoint file" if verbose else None
        )
    if path.is_dir():
        digest = hashlib.sha256()
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            raise FileNotFoundError(f"Checkpoint directory is empty: {path}")
        if verbose:
            print(
                f"Hashing checkpoint directory: {path} ({len(files)} files)",
                flush=True,
            )
        for item in files:
            relative = item.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            if mode == "hash":
                item_digest = sha256_file(
                    item,
                    label=f"checkpoint/{item.relative_to(path).as_posix()}"
                    if verbose
                    else None,
                )
                kind = "sha256-directory-contents"
            elif mode == "stat":
                if verbose:
                    print(
                        f"Fingerprinting checkpoint/{item.relative_to(path).as_posix()} "
                        "with size/mtime",
                        flush=True,
                    )
                item_digest = _stat_digest(item, root=path)
                kind = "stat-directory-contents"
            else:
                raise ValueError(f"Unknown provenance mode: {mode!r}")
            digest.update(bytes.fromhex(item_digest))
        value = digest.hexdigest()
        if verbose:
            print(f"Finished checkpoint directory provenance: {value[:12]}...", flush=True)
        return value, kind

    # Remote Hugging Face IDs cannot be mapped to immutable bytes before model
    # loading. Record this weaker provenance explicitly rather than pretending.
    digest = hashlib.sha256(f"model-id:{checkpoint}".encode("utf-8")).hexdigest()
    return digest, "sha256-model-identifier"


def _selected_rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {"id": str(row["id"]), "opt": str(row["opt"]), "split": str(row["split"])}
        for row in rows
    ]
    serialised = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _signature(metadata: Mapping[str, Any]) -> str:
    semantic = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "created_utc",
            "updated_utc",
            "completed",
            "num_embeddings",
            "run_signature",
        }
    }
    encoded = json.dumps(
        semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_metadata(
    *,
    args: argparse.Namespace,
    checkpoint: str,
    input_paths: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> dict:
    spec = MODEL_REGISTRY[args.model]
    resolved_mask = (
        args.nova_mask or spec.default_mask or "native"
        if spec.kind in {"nova", "joint_scratch"}
        else None
    )
    resolved_pooling = (
        args.pooling or spec.default_pooling or "mean"
        if spec.kind in {"nova", "joint_scratch"}
        else None
    )
    if resolved_pooling == "auto":
        candidate = Path(checkpoint).expanduser() / "pooling_head.pt"
        resolved_pooling = "attention" if candidate.is_file() else "mean"

    if args.provenance_mode == "hash":
        print("Using full SHA-256 provenance mode.", flush=True)
    else:
        print("Recording fast stat-based provenance fingerprints...", flush=True)
    checkpoint_digest, checkpoint_hash_kind = checkpoint_hash(
        checkpoint, mode=args.provenance_mode, verbose=True
    )
    inputs = []
    for path in input_paths:
        path_obj = Path(path)
        digest, kind = file_fingerprint(
            path_obj,
            mode=args.provenance_mode,
            label=f"input JSONL {path_obj.name}",
        )
        inputs.append(
            {
                "path": str(path_obj.resolve()),
                "sha256": digest,
                "sha256_kind": kind,
            }
        )
    manifest_path = Path(args.manifest)
    manifest_digest = sha256_file(manifest_path, label="manifest")
    print("Hashing selected row identity...", flush=True)
    selected_rows_digest = _selected_rows_hash(rows)
    print(f"Finished selected row identity: {selected_rows_digest[:12]}...", flush=True)
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 1,
        "model": args.model,
        "adapter_kind": spec.kind,
        "role": spec.role,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_hash_kind": checkpoint_hash_kind,
        "provenance_mode": args.provenance_mode,
        "input_files": inputs,
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_digest,
        "selected_rows_sha256": selected_rows_digest,
        "splits": sorted(set(str(row["split"]) for row in rows)),
        "mask": resolved_mask,
        "pooling": resolved_pooling,
        "preprocessing": {
            "assembly_only": True,
            "symmetric": True,
            "instruction_prefix": False,
            "explicit_settings_refused": True,
            "input_format": spec.input_format,
        },
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "asm2vec_epochs": args.asm2vec_epochs if spec.kind == "asm2vec" else None,
        "created_utc": now,
        "updated_utc": now,
        "completed": False,
        "num_embeddings": 0,
    }
    metadata["run_signature"] = _signature(metadata)
    return metadata


def _load_torch_cache(path: Path) -> dict:
    import torch

    try:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict):
        raise ValueError(f"Embedding cache must be a dict: {path}")
    required = {"ids", "opts", "embeddings", "metadata"}
    if set(cache) != required:
        raise ValueError(
            f"Embedding cache keys must be exactly {sorted(required)}, got "
            f"{sorted(cache)}"
        )
    ids, opts, embeddings = cache["ids"], cache["opts"], cache["embeddings"]
    if not isinstance(ids, list) or not isinstance(opts, list):
        raise TypeError("Cache ids and opts must be lists")
    if not isinstance(cache["metadata"], dict):
        raise TypeError("Cache metadata must be a dict")
    if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
        raise TypeError("Cache embeddings must be a rank-2 torch tensor")
    if len(ids) != len(opts) or len(ids) != embeddings.shape[0]:
        raise ValueError("Cache ids/opts/embeddings lengths disagree")
    if not embeddings.dtype.is_floating_point:
        raise TypeError("Cache embeddings tensor must have floating dtype")
    if len(set(zip(ids, opts))) != len(ids):
        raise ValueError("Cache contains duplicate (id,opt) keys")
    return cache


def _atomic_save(path: Path, cache: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(cache), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cache_from_embeddings(
    keys: Sequence[Key],
    embeddings_by_key: Mapping[Key, Any],
    metadata: Mapping[str, Any],
    *,
    completed: bool,
) -> dict:
    import torch

    present = [key for key in keys if key in embeddings_by_key]
    if present:
        embeddings = torch.stack(
            [embeddings_by_key[key].detach().float().cpu() for key in present]
        )
    else:
        embeddings = torch.empty((0, 0), dtype=torch.float32)
    updated_metadata = dict(metadata)
    updated_metadata["updated_utc"] = datetime.now(timezone.utc).isoformat()
    updated_metadata["completed"] = completed
    updated_metadata["num_embeddings"] = len(present)
    return {
        "ids": [key[0] for key in present],
        "opts": [key[1] for key in present],
        "embeddings": embeddings.float(),
        "metadata": updated_metadata,
    }


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return requested


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_extraction(args: argparse.Namespace) -> Path:
    import torch

    spec = MODEL_REGISTRY[args.model]
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZES[spec.kind]
    if args.max_length is None:
        args.max_length = DEFAULT_MAX_LENGTHS[spec.kind]
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("--batch-size and --max-length must be positive")
    if args.save_every < 0:
        raise ValueError("--save-every must be non-negative")
    if args.asm2vec_epochs <= 0:
        raise ValueError("--asm2vec-epochs must be positive")

    output_path = Path(args.output)
    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite existing cache: {output_path}. "
            "Use --resume only for an exactly matching partial cache."
        )
    if args.resume and not output_path.exists():
        raise FileNotFoundError(f"Cannot resume missing cache: {output_path}")

    checkpoint = args.checkpoint or spec.checkpoint
    input_paths = list(
        args.input_jsonl
        if args.input_jsonl is not None
        else DEFAULT_INPUT_JSONLS[spec.input_format]
    )
    absent_inputs = [path for path in input_paths if not Path(path).is_file()]
    if absent_inputs:
        raise FileNotFoundError(
            "Missing model input JSONL:\n  " + "\n  ".join(absent_inputs)
        )

    print(f"Loading manifest: {args.manifest}", flush=True)
    rows = load_manifest(args.manifest, args.split)
    print(f"Loaded {len(rows):,} manifest rows", flush=True)
    requested_keys = {(str(row["id"]), str(row["opt"])) for row in rows}
    assembly_index = load_assembly_index(input_paths, requested_keys=requested_keys)
    print(
        f"Indexed {len(assembly_index):,}/{len(requested_keys):,} requested assemblies",
        flush=True,
    )
    print("Aligning assemblies and checking setting leakage...", flush=True)
    keys, assemblies = select_assemblies(rows, assembly_index)
    metadata = build_metadata(
        args=args,
        checkpoint=checkpoint,
        input_paths=input_paths,
        rows=rows,
    )
    print("Provenance metadata ready", flush=True)

    embeddings_by_key: Dict[Key, Any] = {}
    if args.resume:
        existing = _load_torch_cache(output_path)
        old_signature = existing["metadata"].get("run_signature")
        if old_signature != metadata["run_signature"]:
            raise ValueError(
                "Refusing to resume: cache provenance/configuration does not match "
                "this extraction request"
            )
        metadata["created_utc"] = existing["metadata"].get(
            "created_utc", metadata["created_utc"]
        )
        requested = set(keys)
        for index, key in enumerate(zip(existing["ids"], existing["opts"])):
            if key not in requested:
                raise ValueError(f"Resume cache contains an unrequested key: {key!r}")
            embeddings_by_key[key] = existing["embeddings"][index].float().cpu()
        if existing["metadata"].get("completed") and len(embeddings_by_key) == len(keys):
            print(f"Cache is already complete: {output_path}")
            return output_path

    missing_indices = [
        index for index, key in enumerate(keys) if key not in embeddings_by_key
    ]
    if not missing_indices:
        final_cache = _cache_from_embeddings(
            keys, embeddings_by_key, metadata, completed=True
        )
        _atomic_save(output_path, final_cache)
        return output_path

    device = _resolve_device(args.device)
    _seed_everything(args.seed)
    print(
        f"Loading {args.model} on {device}; extracting "
        f"{len(missing_indices)}/{len(keys)} rows...",
        flush=True,
    )
    embedder = create_embedder(
        args.model,
        checkpoint=checkpoint,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        nova_mask=metadata["mask"],
        pooling=metadata["pooling"],
        tokenizer_path=args.tokenizer_path,
        asm2vec_epochs=args.asm2vec_epochs,
        temp_root=args.temp_root,
    )

    if embedder.requires_full_dataset:
        selected_keys = [keys[index] for index in missing_indices]
        selected_assemblies = [assemblies[index] for index in missing_indices]
        produced = embedder.encode_all(selected_assemblies, selected_keys)
        if produced.ndim != 2 or produced.shape[0] != len(selected_keys):
            raise RuntimeError("Asm2Vec adapter returned an invalid embedding shape")
        for row, key in enumerate(selected_keys):
            embeddings_by_key[key] = produced[row].float().cpu()
    else:
        total_batches = (len(missing_indices) + args.batch_size - 1) // args.batch_size
        for batch_number, start in enumerate(
            range(0, len(missing_indices), args.batch_size), start=1
        ):
            indices = missing_indices[start : start + args.batch_size]
            batch_assemblies = [assemblies[index] for index in indices]
            produced = embedder.encode_batch(batch_assemblies)
            if (
                not torch.is_tensor(produced)
                or produced.ndim != 2
                or produced.shape[0] != len(indices)
            ):
                raise RuntimeError(
                    f"Adapter returned invalid shape for batch {batch_number}: "
                    f"{getattr(produced, 'shape', None)}"
                )
            for row, index in enumerate(indices):
                embeddings_by_key[keys[index]] = produced[row].float().cpu()

            should_report = (
                batch_number == 1
                or batch_number == total_batches
                or (
                    args.progress_every > 0
                    and batch_number % args.progress_every == 0
                )
            )
            if should_report:
                print(
                    f"Extraction progress: {batch_number}/{total_batches} batches "
                    f"({len(embeddings_by_key)}/{len(keys)} cached)",
                    flush=True,
                )
            if args.save_every > 0 and batch_number % args.save_every == 0:
                print(
                    f"Saving resumable partial cache after batch "
                    f"{batch_number}/{total_batches}...",
                    flush=True,
                )
                save_start = time.perf_counter()
                partial = _cache_from_embeddings(
                    keys, embeddings_by_key, metadata, completed=False
                )
                _atomic_save(output_path, partial)
                print(
                    f"Partial cache saved in {time.perf_counter() - save_start:.1f}s",
                    flush=True,
                )

    print("Saving final embedding cache...", flush=True)
    save_start = time.perf_counter()
    final_cache = _cache_from_embeddings(
        keys, embeddings_by_key, metadata, completed=True
    )
    if final_cache["embeddings"].shape[0] != len(keys):
        raise RuntimeError("Extraction completed without every requested embedding")
    _atomic_save(output_path, final_cache)
    save_seconds = time.perf_counter() - save_start
    print(
        f"Saved {len(keys)} float embeddings with shape "
        f"{tuple(final_cache['embeddings'].shape)} to {output_path} "
        f"in {save_seconds:.1f}s",
        flush=True,
    )
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_extraction(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
