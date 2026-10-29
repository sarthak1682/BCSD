from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CLAP_CHECKPOINT = str(
    REPO_ROOT / "binarycorp_bench" / "clap" / "final_finetuned"
)
DEFAULT_RAW_CLAP_CHECKPOINT = "hustcw/clap-asm"
DEFAULT_JTRANS_CHECKPOINT = str(
    REPO_ROOT.parent / "jTrans" / "models" / "jTrans-finetune"
)
DEFAULT_BENCH_JTRANS_CHECKPOINT = str(
    REPO_ROOT / "binarycorp_bench" / "jtrans" / "jtrans_finetuned"
)
DEFAULT_STUDENT_CHECKPOINT = str(
    REPO_ROOT.parent / "model_checkpoints"
    / "nova_student_bench_contrastive_ft_bs64" / "student_best"
)
DEFAULT_PRE_REFINEMENT_STUDENT_CHECKPOINT = str(
    REPO_ROOT.parent / "model_checkpoints" / "nova_student_bench" / "student_best"
)
DEFAULT_EBM_FULL_CHECKPOINT = str(
    REPO_ROOT / "model_checkpoints" / "nova_ebm_bench" / "s3_final"
)
DEFAULT_EBM_NO_CAUSAL_CHECKPOINT = str(
    REPO_ROOT / "model_checkpoints" / "nova_ebm_s3only" / "s3_final"
)
DEFAULT_EBM_S1_CHECKPOINT = str(
    REPO_ROOT / "model_checkpoints" / "nova_ebm_bench" / "s1_final"
)
DEFAULT_ASM2VEC_CHECKPOINT = str(
    REPO_ROOT / "binarycorp_bench" / "asm2vec" / "asm2vec_weights" / "model.pt"
)
DEFAULT_RAW_NOVA_CHECKPOINT = "lt-asset/nova-1.3b"
DEFAULT_JOINT_SCRATCH_CHECKPOINT = str(
    REPO_ROOT
    / "model_checkpoints"
    / "joint_scratch"
    / "clap_strict_30m_lm_mlm_pretrain10_cl1_trainonly_gpu1_seed42"
    / "final"
)
DEFAULT_L4D_EBM_CHECKPOINT = str(
    REPO_ROOT
    / "model_checkpoints"
    / "llm4decompile_ebm"
    / "20260926_044104"
    / "s3_final"
)
DEFAULT_L4D_EBM_MEAN_CHECKPOINT = DEFAULT_L4D_EBM_CHECKPOINT
DEFAULT_L4D_RAW_CHECKPOINT = "LLM4Binary/llm4decompile-1.3b-v1.5"


@dataclass(frozen=True)
class EmbedderSpec:
    kind: str
    checkpoint: str
    input_format: str
    default_mask: Optional[str] = None
    default_pooling: Optional[str] = None
    role: str = "deployed"


MODEL_REGISTRY: Dict[str, EmbedderSpec] = {
    "clap": EmbedderSpec("clap", DEFAULT_CLAP_CHECKPOINT, "clap"),
    "jtrans": EmbedderSpec("jtrans", DEFAULT_JTRANS_CHECKPOINT, "jtrans"),
    "student": EmbedderSpec("student", DEFAULT_STUDENT_CHECKPOINT, "nova"),
    "ebm-full": EmbedderSpec(
        "nova",
        DEFAULT_EBM_FULL_CHECKPOINT,
        "nova",
        default_mask="bidirectional",
        default_pooling="attention",
    ),
    "ebm-no-causal": EmbedderSpec(
        "nova",
        DEFAULT_EBM_NO_CAUSAL_CHECKPOINT,
        "nova",
        default_mask="bidirectional",
        default_pooling="attention",
    ),
    "asm2vec": EmbedderSpec("asm2vec", DEFAULT_ASM2VEC_CHECKPOINT, "nova"),
    "clap-raw": EmbedderSpec(
        "clap", DEFAULT_RAW_CLAP_CHECKPOINT, "clap", role="control"
    ),
    "jtrans-bench": EmbedderSpec(
        "jtrans", DEFAULT_BENCH_JTRANS_CHECKPOINT, "jtrans", role="control"
    ),
    "student-pre": EmbedderSpec(
        "student",
        DEFAULT_PRE_REFINEMENT_STUDENT_CHECKPOINT,
        "nova",
        role="control",
    ),
    "nova-raw": EmbedderSpec(
        "nova",
        DEFAULT_RAW_NOVA_CHECKPOINT,
        "nova",
        default_mask="native",
        default_pooling="mean",
        role="control",
    ),
    "ebm-s1": EmbedderSpec(
        "nova",
        DEFAULT_EBM_S1_CHECKPOINT,
        "nova",
        default_mask="bidirectional",
        default_pooling="mean",
        role="control",
    ),
    "joint-scratch": EmbedderSpec(
        "joint_scratch",
        DEFAULT_JOINT_SCRATCH_CHECKPOINT,
        "clap",
        default_mask="bidirectional",
        default_pooling="attention",
    ),
    "llm4decompile-ebm": EmbedderSpec(
        "l4d_ebm",
        DEFAULT_L4D_EBM_CHECKPOINT,
        "clap",
        default_mask="native",
        default_pooling="attention",
    ),
    "llm4decompile-raw": EmbedderSpec(
        "l4d_raw",
        DEFAULT_L4D_RAW_CHECKPOINT,
        "clap",
        default_mask="native",
        default_pooling="mean",
        role="control",
    ),
    "llm4decompile-ebm-mean": EmbedderSpec(
        "l4d_ebm",
        DEFAULT_L4D_EBM_MEAN_CHECKPOINT,
        "clap",
        default_mask="native",
        default_pooling="mean",
        role="control",
    ),
}


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY)


_SETTINGS_MARKER = re.compile(
    r"^\s*(?:<settings>|\[settings\]|settings\s*:|optimization\s*:)",
    re.IGNORECASE,
)


def _as_lines(asm: Any) -> List[str]:
    if isinstance(asm, str):
        return asm.splitlines()
    if isinstance(asm, (list, tuple)) and all(isinstance(x, str) for x in asm):
        return list(asm)
    raise TypeError("asm must be a string or a list of instruction strings")


def assert_no_explicit_settings(asm: Any, opt: str) -> None:
    """Reject an explicit compiler/optimization settings block.

    Architecture-specific instructions are legitimate evidence.  What is not
    legitimate is the synthetic final block produced by
    ``bench_setting_block(opt)`` (for example ``o3 gcc x86_64 none``), or a
    marked settings/optimization field.  The check intentionally does not ban
    incidental strings such as a symbol containing ``O3``.
    """

    lines = [line.strip() for line in _as_lines(asm) if line.strip()]
    for line in lines:
        if _SETTINGS_MARKER.match(line):
            raise ValueError(
                "Refusing explicit settings-label leakage in assembly input: "
                f"{line!r}"
            )

    parts = opt.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected bench opt label: {opt}")
    explicit = " ".join(
        (
            parts[1].lower(),
            parts[0].lower(),
            "_".join(parts[2:-1]).lower(),
            parts[-1].lower(),
        )
    )
    if lines and re.sub(r"\s+", " ", lines[-1].lower()) == explicit:
        raise ValueError(
            "Refusing explicit settings-label leakage: assembly ends with the "
            f"settings block for {opt!r}"
        )


def _normalise_asm_text(asm: Any) -> str:
    lines = _as_lines(asm)
    text = "\n".join(lines)
    if not text.strip():
        raise ValueError("Cannot embed empty assembly")
    return text


class SharedEmbedderAdapter:
    requires_full_dataset = False

    def __init__(
        self,
        kind: str,
        checkpoint: str,
        *,
        device: str,
        batch_size: int,
        max_length: int,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        # Importing shared.embedders loads transformers, so keep it in __init__.
        from shared.embedders import CLAPEmbedder, JTransEmbedder

        if kind == "clap":
            self.delegate = CLAPEmbedder(
                model_path=checkpoint,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            )
        elif kind == "jtrans":
            self.delegate = JTransEmbedder(
                model_path=checkpoint,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                tokenizer_path=tokenizer_path,
            )
        else:  # pragma: no cover - guarded by factory
            raise ValueError(f"Unsupported shared embedder kind: {kind}")

    def encode_batch(self, assemblies: Sequence[Any]):
        return self.delegate.encode_batch([{"asm": asm} for asm in assemblies])


class StudentEmbedderAdapter:
    requires_full_dataset = False

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        import torch

        from shared.nova_utils import setup_nova_tokenizer
        from shared.student_model import LatentAttentionLayer, StudentDistillationModule

        checkpoint_dir = Path(checkpoint)
        config_path = checkpoint_dir / "student_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing student config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))

        base_tokenizer, self.tokenizer, _ = setup_nova_tokenizer()
        self.pad_id = base_tokenizer.pad_token_id or 0
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.student = StudentDistillationModule(
            vocab_size=config["vocab_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            pad_id=self.pad_id,
        )
        self.pooling_head = LatentAttentionLayer(hidden_dim=config["hidden_dim"])

        student_path = _resolve_student_file(checkpoint_dir, "student_model.pt")
        head_path = _resolve_student_file(checkpoint_dir, "lal_head.pt")
        self.student.load_state_dict(_torch_load_weights(student_path))
        self.pooling_head.load_state_dict(_torch_load_weights(head_path))

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.student = self.student.to(device=device, dtype=dtype).eval()
        self.pooling_head = self.pooling_head.to(device=device, dtype=dtype).eval()

    def encode_batch(self, assemblies: Sequence[Any]):
        import numpy as np
        import torch
        import torch.nn.functional as functional

        encoded: List[List[int]] = []
        for asm in assemblies:
            text = _normalise_asm_text(asm)
            result = self.tokenizer.encode("", text, "1" * len(text))
            ids = result["input_ids"][: self.max_length]
            if len(ids) == 0:
                raise ValueError("Nova tokenizer produced an empty student input")
            encoded.append(ids)

        width = max(map(len, encoded))
        padded = np.full((len(encoded), width), self.pad_id, dtype=np.int64)
        for row, ids in enumerate(encoded):
            padded[row, : len(ids)] = ids

        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        padding_mask = input_ids == self.pad_id
        with torch.no_grad():
            hidden = self.student(input_ids, key_padding_mask=padding_mask)
            # pool_mask contains padding only: there is no instruction prefix.
            embeddings = self.pooling_head(
                hidden,
                key_padding_mask=padding_mask,
                pool_mask=padding_mask,
            )
        return functional.normalize(embeddings.float(), p=2, dim=1)


def _resolve_student_file(checkpoint: Path, filename: str) -> Path:
    best = checkpoint / filename.replace(".pt", "_best.pt")
    plain = checkpoint / filename
    path = best if best.is_file() else plain
    if not path.is_file():
        raise FileNotFoundError(f"Missing student checkpoint file: {plain} or {best}")
    return path


def _torch_load_weights(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older torch
        return torch.load(path, map_location="cpu")


class NovaEmbedderAdapter:
    requires_full_dataset = False

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str,
        batch_size: int,
        max_length: int,
        mask: str,
        pooling: str,
    ) -> None:
        import torch

        from shared.nova_utils import NOVA_CACHE_DIR, setup_nova_tokenizer

        if mask not in {"native", "bidirectional"}:
            raise ValueError("Nova mask must be 'native' or 'bidirectional'")
        if pooling not in {"mean", "attention", "auto"}:
            raise ValueError("Nova pooling must be 'mean', 'attention', or 'auto'")

        base_tokenizer, self.tokenizer, _ = setup_nova_tokenizer()
        self.pad_id = base_tokenizer.pad_token_id or 0
        self.label_ids = set(self.tokenizer.labels)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.mask = mask

        local_checkpoint = Path(checkpoint).expanduser()
        if checkpoint == DEFAULT_RAW_NOVA_CHECKPOINT:
            load_from = Path(NOVA_CACHE_DIR)
        else:
            load_from = local_checkpoint
        if not load_from.is_dir():
            raise FileNotFoundError(f"Nova checkpoint directory not found: {load_from}")

        # setup_nova_tokenizer put the custom modeling module on sys.path.
        from modeling_nova import NovaForCausalLM

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
        }
        if device.startswith("cuda"):
            index = _cuda_device_index(device)
            kwargs["device_map"] = {"": index}
        self.model = NovaForCausalLM.from_pretrained(str(load_from), **kwargs)
        self.model.resize_token_embeddings(len(base_tokenizer))
        if not device.startswith("cuda"):
            self.model = self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        head_path = load_from / "pooling_head.pt"
        if pooling == "auto":
            pooling = "attention" if head_path.is_file() else "mean"
        if pooling == "attention" and not head_path.is_file():
            raise FileNotFoundError(
                "Attention pooling was requested, but this checkpoint has no "
                f"pooling_head.pt: {load_from}"
            )
        self.pooling = pooling
        self.pooling_head = None
        if pooling == "attention":
            from shared.pooling import AttentionPooling

            head = AttentionPooling(self.model.config.hidden_size)
            head.load_state_dict(_torch_load_weights(head_path))
            model_device = next(self.model.parameters()).device
            self.pooling_head = head.to(device=model_device, dtype=dtype).eval()

    def encode_batch(self, assemblies: Sequence[Any]):
        import numpy as np
        import torch
        import torch.nn.functional as functional

        all_ids: List[List[int]] = []
        all_masks: List[Any] = []
        label_positions: List[List[int]] = []
        for asm in assemblies:
            text = _normalise_asm_text(asm)
            result = self.tokenizer.encode("", text, "1" * len(text))
            ids = result["input_ids"][: self.max_length]
            if len(ids) == 0:
                raise ValueError("Nova tokenizer produced an empty input")
            length = len(ids)
            native = result["nova_attention_mask"][:length, :length]
            nova_mask = (
                np.maximum(native, native.T)
                if self.mask == "bidirectional"
                else native
            )
            positions = [
                index for index, token_id in enumerate(ids) if token_id in self.label_ids
            ]
            if not positions:
                raise ValueError(
                    "Nova input has no label tokens after truncation; fixed "
                    "label-token pooling would be undefined"
                )
            all_ids.append(ids)
            all_masks.append(nova_mask)
            label_positions.append(positions)

        width = max(map(len, all_ids))
        padded_ids = np.full((len(all_ids), width), self.pad_id, dtype=np.int64)
        padded_masks = np.zeros((len(all_ids), width, width), dtype=np.float32)
        for row, (ids, nova_mask) in enumerate(zip(all_ids, all_masks)):
            length = len(ids)
            padded_ids[row, :length] = ids
            padded_masks[row, :length, :length] = nova_mask

        model_device = next(self.model.parameters()).device
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=model_device)
        mask_dtype = (
            torch.bfloat16 if model_device.type == "cuda" else torch.float32
        )
        nova_attention_mask = torch.tensor(
            padded_masks, dtype=mask_dtype, device=model_device
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                nova_attention_mask=nova_attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]
            if self.pooling_head is not None:
                embeddings = self.pooling_head(hidden, label_positions)
            else:
                embeddings = torch.stack(
                    [
                        hidden[row, positions, :].mean(dim=0)
                        for row, positions in enumerate(label_positions)
                    ]
                )
        return functional.normalize(embeddings.float(), p=2, dim=1)


def _cuda_device_index(device: str) -> int:
    if ":" not in device:
        return 0
    try:
        return int(device.rsplit(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Invalid CUDA device: {device!r}") from exc


class JointScratchEmbedderAdapter:
    requires_full_dataset = False

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        import torch

        from joint_scratch.data import JointScratchCollator
        from joint_scratch.model import ScratchJointModel
        from joint_scratch.tokenizer import load_nova_tokenizer

        checkpoint_dir = self._resolve_checkpoint_dir(Path(checkpoint).expanduser())
        _, nova_tokenizer = load_nova_tokenizer()
        self.collator = JointScratchCollator(
            nova_tokenizer,
            max_length=max_length,
            raw_chars_per_token=8,
        )
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = ScratchJointModel.from_pretrained(
            checkpoint_dir, map_location="cpu"
        ).to(device)
        self.model.eval()
        self.use_amp = device.startswith("cuda")

    @staticmethod
    def _resolve_checkpoint_dir(path: Path) -> Path:
        if (path / "model_config.json").is_file() and (path / "model.pt").is_file():
            return path
        for child in ("final", "best"):
            candidate = path / child
            if (candidate / "model_config.json").is_file() and (
                candidate / "model.pt"
            ).is_file():
                return candidate
        raise FileNotFoundError(
            "Joint scratch checkpoint must be a checkpoint dir, or a run dir "
            f"containing final/best: {path}"
        )

    def encode_batch(self, assemblies: Sequence[Any]):
        import numpy as np
        import torch

        rows = [
            self.collator._encode_retrieval(_normalise_asm_text(asm))
            for asm in assemblies
        ]
        width = max(map(len, rows))
        input_ids = np.full((len(rows), width), self.collator.pad_id, dtype=np.int64)
        attention = np.zeros((len(rows), width), dtype=np.bool_)
        for row, ids in enumerate(rows):
            length = len(ids)
            input_ids[row, :length] = ids
            attention[row, :length] = True

        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_tensor = torch.tensor(attention, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.use_amp
                else nullcontext()
            )
            with context:
                output = self.model.forward_retrieval(
                    input_tensor,
                    attention_tensor,
                    attention_tensor,
                    normalize=True,
                )
        return output.embeddings.detach().float().cpu()


class L4DEBMEmbedderAdapter:
    """Attention and mean pooling use the same assembly span."""

    requires_full_dataset = False

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str,
        batch_size: int,
        max_length: int,
        pooling: str = "attention",
    ) -> None:
        import torch

        from binarycorp_bench.llm4decompile_ebm.eval import (
            load_ebm_for_inference,
        )

        if pooling not in {"attention", "mean"}:
            raise ValueError(
                f"L4D-EBM pooling must be 'attention' or 'mean', got {pooling!r}"
            )
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                "The 1.3B LLM4Decompile EBM extractor requires a CUDA device"
            )
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        (
            self.tokenizer,
            self.model,
            attention_head,
            _,
        ) = load_ebm_for_inference(checkpoint, self.device, bidirectional=False)
        self.pooling_head = (
            attention_head if pooling == "attention" else MaskedMeanPoolingHead()
        )

    def encode_batch(self, assemblies: Sequence[Any]):
        from binarycorp_bench.llm4decompile_ebm.eval import encode_assemblies

        return encode_assemblies(
            self.model,
            self.pooling_head,
            self.tokenizer,
            assemblies,
            batch_size=min(self.batch_size, len(assemblies)),
            max_length=self.max_length,
            device=self.device,
            num_workers=0,
            description=f"l4d-ebm opt embeddings ({self.pooling})",
            bidirectional=False,
        )


class MaskedMeanPoolingHead:
    """Implements the pooling-head interface without importing torch at module load."""

    def eval(self) -> "MaskedMeanPoolingHead":
        return self

    def __call__(self, hidden_states, pool_mask):
        from binarycorp_bench.nova_causal_pooling.core import masked_mean_pool

        return masked_mean_pool(hidden_states, pool_mask)


class L4DRawEmbedderAdapter:
    """Use the EBM adapter's prompt and assembly span with the base model and mean pooling."""

    requires_full_dataset = False

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        import torch

        from binarycorp_bench.llm4decompile_causal_pooling.core import (
            load_llm4decompile,
        )
        from binarycorp_bench.llm4decompile_ebm.collators import L4DAssemblyCollator

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                "The 1.3B LLM4Decompile extractor requires a CUDA device"
            )
        self.batch_size = batch_size
        self.max_length = max_length
        # ``eager`` is what FrozenL4DTeacher.load() uses for the KD teacher.
        self.tokenizer, self.model = load_llm4decompile(
            checkpoint, self.device, attn_implementation="eager"
        )
        self.pooling_head = MaskedMeanPoolingHead()
        self.collator = L4DAssemblyCollator(
            self.tokenizer, max_length=self.max_length
        )

    def encode_batch(self, assemblies: Sequence[Any]):
        from torch.utils.data import DataLoader

        from binarycorp_bench.llm4decompile_ebm.eval import encode_loader
        from shared.data_utils import asm_to_text

        texts = [asm_to_text(assembly) for assembly in assemblies]
        loader = DataLoader(
            texts,
            batch_size=min(self.batch_size, len(texts)),
            shuffle=False,
            collate_fn=self.collator,
            num_workers=0,
            pin_memory=True,
        )
        return encode_loader(
            self.model,
            self.pooling_head,
            loader,
            self.device,
            description="l4d-raw opt embeddings",
        )


class Asm2VecEmbedderAdapter:
    requires_full_dataset = True

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str,
        batch_size: int,
        epochs: int = 10,
        temp_root: Optional[str] = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Asm2Vec checkpoint not found: {self.checkpoint}")
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.temp_root = Path(temp_root).expanduser() if temp_root else None

    def encode_all(self, assemblies: Sequence[Any], keys: Sequence[tuple[str, str]]):
        import torch
        import torch.nn.functional as functional

        from asm2vec.utils import load_data, load_model, train

        if len(assemblies) != len(keys):
            raise ValueError("assemblies and keys must have the same length")
        if self.temp_root is not None:
            self.temp_root.mkdir(parents=True, exist_ok=True)

        # TemporaryDirectory owns only its generated child. It never removes the
        # caller's temp root or the evaluator's historical asm2vec_test_temp.
        with tempfile.TemporaryDirectory(
            prefix="opt-classification-asm2vec-",
            dir=str(self.temp_root) if self.temp_root else None,
        ) as temp_name:
            temp_dir = Path(temp_name)
            filename_to_index: Dict[str, int] = {}
            for index, (asm, key) in enumerate(zip(assemblies, keys)):
                converted = _asm_to_asm2vec_text(asm)
                if converted.strip() == "_block_label:":
                    raise ValueError(f"Asm2Vec input is empty for {key!r}")
                digest = hashlib.sha256(
                    f"{index}\0{key[0]}\0{key[1]}".encode("utf-8")
                ).hexdigest()[:16]
                filename = f"row_{index:08d}_{digest}.s"
                (temp_dir / filename).write_text(converted, encoding="utf-8")
                filename_to_index[filename] = index

            model, tokens = load_model(str(self.checkpoint), device=self.device)
            functions, new_tokens = load_data(str(temp_dir), limit=None)
            if len(functions) != len(keys):
                raise RuntimeError(
                    "Asm2Vec parser did not preserve every selected row: "
                    f"expected {len(keys)}, parsed {len(functions)}"
                )
            # Match asm2vec's official test script: grow the token lookup if
            # needed and, critically, initialize one fresh function vector per
            # selected row before optimizing only those function vectors.
            tokens.update(new_tokens)
            model.update(len(functions), tokens.size())
            model = model.to(self.device)
            model = train(
                functions=functions,
                tokens=tokens,
                model=model,
                embedding_size=200,
                batch_size=self.batch_size,
                epochs=self.epochs,
                neg_sample_num=5,
                calc_acc=False,
                device=self.device,
                mode="test",
                learning_rate=0.02,
            ).cpu()

            ordered: List[Optional[Any]] = [None] * len(keys)
            with torch.no_grad():
                for function_index, function in enumerate(functions):
                    raw_name = function.meta.get("name")
                    filename = os.path.basename(str(raw_name)) if raw_name else ""
                    if filename not in filename_to_index:
                        raise RuntimeError(
                            f"Asm2Vec returned an unknown function file: {raw_name!r}"
                        )
                    row_index = filename_to_index[filename]
                    ordered[row_index] = model.embeddings_f(
                        torch.tensor([function_index])
                    ).detach().squeeze(0)
            if any(embedding is None for embedding in ordered):
                raise RuntimeError("Asm2Vec failed to produce every requested embedding")
            stacked = torch.stack(ordered).float()  # type: ignore[arg-type]
            return functional.normalize(stacked, p=2, dim=1)


def _asm_to_asm2vec_text(asm: Any) -> str:
    from binarycorp_bench.asm2vec.common import asm_to_asm2vec_text

    return asm_to_asm2vec_text(_as_lines(asm))


def create_embedder(
    model_name: str,
    *,
    checkpoint: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 1024,
    nova_mask: Optional[str] = None,
    pooling: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    asm2vec_epochs: int = 10,
    temp_root: Optional[str] = None,
):
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {model_name!r}; choices: {', '.join(list_models())}"
        )
    spec = MODEL_REGISTRY[model_name]
    if spec.kind in {"l4d_ebm", "l4d_raw"} and nova_mask not in {None, "native"}:
        raise ValueError(
            "L4D adapters require the default attention setting; "
            "--nova-mask must be native."
        )
    resolved_checkpoint = checkpoint or spec.checkpoint
    if spec.kind in {"clap", "jtrans"}:
        return SharedEmbedderAdapter(
            spec.kind,
            resolved_checkpoint,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            tokenizer_path=tokenizer_path,
        )
    if spec.kind == "student":
        return StudentEmbedderAdapter(
            resolved_checkpoint,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    if spec.kind == "nova":
        return NovaEmbedderAdapter(
            resolved_checkpoint,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            mask=nova_mask or spec.default_mask or "native",
            pooling=pooling or spec.default_pooling or "mean",
        )
    if spec.kind == "asm2vec":
        return Asm2VecEmbedderAdapter(
            resolved_checkpoint,
            device=device,
            batch_size=batch_size,
            epochs=asm2vec_epochs,
            temp_root=temp_root,
        )
    if spec.kind == "joint_scratch":
        return JointScratchEmbedderAdapter(
            resolved_checkpoint,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    if spec.kind == "l4d_ebm":
        return L4DEBMEmbedderAdapter(
            resolved_checkpoint,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            pooling=pooling or spec.default_pooling or "attention",
        )
    if spec.kind == "l4d_raw":
        return L4DRawEmbedderAdapter(
            resolved_checkpoint,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    raise AssertionError(f"Unhandled embedder kind: {spec.kind}")
