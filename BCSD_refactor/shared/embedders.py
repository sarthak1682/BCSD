"""Embedder classes for inference + benchmarking.

Ported from BCSD/models.py into the refactored repo.

Key changes vs the original:
  - InferenceProfiler imported from shared.profiling (identical).
  - StudentDistillationModule / LatentAttentionLayer imported from
    shared.student_model (refactored architecture).
  - NovaStudentEmbedder now computes pool_mask (instruction-prefix
    exclusion) to match the refactored LAL forward signature:
      lal_head(hidden, key_padding_mask=..., pool_mask=...)
    The old version had no pool_mask, so the LAL would include
    instruction tokens in the mean pool — now consistent with eval.py.
  - CLAPEmbedder / JTransEmbedder are unchanged from the original.
"""

import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from shared.profiling import InferenceProfiler
from shared.student_model import LatentAttentionLayer, StudentDistillationModule


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    def __init__(self, device: str = "cuda", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.profiler = InferenceProfiler(device)
        self.model = None  # subclasses set this for the eval() call in run_inference

    @abstractmethod
    def prepare_input(self, batch: List[Dict[str, Any]]) -> Any:
        pass

    @abstractmethod
    def forward(self, prepared_inputs: Any) -> Any:
        pass

    @abstractmethod
    def pool(self, hidden_states: Any, prepared_inputs: Any) -> torch.Tensor:
        pass

    @torch.no_grad()
    def encode_batch(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        inputs = self.prepare_input(batch)
        with self.profiler:
            hidden = self.forward(inputs)
            embeddings = self.pool(hidden, inputs)
        self.profiler.total_samples += len(batch)
        return F.normalize(embeddings.float(), p=2, dim=1)

    def run_inference(
        self, dataset: List[Dict[str, Any]], progress_every: int = 10
    ) -> Dict[str, Any]:
        if self.model is not None:
            self.model.eval()

        all_ids, all_opts, all_embs = [], [], []

        # Warmup pass to exclude CUDA init from profiler timing
        if dataset:
            dummy = dataset[: min(self.batch_size, len(dataset))]
            _ = self.encode_batch(dummy)
            self.profiler.total_time_ms = 0.0
            self.profiler.total_samples = 0

        total_samples = len(dataset)
        total_batches = (total_samples + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        for batch_idx, i in enumerate(range(0, total_samples, self.batch_size), start=1):
            batch = dataset[i : i + self.batch_size]
            embs = self.encode_batch(batch)
            all_embs.append(embs.cpu())
            all_ids.extend(s["id"] for s in batch)
            all_opts.extend(s.get("opt", "unknown") for s in batch)

            if progress_every and (
                batch_idx == 1
                or batch_idx % progress_every == 0
                or batch_idx == total_batches
            ):
                processed = min(i + len(batch), total_samples)
                elapsed = time.time() - start_time
                sps = processed / max(elapsed, 1e-9)
                eta = max(total_samples - processed, 0) / max(sps, 1e-9)
                print(
                    f"Inference progress: {batch_idx}/{total_batches} batches "
                    f"({processed}/{total_samples} samples, {sps:.1f} samples/s, "
                    f"ETA {eta / 60:.1f} min)",
                    flush=True,
                )

        return {
            "ids": all_ids,
            "opts": all_opts,
            "embeddings": torch.cat(all_embs, dim=0),
            "stats": self.profiler.get_stats(),
        }


# ---------------------------------------------------------------------------
# JTrans
# ---------------------------------------------------------------------------

class JTransEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 1024,
    ):
        super().__init__(device, batch_size)
        self.max_length = max_length
        print(f"Loading JTrans from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Patch broken config size mismatch in jTrans checkpoint
        config = AutoConfig.from_pretrained(model_path)
        config.max_position_embeddings = 2902
        self.model = AutoModel.from_pretrained(model_path, config=config).to(device)
        self.model.eval()

    def _normalize_asm(self, asm_list: List[str]) -> str:
        """Regex normalizer to reduce OOV explosion from raw GCC -S output."""
        normalized = []
        for inst in asm_list:
            inst = re.sub(r"(#|;).*", "", inst).strip()
            if not inst:
                continue
            inst = re.sub(r"-?0x[0-9a-fA-F]+", "IMM", inst)
            inst = re.sub(r"\.L[A-Za-z0-9_]+", "LBL", inst)
            inst = re.sub(r"\[.*?\]", "[MEM]", inst)
            inst = re.sub(r"\s+", " ", inst)
            normalized.append(inst)
        return "\n".join(normalized)

    def prepare_input(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        for s in batch:
            asm = s["asm"]
            asm_list = asm.split("\n") if isinstance(asm, str) else asm
            texts.append(self._normalize_asm(asm_list))
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**inputs).last_hidden_state

    def pool(self, hidden_states: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return hidden_states[:, 0, :]  # CLS token


# ---------------------------------------------------------------------------
# CLAP
# ---------------------------------------------------------------------------

class CLAPEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_path: str = "hustcw/clap-asm",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 1024,
    ):
        super().__init__(device, batch_size)
        self.max_length = max_length
        print(f"Loading CLAP from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.model.eval()

    def prepare_input(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        formatted_batch = []
        for s in batch:
            asm = s["asm"]
            asm_list = asm.split("\n") if isinstance(asm, str) else asm
            asm_list = asm_list[: self.max_length]
            formatted_batch.append({str(i): inst for i, inst in enumerate(asm_list)})

        # Bypass broken tokenizer kwargs — get raw lists, then truncate/pad manually
        raw = self.tokenizer(formatted_batch)
        pad_id = self.tokenizer.pad_token_id or 0
        final: Dict[str, List] = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

        for i in range(len(batch)):
            i_ids = raw["input_ids"][i][: self.max_length]
            a_mask = raw["attention_mask"][i][: self.max_length]
            t_ids = raw["token_type_ids"][i][: self.max_length]
            pad_len = self.max_length - len(i_ids)
            final["input_ids"].append(i_ids + [pad_id] * pad_len)
            final["attention_mask"].append(a_mask + [0] * pad_len)
            final["token_type_ids"].append(t_ids + [0] * pad_len)

        return {k: torch.tensor(v, dtype=torch.long, device=self.device) for k, v in final.items()}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(**inputs)
        return out[:, 0, :] if out.ndim == 3 else out

    def pool(self, hidden_states: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return hidden_states


# ---------------------------------------------------------------------------
# Nova Student
# ---------------------------------------------------------------------------

class NovaStudentEmbedder(BaseEmbedder):
    """Embedder wrapping the distilled Nova student + LatentAttentionLayer head.

    Differences vs the original BCSD/models.py version:
      - Uses the refactored LatentAttentionLayer from shared.student_model.
        That LAL uses Q=sequence, K/V=latents (pre-LN, masked mean-pool),
        whereas the old version used Q=latents, K/V=sequence (mean latent pool).
      - Computes pool_mask to exclude the instruction prefix from mean pooling,
        matching the behaviour of binarycorp3m/nova_student/eval.py.
    """

    INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "

    def __init__(
        self,
        student_model: StudentDistillationModule,
        lal_head: LatentAttentionLayer,
        tokenizer,          # NovaTokenizer instance
        max_length: int = 1024,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        super().__init__(device, batch_size)
        self.student = student_model.to(device)
        self.lal_head = lal_head.to(device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.tokenizer.pad_token_id or 0
        # For BaseEmbedder.run_inference to call .eval() on the student
        self.model = self.student

        # Precompute instruction token length once for pool_mask building
        self._instruct_token_len = len(
            tokenizer.tokenizer.tokenize(self.INSTRUCT_TEMPLATE)
        )

    def prepare_input(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        all_ids: List[List[int]] = []
        is_query: List[bool] = []

        for s in batch:
            if s.get("opt", "O0") == "O0":
                text = self.INSTRUCT_TEMPLATE + s["asm"]
                char_types = "0" * len(self.INSTRUCT_TEMPLATE) + "1" * len(s["asm"])
                is_query.append(True)
            else:
                text = s["asm"]
                char_types = "1" * len(text)
                is_query.append(False)
            result = self.tokenizer.encode("", text, char_types)
            all_ids.append(result["input_ids"][: self.max_length])

        max_len = max(len(x) for x in all_ids)
        pad_ids = np.full((len(batch), max_len), self.pad_id, dtype=np.int64)
        for j, ids in enumerate(all_ids):
            pad_ids[j, : len(ids)] = ids

        input_ids = torch.tensor(pad_ids, dtype=torch.long, device=self.device)
        key_padding_mask = input_ids == self.pad_id  # [B, S] True = padding

        # pool_mask: exclude padding AND instruction prefix from mean pool
        pool_mask = key_padding_mask.clone()
        for j, query in enumerate(is_query):
            if query:
                excl = min(self._instruct_token_len, len(all_ids[j]))
                pool_mask[j, :excl] = True

        return {
            "input_ids": input_ids,
            "key_padding_mask": key_padding_mask,
            "pool_mask": pool_mask,
        }

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.student(
            input_ids=inputs["input_ids"],
            key_padding_mask=inputs["key_padding_mask"],
        )

    def pool(self, hidden_states: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.lal_head(
            hidden_states,
            key_padding_mask=inputs["key_padding_mask"],
            pool_mask=inputs["pool_mask"],
        )
