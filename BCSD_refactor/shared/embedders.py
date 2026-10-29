import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertModel

from shared.profiling import InferenceProfiler
from shared.student_model import LatentAttentionLayer, StudentDistillationModule


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


class JTransEmbedder(BaseEmbedder):
    SPECIAL_TOKENS = ["[SEP]", "[PAD]", "[CLS]", "[MASK]"]
    UNKNOWN_TOKEN_ID = 512

    class BinBertModel(BertModel):
        def __init__(self, config, add_pooling_layer=True):
            super().__init__(config, add_pooling_layer=add_pooling_layer)
            self.config = config
            self.embeddings.position_embeddings = self.embeddings.word_embeddings

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
        tokenizer_path: Optional[str] = None,
    ):
        super().__init__(device, batch_size)
        self.max_length = max_length
        print(f"Loading JTrans from {model_path}...")

        tokenizer_dir = self._resolve_tokenizer_dir(model_path, tokenizer_path)
        self.vocab = self._load_vocab(tokenizer_dir)
        if self.max_length > len(self.vocab):
            raise ValueError(
                f"JTrans max_length={self.max_length} exceeds the tied token/position "
                f"embedding size ({len(self.vocab)}). Use {len(self.vocab)} or lower."
            )
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]

        config = AutoConfig.from_pretrained(model_path)
        config.max_position_embeddings = len(self.vocab)

        self.model = self.BinBertModel.from_pretrained(model_path, config=config).to(device)
        self.model.eval()

    def _resolve_tokenizer_dir(self, model_path: str, tokenizer_path: Optional[str]) -> Path:
        if tokenizer_path is not None:
            candidate = Path(tokenizer_path)
            if (candidate / "vocab.txt").exists():
                return candidate
            raise FileNotFoundError(f"JTrans tokenizer vocab not found under {candidate}")

        model_dir = Path(model_path)
        candidate_dirs = [
            model_dir,
            model_dir.parent / "jtrans_tokenizer",
            Path(__file__).resolve().parent.parent.parent / "jtrans_tokenizer",
        ]
        for candidate in candidate_dirs:
            if (candidate / "vocab.txt").exists():
                return candidate
        raise FileNotFoundError(
            f"Unable to locate jTrans tokenizer vocab.txt for model path {model_path}"
        )

    def _load_vocab(self, tokenizer_dir: Path) -> Dict[str, int]:
        vocab_path = tokenizer_dir / "vocab.txt"
        vocab_data = vocab_path.read_text(encoding="utf-8").strip().split("\n")
        vocab_data.extend(self.SPECIAL_TOKENS)
        return defaultdict(
            lambda: self.UNKNOWN_TOKEN_ID,
            {token: idx for idx, token in enumerate(vocab_data)},
        )

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        split_line = text.strip().split(" ") if text.strip() else []
        if len(split_line) <= self.max_length - 3:
            split_line = ["[CLS]"] + split_line + ["[SEP]"]
            attention_mask = [1] * len(split_line) + [0] * (self.max_length - len(split_line))
            split_line = split_line + (self.max_length - len(split_line)) * ["[PAD]"]
        else:
            split_line = ["[CLS]"] + split_line[: self.max_length - 2] + ["[SEP]"]
            attention_mask = [1] * self.max_length

        input_ids = [self.vocab[token] for token in split_line]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def prepare_input(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids, attention_masks = [], []
        for s in batch:
            asm_text = s["asm"] if isinstance(s["asm"], str) else " ".join(s["asm"])
            tokenized = self._tokenize_text(asm_text)
            input_ids.append(tokenized["input_ids"])
            attention_masks.append(tokenized["attention_mask"])

        return {
            "input_ids": torch.stack(input_ids).to(self.device),
            "attention_mask": torch.stack(attention_masks).to(self.device),
        }

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Any:
        return self.model(**inputs)

    def pool(self, hidden_states: Any, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return hidden_states.pooler_output


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
            formatted_batch.append({str(i): inst for i, inst in enumerate(asm_list, start=1)})

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


class NovaStudentEmbedder(BaseEmbedder):
    """Pooling excludes padding and the query instruction prefix."""

    INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "

    def __init__(
        self,
        student_model: StudentDistillationModule,
        lal_head: LatentAttentionLayer,
        tokenizer,
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


class NovaRawEmbedder(BaseEmbedder):
    """Mean-pool label-token states, falling back to non-padding tokens when no labels exist.

    O0 queries receive the instruction prefix. Attention is bidirectional within blocks."""

    INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "

    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 1024,
        device: str = "cuda",
        batch_size: int = 16,
    ):
        super().__init__(device, batch_size)
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_ids = set(tokenizer.labels)
        self.pad_id = tokenizer.tokenizer.pad_token_id or 0
        self.model_device = next(model.parameters()).device

    def prepare_input(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_ids: List[List[int]] = []
        batch_masks: List[np.ndarray] = []
        label_positions: List[List[int]] = []

        for s in batch:
            asm = s["asm"]
            asm = "\n".join(asm) if isinstance(asm, list) else asm
            if s.get("opt", "O0") == "O0":
                text = self.INSTRUCT_TEMPLATE + asm
                char_types = "0" * len(self.INSTRUCT_TEMPLATE) + "1" * len(asm)
            else:
                text = asm
                char_types = "1" * len(asm)

            result = self.tokenizer.encode("", text, char_types)
            ids = result["input_ids"][: self.max_length]
            raw_mask = result["nova_attention_mask"]
            L = len(ids)
            mask = np.maximum(raw_mask[:L, :L], raw_mask[:L, :L].T)

            batch_ids.append(ids)
            batch_masks.append(mask)
            label_positions.append([j for j, tid in enumerate(ids) if tid in self.label_ids])

        max_len = max(len(x) for x in batch_ids)
        padded_ids = np.full((len(batch), max_len), self.pad_id, dtype=np.int64)
        padded_masks = np.zeros((len(batch), max_len, max_len), dtype=np.float32)
        for j, (ids, mask) in enumerate(zip(batch_ids, batch_masks)):
            L = len(ids)
            padded_ids[j, :L] = ids
            padded_masks[j, :L, :L] = mask

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=self.model_device),
            "nova_attention_mask": torch.tensor(padded_masks, dtype=torch.bfloat16, device=self.model_device),
            "label_positions": label_positions,
        }

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        outputs = self.model(
            input_ids=inputs["input_ids"],
            nova_attention_mask=inputs["nova_attention_mask"],
            output_hidden_states=True,
        )
        return outputs.hidden_states[-1]

    def pool(self, hidden_states: torch.Tensor, inputs: Dict[str, Any]) -> torch.Tensor:
        hidden = hidden_states.float()
        B, L, _ = hidden.shape
        non_pad = inputs["input_ids"] != self.pad_id  # [B, L]
        pooled = []
        for i in range(B):
            positions = [p for p in inputs["label_positions"][i] if p < L]
            if positions:
                vec = hidden[i, positions, :].mean(dim=0)
            else:
                row_mask = non_pad[i]
                if row_mask.any():
                    vec = hidden[i, row_mask, :].mean(dim=0)
                else:
                    vec = hidden[i].mean(dim=0)
            pooled.append(vec)
        return torch.stack(pooled, dim=0)


class QwenEmbedder(BaseEmbedder):
    """Last-token pooling with a four-dimensional bidirectional attention mask."""

    INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-0.5B",
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 1024,
    ):
        super().__init__(device, batch_size)
        self.max_length = max_length
        self.model_id = model_id
        print(f"Loading Qwen from {model_id}...")

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device.split(":")[-1]} if device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        self.model.eval()

    @staticmethod
    def _make_bidirectional_4d_mask(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        """[B, L] → [B, 1, L, L] all-attend mask (0=attend, min_dtype=mask)."""
        B, L = input_ids.shape
        pad_mask_2d = (input_ids == pad_id)
        key_pad = pad_mask_2d.unsqueeze(1).unsqueeze(2).to(torch.bfloat16)
        qry_pad = pad_mask_2d.unsqueeze(1).unsqueeze(3).to(torch.bfloat16)
        return (qry_pad | key_pad) * torch.finfo(torch.bfloat16).min

    def prepare_input(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        for s in batch:
            asm = s["asm"]
            asm_text = "\n".join(asm) if isinstance(asm, list) else asm
            if s.get("opt", "O0") == "O0":
                texts.append(self.INSTRUCT_TEMPLATE + asm_text)
            else:
                texts.append(asm_text)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attn_4d = self._make_bidirectional_4d_mask(input_ids, self.pad_id)
        attn_2d = enc["attention_mask"].bool()

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attn_4d.to(self.device),
            "attn_2d": attn_2d,  # stays on CPU, used for pooling
        }

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        return out.hidden_states[-1]

    def pool(self, hidden_states: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        attn_2d = inputs["attn_2d"].to(hidden_states.device)
        lengths = attn_2d.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
        return hidden_states.gather(1, idx).squeeze(1)
