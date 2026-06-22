import torch
import torch.nn.functional as F
import numpy as np
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertModel

class InferenceProfiler:
    """Tracks asynchronous CUDA time and peak memory."""
    def __init__(self, device: str):
        self.device = device
        self.is_cuda = 'cuda' in device
        if self.is_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_time_ms = 0.0
        self.peak_memory_mb = 0.0
        self.total_samples = 0

    def __enter__(self):
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_event.record()
        else:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            self.total_time_ms += self.start_event.elapsed_time(self.end_event)
            peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            self.peak_memory_mb = max(self.peak_memory_mb, peak_mem)
        else:
            self.total_time_ms += (time.time() - self.start_time) * 1000

    def get_stats(self) -> Dict[str, float]:
        return {
            "avg_ms_per_sample": self.total_time_ms / max(1, self.total_samples),
            "peak_memory_mb": self.peak_memory_mb
        }

class BaseEmbedder(ABC):
    def __init__(self, device: str = "cuda", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.profiler = InferenceProfiler(device)
        self.model = None

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
        return F.normalize(embeddings, p=2, dim=1)

    def run_inference(self, dataset: List[Dict[str, Any]], progress_every: int = 10) -> Dict[str, Any]:
        if self.model is not None:
            self.model.eval()
            
        all_ids, all_opts, all_embs = [], [], []

        # warmup pass to exclude CUDA init from profiler
        if len(dataset) > 0:
            dummy_batch = dataset[:min(self.batch_size, len(dataset))]
            _ = self.encode_batch(dummy_batch)
            self.profiler.total_time_ms = 0.0
            self.profiler.total_samples = 0

        total_samples = len(dataset)
        total_batches = (total_samples + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        for batch_idx, i in enumerate(range(0, len(dataset), self.batch_size), start=1):
            batch = dataset[i:i + self.batch_size]
            embs = self.encode_batch(batch)
            all_embs.append(embs.cpu())
            all_ids.extend([s['id'] for s in batch])
            all_opts.extend([s.get('opt', 'unknown') for s in batch])

            if progress_every and (batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == total_batches):
                processed = min(i + len(batch), total_samples)
                elapsed = time.time() - start_time
                samples_per_sec = processed / max(elapsed, 1e-9)
                remaining = max(total_samples - processed, 0)
                eta_sec = remaining / max(samples_per_sec, 1e-9)
                print(
                    f"Inference progress: {batch_idx}/{total_batches} batches "
                    f"({processed}/{total_samples} samples, {samples_per_sec:.1f} samples/s, "
                    f"ETA {eta_sec / 60:.1f} min)",
                    flush=True
                )
            
        return {
            "ids": all_ids,
            "opts": all_opts,
            "embeddings": torch.cat(all_embs, dim=0),
            "stats": self.profiler.get_stats()
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
            Path(__file__).resolve().parent / "jTrans" / "models" / "jtrans_tokenizer",
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
    def __init__(self, model_path: str = "hustcw/clap-asm", device: str = "cuda", batch_size: int = 32, max_length: int = 1024):
        super().__init__(device, batch_size)
        self.max_length = max_length
        print(f"Loading CLAP from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.model.eval()

    def prepare_input(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        formatted_batch = []
        for s in batch:
            asm_list = s['asm'].split('\n') if isinstance(s['asm'], str) else s['asm']
            # Limit to 1024 instructions to save tokenization time
            asm_list = asm_list[:self.max_length]
            clap_dict = {str(i): inst for i, inst in enumerate(asm_list, start=1)}
            formatted_batch.append(clap_dict)
            
        # bypass broken tokenizer kwargs — get raw lists, then truncate/pad manually
        raw_inputs = self.tokenizer(formatted_batch)

        final_inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        pad_id = self.tokenizer.pad_token_id or 0

        for i in range(len(batch)):
            i_ids = raw_inputs["input_ids"][i][:self.max_length]
            a_mask = raw_inputs["attention_mask"][i][:self.max_length]
            t_ids = raw_inputs["token_type_ids"][i][:self.max_length]
            
            pad_len = self.max_length - len(i_ids)
            
            final_inputs["input_ids"].append(i_ids + [pad_id] * pad_len)
            final_inputs["attention_mask"].append(a_mask + [0] * pad_len)
            final_inputs["token_type_ids"].append(t_ids + [0] * pad_len)
            
        return {k: torch.tensor(v, dtype=torch.long).to(self.device) for k, v in final_inputs.items()}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**inputs)
        if outputs.ndim == 3:
            return outputs[:, 0, :]
        return outputs

    def pool(self, hidden_states: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return hidden_states


class NovaStudentEmbedder(BaseEmbedder):
    INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "

    def __init__(self, student_model, lal_head, tokenizer, max_length=1024, device="cuda", batch_size=32):
        super().__init__(device, batch_size)
        self.student = student_model.to(device)
        self.lal_head = lal_head.to(device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.tokenizer.pad_token_id or 0
        self.model = self.student
    
    def prepare_input(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        all_ids = []
        for s in batch:
            if s.get('opt', 'O0') == 'O0':
                text = self.INSTRUCT_TEMPLATE + s['asm']
                char_types = "0" * len(self.INSTRUCT_TEMPLATE) + "1" * len(s['asm'])
            else:
                text = s['asm']
                char_types = "1" * len(text)
            
            result = self.tokenizer.encode("", text, char_types)
            all_ids.append(result['input_ids'][:self.max_length])

        max_len = max(len(x) for x in all_ids)
        pad_ids = np.full((len(batch), max_len), self.pad_id, dtype=np.int64)
        for i, ids in enumerate(all_ids):
            pad_ids[i, :len(ids)] = ids

        input_ids = torch.tensor(pad_ids, dtype=torch.long, device=self.device)
        key_padding_mask = (input_ids == self.pad_id)
        return {"input_ids": input_ids, "key_padding_mask": key_padding_mask}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.student(input_ids=inputs["input_ids"], key_padding_mask=inputs["key_padding_mask"])

    def pool(self, hidden_states: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.lal_head(hidden_states, key_padding_mask=inputs.get("key_padding_mask"))
