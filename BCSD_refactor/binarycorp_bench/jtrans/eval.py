import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoConfig, BertModel

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../../")))

from binarycorp_bench.eval_bench import (
    build_eval_pairs,
    compute_report,
    load_jsonl,
    print_report_summary,
    set_seed,
)
from shared.profiling import InferenceProfiler

DEFAULT_MODEL = os.path.join(script_dir, "jtrans_weights")


def resolve_test_path():
    candidates = [
        os.path.abspath(os.path.join(script_dir, "../../../nvemb/output_benchset_rebalanced_test_jtrans_strict.jsonl")),
        os.path.abspath(os.path.join(script_dir, "../../../output_benchset_rebalanced_test_jtrans_strict.jsonl")),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


# Adapted from BCSD/models.py.

SPECIAL_TOKENS = ["[SEP]", "[PAD]", "[CLS]", "[MASK]"]
UNKNOWN_TOKEN_ID = 512


class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.config = config
        self.embeddings.position_embeddings = self.embeddings.word_embeddings


def _resolve_tokenizer_dir(model_path, tokenizer_path=None):
    if tokenizer_path is not None:
        candidate = Path(tokenizer_path)
        if (candidate / "vocab.txt").exists():
            return candidate
        raise FileNotFoundError(f"JTrans tokenizer vocab not found under {candidate}")
    model_dir = Path(model_path)
    candidates = [
        model_dir,
        model_dir.parent / "jtrans_tokenizer",
        Path(__file__).resolve().parent.parent.parent.parent / "jtrans_tokenizer",
    ]
    for candidate in candidates:
        if (candidate / "vocab.txt").exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate jTrans tokenizer vocab.txt for model path {model_path}")


def _load_vocab(tokenizer_dir):
    vocab_path = Path(tokenizer_dir) / "vocab.txt"
    vocab_data = vocab_path.read_text(encoding="utf-8").strip().split("\n")
    vocab_data.extend(SPECIAL_TOKENS)
    return defaultdict(
        lambda: UNKNOWN_TOKEN_ID,
        {token: idx for idx, token in enumerate(vocab_data)},
    )


def _jtrans_tokenize(vocab, text, max_length):
    split_line = text.strip().split(" ") if text.strip() else []
    if len(split_line) <= max_length - 3:
        split_line = ["[CLS]"] + split_line + ["[SEP]"]
        attention_mask = [1] * len(split_line) + [0] * (max_length - len(split_line))
        split_line = split_line + (max_length - len(split_line)) * ["[PAD]"]
    else:
        split_line = ["[CLS]"] + split_line[: max_length - 2] + ["[SEP]"]
        attention_mask = [1] * max_length

    input_ids = [vocab[token] for token in split_line]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


class JTransBenchEmbedder:
    def __init__(self, model_path=DEFAULT_MODEL, device="cuda", batch_size=64, max_length=512, tokenizer_path=None):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        print(f"Loading jTrans from {model_path}...")

        tokenizer_dir = _resolve_tokenizer_dir(model_path, tokenizer_path)
        self.vocab = _load_vocab(tokenizer_dir)

        config = AutoConfig.from_pretrained(model_path)
        config.max_position_embeddings = len(self.vocab)
        self.model = BinBertModel.from_pretrained(model_path, config=config).to(device)
        self.model.eval()

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        input_ids_list, attention_masks_list = [], []
        for t in texts:
            asm_text = "\n".join(t) if isinstance(t, list) else t
            tok = _jtrans_tokenize(self.vocab, asm_text, self.max_length)
            input_ids_list.append(tok["input_ids"])
            attention_masks_list.append(tok["attention_mask"])
        return {
            "input_ids": torch.stack(input_ids_list).to(self.device),
            "attention_mask": torch.stack(attention_masks_list).to(self.device),
        }

    @torch.no_grad()
    def encode_batch(self, batch: List[Dict[str, Any]], profiler=None) -> torch.Tensor:
        inputs = self._tokenize([s["asm"] for s in batch])
        if profiler is not None:
            with profiler:
                out = self.model(**inputs)
                embeddings = out.pooler_output
            profiler.total_samples += len(batch)
        else:
            out = self.model(**inputs)
            embeddings = out.pooler_output
        return F.normalize(embeddings.float(), p=2, dim=1)

    @torch.no_grad()
    def encode_samples(self, samples: List[Dict[str, Any]], progress_every=10, profiler=None) -> torch.Tensor:
        all_embeddings = []
        total_batches = (len(samples) + self.batch_size - 1) // self.batch_size
        start_time = time.time()
        for batch_idx, start in enumerate(range(0, len(samples), self.batch_size), start=1):
            batch = samples[start:start + self.batch_size]
            all_embeddings.append(self.encode_batch(batch, profiler=profiler).cpu())
            if progress_every and (batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == total_batches):
                processed = min(start + len(batch), len(samples))
                elapsed = time.time() - start_time
                sps = processed / max(elapsed, 1e-9)
                eta = max(len(samples) - processed, 0) / max(sps, 1e-9)
                print(
                    f"Inference progress: {batch_idx}/{total_batches} batches "
                    f"({processed}/{len(samples)} samples, {sps:.1f} samples/s, ETA {eta / 60:.1f} min)",
                    flush=True,
                )
        return torch.cat(all_embeddings, dim=0)


def extract_jtrans_bench_embeddings(embedder, pairs, progress_every=10):
    ids, query_opts, target_opts = [], [], []
    query_samples, target_samples = [], []
    for pair in pairs:
        ids.append(pair["id"])
        query_opts.append(pair["query"]["opt"])
        target_opts.append(pair["target"]["opt"])
        query_samples.append(pair["query"])
        target_samples.append(pair["target"])

    profiler = InferenceProfiler(embedder.device)

    # Warmup so CUDA init is excluded from the profiler timing.
    print("Warmup pass...")
    _ = embedder.encode_samples(query_samples[:embedder.batch_size], progress_every=0)

    print("Encoding jTrans query embeddings...")
    query_embeddings = embedder.encode_samples(query_samples, progress_every=progress_every, profiler=profiler)
    print("Encoding jTrans target embeddings...")
    target_embeddings = embedder.encode_samples(target_samples, progress_every=progress_every, profiler=profiler)

    return {
        "ids": ids,
        "query_opts": query_opts,
        "target_opts": target_opts,
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
        "stats": profiler.get_stats(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate jTrans on the rebalanced bench split.")
    parser.add_argument("--data", default=None, help="jTrans-strict bench JSONL path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="jTrans model dir (pretrained or finetuned).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embeddings_path", default=os.path.join(script_dir, "embeddings_jtrans_bench.pt"))
    parser.add_argument("--report_path", default=os.path.join(script_dir, "eval_jtrans_bench_full_report.json"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = args.data or resolve_test_path()
    samples = load_jsonl(data_path)
    pairs = build_eval_pairs(samples, seed=args.seed)
    if not pairs:
        raise RuntimeError("No jTrans bench eval pairs were built.")

    embedder = JTransBenchEmbedder(
        model_path=args.model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    result = extract_jtrans_bench_embeddings(embedder, pairs, progress_every=args.progress_every)

    torch.save(result, args.embeddings_path)
    print(f"Saved embeddings to {args.embeddings_path}")

    print("\n" + "=" * 50)
    print(f"JTRANS | Latency: {result['stats']['avg_ms_per_sample']:.2f} ms/function")
    print(f"JTRANS | Memory:  {result['stats']['peak_memory_mb']:.1f} MB peak")
    print("=" * 50)

    report = compute_report(result)
    report["source"] = os.path.abspath(args.embeddings_path)
    report["model"] = args.model
    report["data"] = os.path.abspath(data_path)
    report["latency_ms_per_sample"] = result["stats"]["avg_ms_per_sample"]
    report["peak_memory_mb"] = result["stats"]["peak_memory_mb"]

    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full report to {args.report_path}")

    print_report_summary(report)


if __name__ == "__main__":
    main()
