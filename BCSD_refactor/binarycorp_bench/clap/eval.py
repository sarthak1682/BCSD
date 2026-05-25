import argparse
import json
import os
import random
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Setup path to import evaluation library
script_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../")))

from eval_bench import set_seed, load_jsonl, build_eval_pairs, compute_report, print_report_summary

MODEL_ID = "hustcw/clap-asm"

repo_root = os.path.abspath(os.path.join(script_dir, "../../../"))
DEFAULT_TEST_PATH = os.path.join(repo_root, "nvemb", "output_benchset_rebalanced_test_clap_strict.jsonl")
if not os.path.exists(DEFAULT_TEST_PATH):
    DEFAULT_TEST_PATH = os.path.join(repo_root, "output_benchset_rebalanced_test_clap_strict.jsonl")

DEFAULT_EMBEDDINGS_PATH = os.path.join(script_dir, "embeddings_clap_bench.pt")
DEFAULT_REPORT_PATH = os.path.join(script_dir, "eval_clap_bench_full_report.json")
OPT_LEVELS = ("O0", "O1", "O2", "O3")


class CLAPEmbedder:
    def __init__(self, model_path=MODEL_ID, device="cuda", batch_size=32, max_length=1024):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        print(f"Loading CLAP from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.model.eval()

    def prepare_input(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        formatted_batch = []
        for sample in batch:
            asm = sample["asm"]
            asm_list = asm.split("\n") if isinstance(asm, str) else asm
            asm_list = asm_list[:self.max_length]
            formatted_batch.append({str(i): inst for i, inst in enumerate(asm_list)})

        raw_inputs = self.tokenizer(formatted_batch)
        pad_id = self.tokenizer.pad_token_id or 0
        final_inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

        for i in range(len(batch)):
            input_ids = raw_inputs["input_ids"][i][:self.max_length]
            attention_mask = raw_inputs["attention_mask"][i][:self.max_length]
            token_type_ids = raw_inputs["token_type_ids"][i][:self.max_length]
            pad_len = self.max_length - len(input_ids)

            final_inputs["input_ids"].append(input_ids + [pad_id] * pad_len)
            final_inputs["attention_mask"].append(attention_mask + [0] * pad_len)
            final_inputs["token_type_ids"].append(token_type_ids + [0] * pad_len)

        return {k: torch.tensor(v, dtype=torch.long, device=self.device) for k, v in final_inputs.items()}

    @torch.no_grad()
    def encode_batch(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        inputs = self.prepare_input(batch)
        outputs = self.model(**inputs)
        if outputs.ndim == 3:
            embeddings = outputs[:, 0, :]
        else:
            embeddings = outputs
        return F.normalize(embeddings.float(), p=2, dim=1)

    @torch.no_grad()
    def encode_samples(self, samples: List[Dict[str, Any]], progress_every=10) -> torch.Tensor:
        all_embeddings = []
        total_batches = (len(samples) + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        for batch_idx, start in enumerate(range(0, len(samples), self.batch_size), start=1):
            batch = samples[start:start + self.batch_size]
            embeddings = self.encode_batch(batch)
            all_embeddings.append(embeddings.cpu())

            if progress_every and (batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == total_batches):
                processed = min(start + len(batch), len(samples))
                elapsed = time.time() - start_time
                samples_per_sec = processed / max(elapsed, 1e-9)
                remaining = max(len(samples) - processed, 0)
                eta_sec = remaining / max(samples_per_sec, 1e-9)
                print(
                    f"Inference progress: {batch_idx}/{total_batches} batches "
                    f"({processed}/{len(samples)} samples, {samples_per_sec:.1f} samples/s, "
                    f"ETA {eta_sec / 60:.1f} min)",
                    flush=True,
                )

        return torch.cat(all_embeddings, dim=0)


def extract_clap_bench_embeddings(embedder, pairs, progress_every=10):
    ids = []
    query_opts = []
    target_opts = []
    query_samples = []
    target_samples = []

    for pair in pairs:
        ids.append(pair["id"])
        query_opts.append(pair["query"]["opt"])
        target_opts.append(pair["target"]["opt"])
        query_samples.append(pair["query"])
        target_samples.append(pair["target"])

    print("Encoding CLAP query embeddings...")
    query_embeddings = embedder.encode_samples(query_samples, progress_every=progress_every)
    print("Encoding CLAP target embeddings...")
    target_embeddings = embedder.encode_samples(target_samples, progress_every=progress_every)

    return {
        "ids": ids,
        "query_opts": query_opts,
        "target_opts": target_opts,
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLAP on the NovaXLLM2Vec rebalanced bench split.")
    parser.add_argument("--data", default=DEFAULT_TEST_PATH, help="CLAP-format bench JSONL path.")
    parser.add_argument("--model", default=MODEL_ID, help="CLAP model ID or local path.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embeddings_path", default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--report_path", default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = load_jsonl(args.data)
    pairs = build_eval_pairs(samples, seed=args.seed)
    if not pairs:
        raise RuntimeError("No CLAP bench eval pairs were built.")

    embedder = CLAPEmbedder(
        model_path=args.model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    result = extract_clap_bench_embeddings(embedder, pairs, progress_every=args.progress_every)

    torch.save(result, args.embeddings_path)
    print(f"Saved embeddings to {args.embeddings_path}")

    report = compute_report(result)
    report["source"] = os.path.abspath(args.embeddings_path)
    report["model"] = args.model
    report["data"] = os.path.abspath(args.data)

    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full report to {args.report_path}")

    print_report_summary(report)


if __name__ == "__main__":
    main()
