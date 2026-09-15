import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from peft import PeftModel

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../")))

from eval_bench import build_eval_pairs, compute_report, print_report_summary
from shared.data_utils import asm_to_text, get_embeddings_dir, load_jsonl, set_seed
from shared.nova_utils import load_nova, setup_nova_tokenizer
from shared.pooling import MeanLabelPooling
from shared.profiling import InferenceProfiler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "


def default_eval_path():
    repo_root = os.path.abspath(os.path.join(script_dir, "../../../"))
    candidates = [
        os.path.join(repo_root, "nvemb", "output_benchset_rebalanced_test_nova.jsonl"),
        os.path.join(repo_root, "output_benchset_rebalanced_test_nova.jsonl"),
        "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_test_nova.jsonl",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Nova inst-mean BCSD adapter on the BinaryCorp bench report."
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=os.environ.get("NOVA_INSTMEAN_DIR", "./model_checkpoints/nova_teacher_instmean_bench"),
    )
    parser.add_argument("--eval_data", default=default_eval_path())
    parser.add_argument("--eval_tag", default=os.environ.get("EVAL_TAG", "").strip())
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def slug(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_.-")


def default_eval_label(checkpoint_dir):
    abs_dir = os.path.abspath(checkpoint_dir)
    parent = os.path.basename(os.path.dirname(abs_dir))
    leaf = os.path.basename(abs_dir)
    return slug(f"{parent}_{leaf}") or "nova_teacher_instmean"


def ensure_outputs_do_not_exist(paths):
    existing = [path for path in paths if os.path.exists(path)]
    if existing:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing eval output(s):\n"
            f"{formatted}\n"
            "Set --eval_tag or EVAL_TAG to a fresh label and rerun."
        )


@torch.no_grad()
def encode_texts(model, nova_tokenizer, pooling, texts, batch_size=16, max_length=2048, device="cuda", profiler=None):
    model.eval()
    pooling.eval()

    label_ids = nova_tokenizer.labels
    base_tokenizer = nova_tokenizer.tokenizer
    pad_id = base_tokenizer.pad_token_id or 0
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in range(0, len(texts), batch_size):
        batch_idx = (start // batch_size) + 1
        batch_texts = texts[start:start + batch_size]
        batch_input_ids = []
        batch_masks = []
        batch_label_positions = []

        for text, char_types in batch_texts:
            result = nova_tokenizer.encode("", text, char_types)
            input_ids = result["input_ids"][:max_length]
            raw_mask = result["nova_attention_mask"]
            length = len(input_ids)
            mask = np.maximum(raw_mask[:length, :length], raw_mask[:length, :length].T)
            label_pos = [idx for idx, token_id in enumerate(input_ids) if token_id in label_ids]
            batch_input_ids.append(input_ids)
            batch_masks.append(mask)
            batch_label_positions.append(label_pos)

        max_len = max(len(ids) for ids in batch_input_ids)
        padded_ids = np.full((len(batch_texts), max_len), pad_id, dtype=np.int64)
        padded_masks = np.zeros((len(batch_texts), max_len, max_len), dtype=np.float32)

        for row, (input_ids, mask) in enumerate(zip(batch_input_ids, batch_masks)):
            length = len(input_ids)
            padded_ids[row, :length] = input_ids
            padded_masks[row, :length, :length] = mask

        input_ids_t = torch.tensor(padded_ids, dtype=torch.long, device=device)
        nova_mask_t = torch.tensor(padded_masks, dtype=torch.bfloat16, device=device)
        key_padding_mask = input_ids_t.eq(pad_id)

        if profiler is not None:
            with profiler:
                outputs = model(input_ids=input_ids_t, nova_attention_mask=nova_mask_t, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                pooled = pooling(hidden, batch_label_positions, key_padding_mask=key_padding_mask)
            profiler.total_samples += len(batch_texts)
        else:
            outputs = model(input_ids=input_ids_t, nova_attention_mask=nova_mask_t, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            pooled = pooling(hidden, batch_label_positions, key_padding_mask=key_padding_mask)

        all_embeddings.append(pooled.float().cpu())

        if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches:
            print(f"Processed batch {batch_idx}/{total_batches} ({min(start + batch_size, len(texts))}/{len(texts)} texts)")

        del input_ids_t, nova_mask_t, key_padding_mask, outputs, hidden, pooled

    return torch.cat(all_embeddings, dim=0)


def extract_eval_embeddings(model, nova_tokenizer, pooling, pairs, batch_size, max_length, device):
    query_texts = []
    target_texts = []
    ids = []
    query_opts = []
    target_opts = []

    for pair in pairs:
        query_asm = asm_to_text(pair["query"]["asm"])
        target_asm = asm_to_text(pair["target"]["asm"])
        query_text = INSTRUCT_TEMPLATE + query_asm
        query_texts.append((query_text, "0" * len(INSTRUCT_TEMPLATE) + "1" * len(query_asm)))
        target_texts.append((target_asm, "1" * len(target_asm)))
        ids.append(pair["id"])
        query_opts.append(pair["query"]["opt"])
        target_opts.append(pair["target"]["opt"])

    profiler = InferenceProfiler(device)

    print("Warmup pass...")
    _ = encode_texts(model, nova_tokenizer, pooling, query_texts[:batch_size], batch_size, max_length, device)

    print("Encoding query embeddings...")
    query_embeddings = encode_texts(
        model, nova_tokenizer, pooling, query_texts, batch_size, max_length, device, profiler
    )
    print("Encoding target embeddings...")
    target_embeddings = encode_texts(
        model, nova_tokenizer, pooling, target_texts, batch_size, max_length, device, profiler
    )

    return {
        "ids": ids,
        "query_opts": query_opts,
        "target_opts": target_opts,
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
        "stats": profiler.get_stats(),
    }


def main():
    args = parse_args()
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not os.path.exists(args.eval_data):
        raise FileNotFoundError(f"Eval data not found: {args.eval_data}")

    set_seed(args.seed)
    eval_label = slug(args.eval_tag) if args.eval_tag else default_eval_label(checkpoint_dir)
    eval_suffix = f"_{eval_label}" if eval_label else ""
    emb_dir = get_embeddings_dir("binarycorp_bench", "nova_teacher_instmean")
    embeddings_path = os.path.join(emb_dir, f"eval_bench_embeddings{eval_suffix}.pt")
    metrics_path = os.path.join(emb_dir, f"eval_teacher_bench_metrics{eval_suffix}.json")
    ensure_outputs_do_not_exist([embeddings_path, metrics_path])

    print("Nova inst-mean eval")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Eval data:  {args.eval_data}")
    print(f"Embeddings: {embeddings_path}")
    print(f"Metrics:    {metrics_path}")

    base_tokenizer, nova_tokenizer, _ = setup_nova_tokenizer()
    base_model = load_nova(device_id=0, base_tokenizer=base_tokenizer)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()
    device = next(model.parameters()).device
    pooling = MeanLabelPooling().to(device)

    eval_samples = load_jsonl(args.eval_data)
    eval_pairs = build_eval_pairs(eval_samples, seed=args.seed)
    results = extract_eval_embeddings(
        model=model,
        nova_tokenizer=nova_tokenizer,
        pooling=pooling,
        pairs=eval_pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=str(device),
    )

    torch.save(results, embeddings_path)
    print(f"query embeddings shape: {results['query_embeddings'].shape}")
    print(f"target embeddings shape: {results['target_embeddings'].shape}")
    print(f"Saved embeddings to {embeddings_path}")

    print("\n" + "=" * 50)
    print(f"NOVA INST-MEAN TEACHER | Latency: {results['stats']['avg_ms_per_sample']:.2f} ms/function")
    print(f"NOVA INST-MEAN TEACHER | Memory:  {results['stats']['peak_memory_mb']:.1f} MB peak")
    print("=" * 50)

    report = compute_report(results)
    report["source"] = os.path.abspath(embeddings_path)
    report["model"] = f"nova_teacher_instmean{eval_suffix}"
    report["checkpoint"] = checkpoint_dir
    report["pooling"] = "mean_label_positions"
    report["data"] = os.path.abspath(args.eval_data)

    with open(metrics_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved full report to {metrics_path}")
    print_report_summary(report)


if __name__ == "__main__":
    main()
