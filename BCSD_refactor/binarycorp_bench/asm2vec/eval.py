import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../../")))

from asm2vec.utils import load_data, load_model, train
from binarycorp_bench.asm2vec.common import asm_to_asm2vec_text
from binarycorp_bench.eval_bench import compute_report, print_report_summary
from shared.data_utils import build_eval_pairs, get_embeddings_dir, load_jsonl, set_seed


def resolve_bench_test_path():
    candidates = [
        os.path.abspath(os.path.join(script_dir, "../../../nvemb/output_benchset_rebalanced_test_nova.jsonl")),
        os.path.abspath(os.path.join(script_dir, "../../../output_benchset_rebalanced_test_nova.jsonl")),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find bench test JSONL. Looked in:\n  " + "\n  ".join(candidates)
    )


def dump_test_data(jsonl_path, temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    file_metadata = {}
    print(f"Dumping test JSONL to {temp_dir}...")

    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            asm_text = asm_to_asm2vec_text(data['asm'])
            if asm_text.strip() == "_block_label:":
                continue

            # Hash combines id+opt so distinct bench variants get unique filenames.
            safe_id = data['id'].replace("/", "_")[:100]
            file_hash = hashlib.md5(f"{data['id']}_{data['opt']}".encode()).hexdigest()[:8]
            filename = f"{safe_id}_{data['opt']}_{file_hash}.s"

            file_metadata[filename] = {"id": data['id'], "opt": data['opt']}

            with open(os.path.join(temp_dir, filename), 'w') as out_f:
                out_f.write(asm_text)

    return file_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None,
                        help="Bench JSONL to evaluate (defaults to the official bench test split).")
    parser.add_argument("--tag", default="",
                        help="Suffix for temp dir and output filenames (unique per unseen project).")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for embeddings/report (defaults to the asm2vec embeddings dir).")
    args = parser.parse_args()
    set_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_jsonl_path = args.data or resolve_bench_test_path()
    suffix = f"_{args.tag}" if args.tag else ""
    temp_dir = os.path.join(script_dir, f"asm2vec_test_temp{suffix}")
    model_path = os.path.join(script_dir, "asm2vec_weights/model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained Asm2Vec model at {model_path}. Run train.py first."
        )

    file_metadata = dump_test_data(test_jsonl_path, temp_dir)

    print("\nLoading Asm2Vec model...")
    model, tokens = load_model(model_path, device=device)

    print("Parsing test data...")
    functions, _ = load_data(temp_dir, limit=None)

    print(f"\nTest-time optimization for {len(functions)} functions...")
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    # mode='test' freezes vocabulary weights; only function vectors are updated
    model = train(
        functions=functions,
        tokens=tokens,
        model=model,
        embedding_size=200,
        batch_size=2048,
        epochs=10,
        neg_sample_num=5,
        calc_acc=False,
        device=device,
        mode='test',
        learning_rate=0.02
    )

    inference_time_ms = (time.time() - start_time) * 1000
    avg_ms_per_sample = inference_time_ms / max(1, len(functions))
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device == 'cuda' else 0.0

    print("Extracting per-instance embeddings...")
    model = model.cpu()
    inst_emb = {}
    for i, func in enumerate(functions):
        filename = func.meta.get('name')
        if filename is None:
            continue
        basename = os.path.basename(str(filename))
        meta = file_metadata.get(basename)
        if not meta:
            continue
        emb = model.embeddings_f(torch.tensor([i])).detach().squeeze(0)
        inst_emb[(meta['id'], meta['opt'])] = emb

    print(f"Built embeddings for {len(inst_emb)} instances.")

    eval_samples = load_jsonl(test_jsonl_path)
    eval_pairs = build_eval_pairs(eval_samples, seed=42)

    ids, query_opts, target_opts = [], [], []
    query_embs, target_embs = [], []
    skipped = 0
    for pair in eval_pairs:
        fid = pair["id"]
        q_key = (fid, pair["query"]["opt"])
        t_key = (fid, pair["target"]["opt"])
        if q_key not in inst_emb or t_key not in inst_emb:
            skipped += 1
            continue
        ids.append(fid)
        query_opts.append(pair["query"]["opt"])
        target_opts.append(pair["target"]["opt"])
        query_embs.append(inst_emb[q_key])
        target_embs.append(inst_emb[t_key])

    print(f"Assembled {len(ids)} eval pairs ({skipped} skipped — missing/empty instances).")

    eval_result = {
        "ids": ids,
        "query_opts": query_opts,
        "target_opts": target_opts,
        "query_embeddings": torch.stack(query_embs).float(),
        "target_embeddings": torch.stack(target_embs).float(),
        "stats": {
            "avg_ms_per_sample": avg_ms_per_sample,
            "peak_memory_mb": peak_mem,
        },
    }

    emb_dir = args.output_dir or get_embeddings_dir("binarycorp_bench", "asm2vec")
    os.makedirs(emb_dir, exist_ok=True)
    save_name = os.path.join(emb_dir, f"eval_bench_embeddings{suffix}.pt")
    torch.save(eval_result, save_name)
    print(f"Embeddings saved to {save_name}")

    print("\n" + "=" * 50)
    print(f"ASM2VEC | latency = {avg_ms_per_sample:.2f} ms/function (GD), memory = {peak_mem:.1f} MB peak")
    print("=" * 50)

    report = compute_report(eval_result)
    report["source"] = os.path.abspath(save_name)
    report["model"] = "asm2vec"
    report["data"] = os.path.abspath(test_jsonl_path)
    report["inference"] = eval_result["stats"]

    metrics_path = os.path.join(emb_dir, f"eval_asm2vec_bench_metrics{suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full report to {metrics_path}")

    print_report_summary(report)


if __name__ == "__main__":
    main()
