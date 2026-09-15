"""Asm2Vec training on the Bench (NewData) dataset.

Mirrors binarycorp3m/asm2vec/train.py but points at the bench train split
(`output_benchset_rebalanced_train_nova.jsonl`). The Nova-format JSONL already
carries the `id`, `opt`, `asm` fields Asm2Vec needs, so no separate conversion
is required beyond dumping each function to a `.s` file the parser understands.

Saves weights to: binarycorp_bench/asm2vec/asm2vec_weights/model.pt
"""

import os
import sys
import json
import hashlib
import shutil
import torch
from tqdm import tqdm

# Setup path to the asm2vec-pytorch submodule relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
asm2vec_submodule_path = os.path.abspath(os.path.join(script_dir, "../../../asm2vec-pytorch"))
sys.path.insert(0, asm2vec_submodule_path)

from asm2vec.utils import load_data, train, save_model
from common import asm_to_asm2vec_text


def resolve_bench_train_path():
    """Locate the bench train JSONL (cluster path first, then repo-relative)."""
    candidates = [
        "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_train_nova.jsonl",
        os.path.abspath(os.path.join(script_dir, "../../../nvemb/output_benchset_rebalanced_train_nova.jsonl")),
        os.path.abspath(os.path.join(script_dir, "../../../output_benchset_rebalanced_train_nova.jsonl")),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find bench train JSONL. Looked in:\n  " + "\n  ".join(candidates)
    )


def dump_train_data(jsonl_path, temp_dir):
    if os.path.exists(temp_dir):
        print(f"Purging old temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    with open(jsonl_path, 'r') as fh:
        total_lines = sum(1 for _ in fh)

    written_files = set()

    print(f"Dumping Assembly to {temp_dir}...")
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Processing JSONL"):
            if not line.strip():
                continue
            data = json.loads(line)
            asm_text = asm_to_asm2vec_text(data['asm'])
            if asm_text.strip() == "_block_label:":
                continue

            # truncate + MD5 suffix to avoid OS filename length limits and collisions.
            # Hash combines id+opt so distinct bench variants never collide.
            safe_id = data['id'].replace("/", "_")[:100]
            file_hash = hashlib.md5(f"{data['id']}_{data['opt']}".encode()).hexdigest()[:8]
            filename = f"{safe_id}_{data['opt']}_{file_hash}.s"

            if filename not in written_files:
                with open(os.path.join(temp_dir, filename), 'w') as out_f:
                    out_f.write(asm_text)
                written_files.add(filename)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    jsonl_path = resolve_bench_train_path()
    temp_dir = os.path.join(script_dir, "asm2vec_train_temp")

    dump_train_data(jsonl_path, temp_dir)

    print("\nLoading data into Asm2Vec (this may take a few minutes)...")
    functions, tokens = load_data(temp_dir, limit=None)
    print(f"Loaded {len(functions)} functions. Vocabulary size: {tokens.size()}")

    if tokens.size() == 0:
        print("ERROR: Vocabulary is empty. Data parsing failed.")
        return

    def callback(context):
        progress = f'Epoch {context["epoch"]} | time = {context["time"]:.2f}s, loss = {context["loss"]:.4f}'
        if context["accuracy"]:
            progress += f', accuracy = {context["accuracy"]:.4f}'
        print(progress)

    print("\nStarting Asm2Vec training...")
    model = train(
        functions=functions,
        tokens=tokens,
        model=None,
        embedding_size=200,
        batch_size=8192,
        epochs=10,
        neg_sample_num=5,
        calc_acc=False,
        device=device,
        callback=callback,
        learning_rate=0.02
    )

    save_dir = os.path.join(script_dir, "asm2vec_weights")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model.pt")
    save_model(save_path, model, tokens)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
