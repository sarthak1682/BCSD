import os
import sys
import json
import hashlib
import shutil
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('./asm2vec-pytorch'))
from asm2vec.utils import load_data, train, save_model

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
            data = json.loads(line)
            raw_lines = data['asm'] if isinstance(data['asm'], list) else data['asm'].split('\n')
            
            valid_lines = [inst.strip() for inst in raw_lines if inst.strip()]
            if not valid_lines:
                continue 
            
            # parser requires at least one label to open a basic block
            asm_text = "_block_label:\n" + "\n".join(f"    {inst}" for inst in valid_lines)

            # truncate + MD5 suffix to avoid OS filename length limits and collisions
            safe_id = data['id'].replace("/", "_")[:100]
            file_hash = hashlib.md5(data['id'].encode()).hexdigest()[:8]
            filename = f"{safe_id}_{data['opt']}_{file_hash}.s"
            
            if filename not in written_files:
                with open(os.path.join(temp_dir, filename), 'w') as out_f:
                    out_f.write(asm_text)
                written_files.add(filename)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    jsonl_path = "/home/ra72yeq/projects/NovaXLLM2Vec/binarycorp_train.jsonl"
    temp_dir = "./asm2vec_train_temp"

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

    os.makedirs("asm2vec_weights", exist_ok=True)
    save_path = "asm2vec_weights/model.pt"
    save_model(save_path, model, tokens)
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    main()