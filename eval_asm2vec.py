import os
import sys
import json
import hashlib
import shutil
import time
import torch
from tqdm import tqdm


sys.path.insert(0, os.path.abspath('./asm2vec-pytorch'))
from asm2vec.utils import load_data, load_model, train
from metrics import EvaluationEngine

def dump_test_data(jsonl_path, temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    file_metadata = {}
    print(f"Dumping test JSONL to {temp_dir}...")
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            raw_lines = data['asm'] if isinstance(data['asm'], list) else data['asm'].split('\n')
            valid_lines = [inst.strip() for inst in raw_lines if inst.strip()]
            
            if not valid_lines:
                continue 
            
            asm_text = "_block_label:\n" + "\n".join(f"    {inst}" for inst in valid_lines)
            
            safe_id = data['id'].replace("/", "_")[:100]
            file_hash = hashlib.md5(data['id'].encode()).hexdigest()[:8]
            filename = f"{safe_id}_{data['opt']}_{file_hash}.s"
            
            # Keep track of original ID and optimization level
            file_metadata[filename] = {"id": data['id'], "opt": data['opt']}
            
            with open(os.path.join(temp_dir, filename), 'w') as out_f:
                out_f.write(asm_text)
                
    return file_metadata

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_jsonl_path = "/home/ra72yeq/projects/NovaXLLM2Vec/binarycorp_test_baseline.jsonl"
    temp_dir = "./asm2vec_test_temp"
    model_path = "asm2vec_weights/model.pt"
    save_path = "/home/ra72yeq/projects/Embedding_Paper/asm2vec_binarycorp_embeddings.pt"

    file_metadata = dump_test_data(test_jsonl_path, temp_dir)

    print("\nLoading Asm2Vec model...")
    model, tokens = load_model(model_path, device=device)

    print("Parsing test data...")
    functions, _ = load_data(temp_dir, limit=None)

    print(f"\nTest-time optimization for {len(functions)} functions...")
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

    torch.save(model.state_dict(), "asm2vec_test_optimized_backup.pt")

    print("Extracting embeddings...")
    all_ids, all_opts, all_embs = [], [], []

    model = model.cpu()

    for i, func in enumerate(functions):
        filename = func.meta.get('name')
        # meta['name'] is a Path object or string injected by load_data in utils.py
        if filename is not None:
            basename = os.path.basename(str(filename))
            meta = file_metadata.get(basename)
        else:
            meta = None

        if not meta:
            continue

        # index i is the exact row in embeddings_f (matches load order)
        emb = model.embeddings_f(torch.tensor([i])).detach().squeeze(0)
        
        all_ids.append(meta['id'])
        all_opts.append(meta['opt'])
        all_embs.append(emb)
        
    results_dict = {
        "ids": all_ids,
        "opts": all_opts,
        "embeddings": torch.stack(all_embs),
        "stats": {
            "avg_ms_per_sample": avg_ms_per_sample,
            "peak_memory_mb": peak_mem
        }
    }
    
    torch.save(results_dict, save_path)
    print(f"\nEmbeddings saved to {save_path}")

    engine = EvaluationEngine(device=device)
    report = engine.evaluate(
        results_dict=results_dict,
        pool_sizes=[50, 100, 200, 500, "global"],
        k_list=[1, 5, 10],
        num_trials=100
    )

    print("\n" + "="*50)
    print(f"ASM2VEC | latency = {avg_ms_per_sample:.2f} ms/function (GD), memory = {peak_mem:.1f} MB peak")
    print("="*50)
    
    for pool, metrics in report.items():
        print(f"\n[{pool}]")
        print(f"NDCG@10:  {metrics['NDCG@10']:.4f}")
        print(f"Recall@1: {metrics['Recall@1']:.4f}")
        print(f"MRR:      {metrics['MRR']:.4f}")

if __name__ == "__main__":
    main()