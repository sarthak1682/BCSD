import os
import sys
import json
import torch
import torch.nn as nn
import argparse
import math
from models import CLAPEmbedder, JTransEmbedder, NovaStudentEmbedder
from metrics import EvaluationEngine

class LatentAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_latents=512, num_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        attn_output, _ = self.cross_attn(query=latents, key=hidden_states, value=hidden_states)
        output = self.mlp(self.layer_norm(attn_output))
        return output.mean(dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class StudentDistillationModule(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers=2, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=1024)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids, key_padding_mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        if key_padding_mask is None:
            key_padding_mask = (input_ids == self.pad_id)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.out_proj(x)

def load_data(filepath: str):
    data =[]
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["clap", "jtrans", "nova_student"])
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--jtrans_path", type=str, default="/home/ra72yeq/projects/Embedding_Paper/jTrans/models/jTrans-finetune")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    dataset = load_data(args.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Initializing {args.model.upper()}...")
    if args.model == "clap":
        embedder = CLAPEmbedder(device=device, batch_size=args.batch_size)
    elif args.model == "jtrans":
        embedder = JTransEmbedder(model_path=args.jtrans_path, device=device, batch_size=args.batch_size)
    elif args.model == "nova_student":
        CACHE_DIR = "/home/ra72yeq/.cache/huggingface/hub/models--lt-asset--nova-1.3b/snapshots/4b4805bac4f13ef8bec678072ef60609ea3b0e77"
        sys.path.insert(0, CACHE_DIR)
        from transformers import AutoTokenizer
        from modeling_nova import NovaTokenizer

        base_tokenizer = AutoTokenizer.from_pretrained("lt-asset/nova-1.3b", cache_dir=CACHE_DIR)
        base_tokenizer.add_special_tokens({'additional_special_tokens': ['[MASK]']})
        nova_tokenizer = NovaTokenizer(base_tokenizer)

        STUDENT_DIR = "/home/ra72yeq/projects/NovaXLLM2Vec/nova_distilled_student_10"

        with open(os.path.join(STUDENT_DIR, "student_config.json"), "r") as f:
            cfg = json.load(f)

        student_model = StudentDistillationModule(
            vocab_size=cfg["vocab_size"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            pad_id=base_tokenizer.pad_token_id or 0
        ).to(device).to(torch.bfloat16)

        lal_head = LatentAttentionLayer(hidden_dim=cfg["hidden_dim"]).to(device).to(torch.bfloat16)

        student_model.load_state_dict(torch.load(os.path.join(STUDENT_DIR, "student_model.pt"), map_location=device))
        lal_head.load_state_dict(torch.load(os.path.join(STUDENT_DIR, "lal_head.pt"), map_location=device))
        
        student_model.eval()
        lal_head.eval()
        
        embedder = NovaStudentEmbedder(
            student_model=student_model, 
            lal_head=lal_head, 
            tokenizer=nova_tokenizer, 
            device=device, 
            batch_size=args.batch_size
        )

    print(f"Running inference (batch_size={args.batch_size})...")
    results = embedder.run_inference(dataset)

    if args.save_path:
        torch.save(results, args.save_path)
        print(f"Embeddings saved to {args.save_path}")

    engine = EvaluationEngine(device=device)
    report = engine.evaluate(
        results_dict=results,
        pool_sizes=[50, 100, 200, 500, "global"],
        k_list=[1, 5, 10],
        num_trials=100
    )
    
    print("\n" + "="*50)
    print(f"Performance Stats for {args.model.upper()}:")
    print(f"Latency: {results['stats']['avg_ms_per_sample']:.2f} ms / function")
    print(f"Memory:  {results['stats']['peak_memory_mb']:.1f} MB peak")
    print("="*50)
    
    for pool, metrics in report.items():
        print(f"\n[{pool}]")
        print(f"NDCG@10:  {metrics['NDCG@10']:.4f}")
        print(f"Recall@1: {metrics['Recall@1']:.4f}")
        print(f"MRR:      {metrics['MRR']:.4f}")

if __name__ == "__main__":
    main()