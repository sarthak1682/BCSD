"""Qwen EBM training — BinaryCorp 3M (O0↔O3 pairs).

Mirrors ``binarycorp3m/nova_ebm/run_stages.py`` but for Qwen2.5-Coder-0.5B:
  - Standard HF tokenizer (no NovaTokenizer / label attention).
  - Last-token pooling (no AttentionPooling on label positions).
  - 4D bidirectional attention mask for contrastive encoding.
  - DoRA only (no full finetuning).  MNTP skipped (useless).

Pipeline:
  Stage 1: Causal-LM cross-opt translation (DoRA, bidirectional pairs)
  Stage 3: Supervised contrastive + last-token pool (DoRA, cgte_loss)

Examples:
  python run_stages.py --stages 1,3
  python run_stages.py --stages 3 --s1_ckpt ./ckpts/s1_final
"""

import argparse
import gc
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Setup paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from metrics import EvaluationEngine
from shared.data_utils import (
    set_seed, load_jsonl, get_embeddings_dir, build_ebm_translation_item,
)
from shared.qwen_utils import (
    QwenPoolingHead,
    QwenTranslationCollator,
    QwenPairCollator,
    run_qwen_train,
    extract_qwen_embeddings,
    add_ebm_special_tokens,
    resize_ebm_token_embeddings,
)

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths & hyperparameters
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
TRAIN_DATA = "/home/ra72yeq/projects/NovaXLLM2Vec/binarycorp3m_train_nova.jsonl"
EVAL_DATA = "/home/ra72yeq/projects/NovaXLLM2Vec/binarycorp3m_test_nova.jsonl"
OUTPUT_DIR = "./model_checkpoints/qwen_ebm"

# DoRA config — same targets as Nova (Qwen2 uses identical linear names)
DORA_CFG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    use_dora=True,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

CFG = dict(
    seed=42,
    gpu_id=1,
    max_length=1024,
    max_grad_norm=1.0,
    warmup_ratio=0.03,
    log_interval=100,
    # Stage 1
    s1_epochs=1, s1_batch=32, s1_grad_accum=8, s1_lr=1.6e-4,
    bidir_pairs=True,
    # Stage 3
    s3_epochs=1, s3_batch=32, s3_grad_accum=4, s3_lr=3e-5,
    temperature=0.05,
    # Eval
    eval_batch_size=16,
)


# ---------------------------------------------------------------------------
# Datasets (identical to Nova's — just text pairs)
# ---------------------------------------------------------------------------

class CrossOptTranslationDataset(Dataset):
    def __init__(self, samples: List[Dict], bidirectional: bool = True) -> None:
        o0 = {s["id"]: s for s in samples if s["opt"] == "O0"}
        o3 = {s["id"]: s for s in samples if s["opt"] == "O3"}
        self.pairs: List[Tuple[str, str, str]] = []
        for fid, sample0 in o0.items():
            if fid in o3:
                sample3 = o3[fid]
                self.pairs.append(build_ebm_translation_item(
                    sample0["asm"], sample0["opt"],
                    sample3["asm"], sample3["opt"],
                ))
                if bidirectional:
                    self.pairs.append(build_ebm_translation_item(
                        sample3["asm"], sample3["opt"],
                        sample0["asm"], sample0["opt"],
                    ))
        print(f"  {len(self.pairs):,} translation pairs "
              f"({'bidir' if bidirectional else 'unidir'})")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


class ContrastivePairDataset(Dataset):
    def __init__(self, samples: List[Dict]) -> None:
        o0 = {s["id"]: s["asm"] for s in samples if s["opt"] == "O0"}
        o3 = {s["id"]: s["asm"] for s in samples if s["opt"] == "O3"}
        template = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
        self.pairs = [(template + o0[fid], o3[fid]) for fid in o0 if fid in o3]
        print(f"Paired samples: {len(self.pairs)} (Total items: {len(self.pairs) * 2})")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


# ---------------------------------------------------------------------------
# DoRA wrapper
# ---------------------------------------------------------------------------

def _wrap_dora(model) -> torch.nn.Module:
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, **DORA_CFG)
    return get_peft_model(model, cfg)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_qwen(from_dir: Optional[str] = None, device: str = "cuda", tokenizer=None):
    src = from_dir or MODEL_ID
    print(f"  loading Qwen2ForCausalLM from {src}")
    m = AutoModelForCausalLM.from_pretrained(
        src,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    if tokenizer is not None:
        resize_ebm_token_embeddings(m, tokenizer)
    return m


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, eval_samples, args, log_fn) -> None:
    log_fn("\nEVALUATION")
    gc.collect()
    torch.cuda.empty_cache()
    device = next(model.parameters()).device

    result = extract_qwen_embeddings(
        model, tokenizer, eval_samples,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        device=str(device),
    )
    emb_dir = get_embeddings_dir("binarycorp3m", "qwen_ebm")
    emb_path = os.path.join(emb_dir, "eval_embeddings.pt")
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    torch.save(result, emb_path)
    log_fn(f"  embeddings {result['embeddings'].shape}  →  {emb_path}")

    log_fn("\nPerformance Stats:")
    log_fn(f"  Latency: {result['stats']['avg_ms_per_sample']:.2f} ms/function")
    log_fn(f"  Memory:  {result['stats']['peak_memory_mb']:.1f} MB peak")

    log_fn("\nEvaluating with EvaluationEngine...")
    engine = EvaluationEngine(device=device)
    report = engine.evaluate(
        results_dict=result,
        pool_sizes=[50, 100, 200, 500, "global"],
        k_list=[1, 5, 10],
        num_trials=100,
    )

    for pool, metrics in report.items():
        log_fn(f"\n[{pool}]")
        log_fn(f"  NDCG@10:  {metrics['NDCG@10']:.4f}")
        log_fn(f"  Recall@1: {metrics['Recall@1']:.4f}")
        log_fn(f"  MRR:      {metrics['MRR']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen EBM 3M training")
    parser.add_argument("--stages", default="1,3",
                        help="e.g. '1,3' | '3' | 'eval'")
    parser.add_argument("--s1_ckpt", default=None)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_length", type=int, default=CFG["max_length"])
    parser.add_argument("--seed", type=int, default=CFG["seed"])
    args = parser.parse_args()

    for k, v in CFG.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    set_seed(args.seed)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir,
                            f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def log(msg: str) -> None:
        print(msg)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    log(f"Qwen EBM  |  {datetime.now()}  |  stages={args.stages}")
    log(f"output_dir={args.output_dir}  device={device}\n")

    run_stages = set(s.strip().lower() for s in args.stages.split(","))

    # Load tokenizer once
    log(f"Loading tokenizer from {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    add_ebm_special_tokens(tokenizer)
    log(f"  tokenizer size={len(tokenizer)}")

    log("Loading datasets …")
    train_samples = load_jsonl(TRAIN_DATA)
    eval_samples = load_jsonl(EVAL_DATA)

    # Stage 1
    if "1" in run_stages:
        log("\nStage 1: Cross-opt translation")
        base = _load_qwen(args.s1_ckpt, str(device), tokenizer)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        model = _wrap_dora(base)
        model.enable_input_require_grads()

        loader = DataLoader(
            CrossOptTranslationDataset(train_samples, bidirectional=args.bidir_pairs),
            batch_size=args.s1_batch, shuffle=True, num_workers=4, pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
            collate_fn=QwenTranslationCollator(tokenizer, args.max_length),
        )
        run_qwen_train(model, loader, args.s1_epochs, args.s1_lr,
                       args.s1_grad_accum, args.max_grad_norm,
                       args.warmup_ratio, args.log_interval, log,
                       causal_lm=True)

        model = model.merge_and_unload()
        s1_ckpt = os.path.join(args.output_dir, "s1_final")
        model.save_pretrained(s1_ckpt)
        tokenizer.save_pretrained(s1_ckpt)
        log(f"  saved (merged) → {s1_ckpt}")
        args.s1_ckpt = s1_ckpt
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 3
    if "3" in run_stages:
        log("\nStage 3: Supervised contrastive + last-token pooling")
        base = _load_qwen(args.s1_ckpt, str(device), tokenizer)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        model = _wrap_dora(base)
        model.enable_input_require_grads()

        pooling_head = QwenPoolingHead().to(device)

        loader = DataLoader(
            ContrastivePairDataset(train_samples),
            batch_size=args.s3_batch, shuffle=True, num_workers=4, pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
            collate_fn=QwenPairCollator(tokenizer, args.max_length),
        )
        run_qwen_train(model, loader, args.s3_epochs, args.s3_lr,
                       args.s3_grad_accum, args.max_grad_norm,
                       args.warmup_ratio, args.log_interval, log,
                       causal_lm=False, pooling_head=pooling_head,
                       temperature=args.temperature)

        model = model.merge_and_unload()
        s3_ckpt = os.path.join(args.output_dir, "s3_final")
        os.makedirs(s3_ckpt, exist_ok=True)
        model.save_pretrained(s3_ckpt)
        tokenizer.save_pretrained(s3_ckpt)
        args.s3_ckpt = s3_ckpt
        log(f"  saved (merged) → {s3_ckpt}")

    # Eval
    if "eval" in run_stages or "3" in run_stages:
        log("\nEval")
        model = _load_qwen(args.s3_ckpt, str(device), tokenizer)
        pooling_head = QwenPoolingHead()
        run_eval(model, tokenizer, eval_samples, args, log)

    log("\nDone.")


if __name__ == "__main__":
    main()
