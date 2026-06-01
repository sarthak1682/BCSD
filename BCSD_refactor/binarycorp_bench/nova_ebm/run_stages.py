"""Training script for Nova EBM stages 1, 3 and evaluation (bench dataset).

Stage 2 (MNTP) is disabled — empirically hurts performance with Nova
(assembly-specialized model that already handles bidirectional attention).

Examples:
  python run_stages.py --stages 1,3
  python run_stages.py --stages 3 --s1_ckpt ./ckpts/s1_final
  python run_stages.py --stages eval --s3_ckpt ./ckpts/s3_final
"""

import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Setup paths to import shared modules and evaluation script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../")))

from eval_bench import build_eval_pairs, compute_report, print_report_summary
from shared.data_utils import set_seed, load_jsonl as load_binarycorp_jsonl, parse_bench_opt, asm_to_text, group_samples_by_id
from shared.nova_utils import make_bidirectional_nova_mask, NOVA_CACHE_DIR, MODEL_ID
from shared.pooling import AttentionPooling
from shared.collators import TranslationCollator, PairCollator
from shared.losses import contrastive_loss_positive_aware
from shared.training import run_generic_train

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Paths (defaults — all overridable via CLI args)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root  = os.path.abspath(os.path.join(_script_dir, "../../../"))

_DEFAULT_TRAIN = "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_train_nova.jsonl"
if not os.path.exists(_DEFAULT_TRAIN):
    _DEFAULT_TRAIN = os.path.join(_repo_root, "nvemb", "output_benchset_rebalanced_train_nova.jsonl")
if not os.path.exists(_DEFAULT_TRAIN):
    _DEFAULT_TRAIN = os.path.join(_repo_root, "output_benchset_rebalanced_train_nova.jsonl")

_DEFAULT_EVAL = "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_test_nova.jsonl"
if not os.path.exists(_DEFAULT_EVAL):
    _DEFAULT_EVAL = os.path.join(_repo_root, "nvemb", "output_benchset_rebalanced_test_nova.jsonl")
if not os.path.exists(_DEFAULT_EVAL):
    _DEFAULT_EVAL = os.path.join(_repo_root, "output_benchset_rebalanced_test_nova.jsonl")

OUTPUT_DIR         = "./model_checkpoints/nova_ebm_bench"
POOLING_HEAD_FNAME = "pooling_head.pt"

# DoRA config (matches adapter_config.json in checkpoints)
DORA_CFG = dict(
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    use_dora       = True,
    bias           = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)

# Hyperparameters
CFG = dict(
    seed           = 42,
    gpu_id         = 1,
    max_length     = 1024,
    max_grad_norm  = 1.0,
    warmup_ratio   = 0.03,
    log_interval   = 100,
    # Stage 1
    s1_epochs      = 1,    s1_batch = 32,  s1_grad_accum = 8,  s1_lr = 1.6e-4,
    bidir_pairs    = True,
    # Stage 2 (MNTP — disabled, kept for reference)
    # s2_epochs    = 1,    s2_batch = 32,  s2_grad_accum = 8,  s2_lr = 1.6e-3,
    # mask_prob    = 0.15,
    # Stage 3
    s3_epochs      = 1,    s3_batch = 32,  s3_grad_accum = 4,  s3_lr = 3e-5,
    temperature    = 0.05,
    # Eval
    eval_batch_size = 16,
)


# Padding helper

def _pad_batch(
    ids_list:  List[np.ndarray],
    mask_list: List[np.ndarray],
    pad_id:    int,
    max_cap:   Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    max_len = max(len(x) for x in ids_list)
    if max_cap is not None:
        max_len = min(max_len, max_cap)
    B = len(ids_list)
    p_ids  = np.full((B, max_len), pad_id, dtype=np.int64)
    p_mask = np.zeros((B, max_len, max_len), dtype=np.float32)
    for i, (ids, msk) in enumerate(zip(ids_list, mask_list)):
        n = min(len(ids), max_len)
        p_ids[i, :n] = ids[:n]
        p_mask[i, :n, :n] = msk[:n, :n]
    return p_ids, p_mask


# DoRA wrapper

def _wrap_dora(model: nn.Module) -> nn.Module:
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, **DORA_CFG)
    return get_peft_model(model, cfg)


# Stage 1: Cross-opt translation

class CrossOptTranslationDataset(Dataset):
    def __init__(self, samples: List[Dict], bidirectional: bool = True) -> None:
        self.pairs: List[Tuple[str, str]] = []
        grouped = group_samples_by_id(samples)
        skipped = 0
        for fid, variants in grouped.items():
            if len(variants) < 2:
                skipped += 1
                continue
            ordered = variants[:]
            random.shuffle(ordered)
            for idx, src in enumerate(ordered):
                tgt = ordered[(idx + 1) % len(ordered)]
                src_asm = asm_to_text(src["asm"])
                tgt_asm = asm_to_text(tgt["asm"])
                self.pairs.append((src_asm, tgt_asm))
                if bidirectional:
                    self.pairs.append((tgt_asm, src_asm))
        print(f"  {len(self.pairs):,} translation pairs "
              f"({'bidir' if bidirectional else 'unidir'}, {skipped} skipped)")

    def __len__(self):  return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


# Stage 3: Contrastive

class ContrastivePairDataset(Dataset):
    def __init__(self, samples: List[Dict]) -> None:
        grouped = group_samples_by_id(samples)
        func_id_to_int = {fid: i for i, fid in enumerate(sorted(grouped))}
        self.pairs: List[Tuple[str, str, int]] = []
        skipped = 0
        for fid, variants in grouped.items():
            if len(variants) < 2:
                skipped += 1
                continue
            ordered = variants[:]
            random.shuffle(ordered)
            func_int = func_id_to_int[fid]
            for idx, query in enumerate(ordered):
                positive = ordered[(idx + 1) % len(ordered)]
                self.pairs.append((asm_to_text(query["asm"]), asm_to_text(positive["asm"]), func_int))
        print(f"Paired samples: {len(self.pairs)} (Total items: {len(self.pairs) * 2})")
        print(f"  {skipped} functions skipped with <2 variants")

    def __len__(self):  return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


# Embedding extraction

@torch.no_grad()
def encode_bench_texts(model, pooling_head, nova_tokenizer, texts,
                       batch_size=16, max_length=1024, device="cuda"):
    model.eval()
    pooling_head.eval()

    label_ids      = nova_tokenizer.labels
    base_tokenizer = nova_tokenizer.tokenizer
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_input_ids       = []
        batch_masks           = []
        batch_label_positions = []

        for text, char_types in batch_texts:
            result = nova_tokenizer.encode("", text, char_types)
            input_ids = result["input_ids"][:max_length]
            raw_mask  = result["nova_attention_mask"]
            L         = len(input_ids)
            mask      = np.maximum(raw_mask[:L, :L], raw_mask[:L, :L].T)
            label_pos = [j for j, tid in enumerate(input_ids) if tid in label_ids]

            batch_input_ids.append(input_ids)
            batch_masks.append(mask)
            batch_label_positions.append(label_pos)

        max_len = max(len(x) for x in batch_input_ids)
        pad_id  = base_tokenizer.pad_token_id or 0

        padded_ids   = np.full((len(batch_texts), max_len), pad_id, dtype=np.int64)
        padded_masks = np.zeros((len(batch_texts), max_len, max_len), dtype=np.float32)

        for j, (ids, mask) in enumerate(zip(batch_input_ids, batch_masks)):
            L = len(ids)
            padded_ids[j, :L]       = ids
            padded_masks[j, :L, :L] = mask

        input_ids_t = torch.tensor(padded_ids, dtype=torch.long, device=device)
        nova_mask_t = torch.tensor(padded_masks, dtype=torch.bfloat16, device=device)

        outputs = model(input_ids=input_ids_t, nova_attention_mask=nova_mask_t,
                        output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        pooled_batch = pooling_head(hidden, batch_label_positions)
        all_embeddings.append(pooled_batch.float().cpu())

        if i % 500 == 0:
            print(f"Encoded {min(i + len(batch_texts), len(texts))}/{len(texts)}")

    return torch.cat(all_embeddings, dim=0)


def extract_bench_eval_embeddings(model, pooling_head, nova_tokenizer, pairs,
                                  batch_size=16, max_length=1024, device="cuda"):
    instruct_template = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
    query_texts = []
    target_texts = []
    ids = []
    query_opts = []
    target_opts = []

    for pair in pairs:
        query_asm = asm_to_text(pair["query"]["asm"])
        target_asm = asm_to_text(pair["target"]["asm"])
        query_text = instruct_template + query_asm
        query_texts.append((query_text, "0" * len(instruct_template) + "1" * len(query_asm)))
        target_texts.append((target_asm, "1" * len(target_asm)))
        ids.append(pair["id"])
        query_opts.append(pair["query"]["opt"])
        target_opts.append(pair["target"]["opt"])

    print("Encoding query embeddings...")
    query_embeddings = encode_bench_texts(model, pooling_head, nova_tokenizer, query_texts,
                                          batch_size=batch_size, max_length=max_length, device=device)
    print("Encoding target embeddings...")
    target_embeddings = encode_bench_texts(model, pooling_head, nova_tokenizer, target_texts,
                                           batch_size=batch_size, max_length=max_length, device=device)

    return {
        "ids": ids,
        "query_opts": query_opts,
        "target_opts": target_opts,
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
    }


def run_eval(model, pooling_head, nova_tokenizer, eval_samples, args, log_fn) -> None:
    log_fn("\nEVALUATION")
    gc.collect(); torch.cuda.empty_cache()
    device = next(model.parameters()).device

    eval_pairs = build_eval_pairs(eval_samples, seed=args.seed)
    result = extract_bench_eval_embeddings(
        model,
        pooling_head,
        nova_tokenizer,
        eval_pairs,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        device=str(device),
    )
    emb_path = os.path.join(args.output_dir, "eval_bench_embeddings.pt")
    torch.save(result, emb_path)
    log_fn(f"  query embeddings {result['query_embeddings'].shape}  →  {emb_path}")
    log_fn(f"  target embeddings {result['target_embeddings'].shape}  →  {emb_path}")

    log_fn("\nEvaluating with bench metrics...")
    report = compute_report(result)
    report["source"] = os.path.abspath(emb_path)
    report["model"] = "nova_ebm_bench"
    report["data"] = os.path.abspath(args.eval_data)

    report_path = os.path.join(args.output_dir, "eval_bench_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log_fn(f"  report → {report_path}")

    print_report_summary(report)


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nova EBM bench training: stages 1 (cross-opt), 3 (contrastive) + eval"
    )
    # --- Stage control ---
    parser.add_argument("--stages",     default="1,3",
                        help="Comma-separated stages to run, e.g. '1,3' | '3' | 'eval'")
    parser.add_argument("--s1_ckpt",    default=None, help="Path to stage-1 checkpoint to resume from")
    parser.add_argument("--s3_ckpt",    default=None, help="Path to stage-3 checkpoint for eval")
    # --- Paths ---
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Root dir for all stage checkpoints")
    parser.add_argument("--train_data", default=_DEFAULT_TRAIN,
                        help="Path to train .jsonl (overrides hardcoded default)")
    parser.add_argument("--eval_data",  default=_DEFAULT_EVAL,
                        help="Path to eval .jsonl  (overrides hardcoded default)")
    # --- Hardware ---
    parser.add_argument("--gpu_id",     type=int, default=CFG["gpu_id"],
                        help="GPU index to use when CUDA_VISIBLE_DEVICES is not set by scheduler")
    # --- Common hyper-params ---
    parser.add_argument("--max_length", type=int,   default=CFG["max_length"])
    parser.add_argument("--seed",       type=int,   default=CFG["seed"])
    parser.add_argument("--max_grad_norm", type=float, default=CFG["max_grad_norm"])
    parser.add_argument("--warmup_ratio",  type=float, default=CFG["warmup_ratio"])
    parser.add_argument("--log_interval", type=int,   default=CFG["log_interval"])
    # --- Stage 1 ---
    parser.add_argument("--s1_epochs",    type=int,   default=CFG["s1_epochs"])
    parser.add_argument("--s1_batch",     type=int,   default=CFG["s1_batch"])
    parser.add_argument("--s1_grad_accum",type=int,   default=CFG["s1_grad_accum"])
    parser.add_argument("--s1_lr",        type=float, default=CFG["s1_lr"])
    parser.add_argument("--bidir_pairs",  action="store_true", default=CFG["bidir_pairs"])
    # --- Stage 2 (MNTP — disabled) ---
    # parser.add_argument("--s2_epochs",    type=int,   default=CFG["s2_epochs"])
    # parser.add_argument("--s2_batch",     type=int,   default=CFG["s2_batch"])
    # parser.add_argument("--s2_grad_accum",type=int,   default=CFG["s2_grad_accum"])
    # parser.add_argument("--s2_lr",        type=float, default=CFG["s2_lr"])
    # parser.add_argument("--mask_prob",    type=float, default=CFG["mask_prob"])
    # parser.add_argument("--s2_max_steps", type=int,   default=1000,
    #                     help="Hard step cap for MNTP stage (default 1000)")
    # --- Stage 3 ---
    parser.add_argument("--s3_epochs",    type=int,   default=CFG["s3_epochs"])
    parser.add_argument("--s3_batch",     type=int,   default=CFG["s3_batch"])
    parser.add_argument("--s3_grad_accum",type=int,   default=CFG["s3_grad_accum"])
    parser.add_argument("--s3_lr",        type=float, default=CFG["s3_lr"])
    parser.add_argument("--temperature",  type=float, default=CFG["temperature"])
    # --- Eval ---
    parser.add_argument("--eval_batch_size", type=int, default=CFG["eval_batch_size"])
    # --- Remote monitoring ---
    parser.add_argument("--wandb_project", default=None,
                        help="W&B project name for remote monitoring (optional). "
                             "Requires: pip install wandb && wandb login")

    args = parser.parse_args()

    # Back-fill any CFG keys not yet exposed as explicit args (forward-compat)
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

    # --- wandb (optional) ---
    _wandb = None
    if args.wandb_project:
        try:
            import wandb as _wandb_mod
            _wandb = _wandb_mod
            _wandb.init(
                project=args.wandb_project,
                name=f"nova_ebm_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args),
            )
            print(f"wandb run: {_wandb.run.url}")
        except Exception as e:
            print(f"[warn] wandb init failed ({e}) — continuing without remote logging")
            _wandb = None

    def log(msg: str) -> None:
        print(msg)
        with open(log_path, "a") as fh: fh.write(msg + "\n")
        if _wandb is not None:
            _wandb.log({"log": msg})

    log(f"Nova EBM  |  {datetime.now()}  |  stages={args.stages}")
    log(f"output_dir={args.output_dir}  device={device}\n")

    run_stages = set(s.strip().lower() for s in args.stages.split(","))

    log(f"Loading Nova from {NOVA_CACHE_DIR}")
    sys.path.insert(0, NOVA_CACHE_DIR)
    from modeling_nova import NovaForCausalLM, NovaTokenizer  

    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    nova_tokenizer = NovaTokenizer(base_tokenizer)

    def _load_nova(from_dir=None):
        src = from_dir or NOVA_CACHE_DIR
        log(f"  loading NovaForCausalLM from {src}")
        m = NovaForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16,
                                             device_map={"": 0})
        m.resize_token_embeddings(len(base_tokenizer))
        return m

    log("Loading datasets …")
    log(f"  train: {args.train_data}")
    log(f"  eval:  {args.eval_data}")
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(
            f"Train data not found: {args.train_data}\n"
            "Run:  python BCSD_refactor/download_dataset.py --output_dir <dir>"
        )
    if not os.path.exists(args.eval_data):
        raise FileNotFoundError(
            f"Eval data not found: {args.eval_data}\n"
            "Run:  python BCSD_refactor/download_dataset.py --output_dir <dir>"
        )
    train_samples = load_binarycorp_jsonl(args.train_data)
    eval_samples  = load_binarycorp_jsonl(args.eval_data)

    # Stage 1
    if "1" in run_stages:
        log("\nStage 1: Cross-opt translation")
        base = _load_nova(args.s1_ckpt)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        model = _wrap_dora(base)
        model.enable_input_require_grads()

        loader = DataLoader(
            CrossOptTranslationDataset(train_samples, bidirectional=args.bidir_pairs),
            batch_size=args.s1_batch, shuffle=True, num_workers=4, pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
            collate_fn=TranslationCollator(nova_tokenizer, args.max_length),
        )
        run_generic_train(model, loader, args.s1_epochs, args.s1_lr,
                          args.s1_grad_accum, args.max_grad_norm,
                          args.warmup_ratio, args.log_interval, log,
                          wandb_log=_wandb.log if _wandb else None)

        model   = model.merge_and_unload()
        s1_ckpt = os.path.join(args.output_dir, "s1_final")
        model.save_pretrained(s1_ckpt)
        log(f"  saved (merged) → {s1_ckpt}")
        args.s1_ckpt = s1_ckpt
        del model; gc.collect(); torch.cuda.empty_cache()

    # Stage 2 (MNTP — disabled, empirically hurts Nova performance)
    # if "2" in run_stages:
    #     log("\nStage 2: MNTP (bidirectional encoder)")
    #     base = _load_nova(args.s1_ckpt)
    #     base.gradient_checkpointing_enable()
    #     base.config.use_cache = False
    #     model = _wrap_dora(base)
    #     model.enable_input_require_grads()
    #
    #     loader = DataLoader(
    #         train_samples,
    #         batch_size=args.s2_batch, shuffle=True, num_workers=4, pin_memory=True,
    #         persistent_workers=True, prefetch_factor=2,
    #         collate_fn=MNTPCollator(nova_tokenizer, MASK_ID,
    #                                 mask_prob=args.mask_prob,
    #                                 max_length=args.max_length),
    #     )
    #     run_generic_train(model, loader, args.s2_epochs, args.s2_lr,
    #                       args.s2_grad_accum, args.max_grad_norm,
    #                       args.warmup_ratio, args.log_interval, log, mntp=True,
    #                       max_steps=args.s2_max_steps,
    #                       wandb_log=_wandb.log if _wandb else None)
    #
    #     model   = model.merge_and_unload()
    #     s2_ckpt = os.path.join(args.output_dir, "s2_final")
    #     model.save_pretrained(s2_ckpt)
    #     log(f"  saved (merged) → {s2_ckpt}")
    #     args.s2_ckpt = s2_ckpt
    #     del model; gc.collect(); torch.cuda.empty_cache()

    # Stage 3
    if "3" in run_stages:
        log("\nStage 3: Supervised contrastive + attention pooling")
        base = _load_nova(args.s1_ckpt)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        model = _wrap_dora(base)
        model.enable_input_require_grads()

        pooling_head = AttentionPooling(model.config.hidden_size).to(device).to(torch.bfloat16)

        loader = DataLoader(
            ContrastivePairDataset(train_samples),
            batch_size=args.s3_batch, shuffle=True, num_workers=4, pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
            collate_fn=PairCollator(nova_tokenizer, args.max_length),
        )
        run_generic_train(model, loader, args.s3_epochs, args.s3_lr,
                          args.s3_grad_accum, args.max_grad_norm,
                          args.warmup_ratio, args.log_interval, log,
                          contrastive=True, pooling_head=pooling_head,
                          temperature=args.temperature,
                          wandb_log=_wandb.log if _wandb else None)

        model   = model.merge_and_unload()
        s3_ckpt = os.path.join(args.output_dir, "s3_final")
        os.makedirs(s3_ckpt, exist_ok=True)
        model.save_pretrained(s3_ckpt)
        torch.save(pooling_head.state_dict(), os.path.join(s3_ckpt, POOLING_HEAD_FNAME))
        log(f"  saved (merged) + pooling head → {s3_ckpt}")
        args.s3_ckpt = s3_ckpt

    # Eval
    if "eval" in run_stages or "3" in run_stages:
        log("\nEval")
        model        = _load_nova(args.s3_ckpt)
        pooling_head = AttentionPooling(model.config.hidden_size).to(device).to(torch.bfloat16)
        head_path    = os.path.join(args.s3_ckpt, POOLING_HEAD_FNAME) if args.s3_ckpt else None
        if head_path and os.path.isfile(head_path):
            pooling_head.load_state_dict(torch.load(head_path, map_location=device))
            log(f"  pooling head ← {head_path}")
        else:
            log("  no pooling head found, using random init")
        run_eval(model, pooling_head, nova_tokenizer, eval_samples, args, log)

    log("\nDone.")


if __name__ == "__main__":
    main()
