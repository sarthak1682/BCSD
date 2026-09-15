"""Qwen EBM training — Bench dataset (cross-arch/opt/compiler/obfuscation).

Mirrors ``binarycorp_bench/nova_ebm/run_stages.py`` but for Qwen2.5-Coder-0.5B-Instruct.

Pipeline:
  Stage 1: Causal-LM cross-variant translation (DoRA, bidirectional, all variants)
  Stage 2: LLM2Vec-style MNTP with bidirectional attention (DoRA)
  Stage 3: cGTE contrastive + last-token pool (DoRA)

Eval emits the full bench report: pooled recalls + cross-arch/opt/compiler/obfuscation
transfer matrices via ``eval_bench.compute_report``.
"""

import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../")))

from eval_bench import build_eval_pairs, compute_report, print_report_summary
from shared.data_utils import (
    set_seed, load_jsonl, group_samples_by_id, get_embeddings_dir,
    ebm_preprocess_asm, build_ebm_translation_item,
)
from shared.profiling import InferenceProfiler
from shared.qwen_utils import (
    patch_qwen_bidirectional,
    QwenPoolingHead,
    QwenMNTPCollator,
    QwenTranslationCollator,
    QwenPairCollator,
    run_qwen_train,
    add_ebm_special_tokens,
    resize_ebm_token_embeddings,
)

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths & hyperparameters
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../../"))

def _first_existing_path(*paths: str) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


# Historical dataset name: these JSONL files are plain bench records
# (id/opt/asm), not Nova model inputs.
_DEFAULT_TRAIN = _first_existing_path(
    os.path.join(_repo_root, "nvemb", "output_benchset_rebalanced_train_nova.jsonl"),
    os.path.join(_repo_root, "output_benchset_rebalanced_train_nova.jsonl"),
    "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_train_nova.jsonl",
)

_DEFAULT_EVAL = _first_existing_path(
    os.path.join(_repo_root, "nvemb", "output_benchset_rebalanced_test_nova.jsonl"),
    os.path.join(_repo_root, "output_benchset_rebalanced_test_nova.jsonl"),
    "/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_test_nova.jsonl",
)

OUTPUT_DIR = "./model_checkpoints/qwen_ebm_bench_fixed"

# DoRA config
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
    max_length=512,
    max_grad_norm=1.0,
    warmup_ratio=0.03,
    log_interval=100,
    # Stage 1
    s1_epochs=5, s1_batch=4, s1_grad_accum=4, s1_lr=1e-5,
    bidir_pairs=True,
    # Stage 2
    s2_epochs=1, s2_batch=4, s2_grad_accum=4, s2_lr=1e-5,
    mask_prob=0.15,
    # Stage 3
    s3_epochs=5, s3_batch=4, s3_grad_accum=4, s3_lr=1e-5,
    temperature=0.05,
    # Eval
    eval_batch_size=16,
)

INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class CrossOptTranslationDataset(Dataset):
    """Pairs asm variants of the same function for causal-LM translation."""

    def __init__(self, samples: List[Dict], bidirectional: bool = True) -> None:
        self.pairs: List[Tuple[str, str, str]] = []
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
                self.pairs.append(build_ebm_translation_item(
                    src["asm"], src["opt"], tgt["asm"], tgt["opt"],
                    preprocess=True,
                ))
                if bidirectional:
                    self.pairs.append(build_ebm_translation_item(
                        tgt["asm"], tgt["opt"], src["asm"], src["opt"],
                        preprocess=True,
                    ))
        print(f"  {len(self.pairs):,} translation pairs "
              f"({'bidir' if bidirectional else 'unidir'}, {skipped} skipped)")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


class MNTPSampleDataset(Dataset):
    """Single-function samples for bidirectional MNTP."""

    def __init__(self, samples: List[Dict]) -> None:
        self.samples = [
            {"asm": ebm_preprocess_asm(s["asm"], s["opt"], include_settings=True)}
            for s in samples
        ]
        print(f"  {len(self.samples):,} MNTP samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


class ContrastivePairDataset(Dataset):
    """Produces (query, positive) pairs for cGTE contrastive loss."""

    def __init__(self, samples: List[Dict]) -> None:
        grouped = group_samples_by_id(samples)
        self.pairs: List[Tuple[str, str]] = []
        skipped = 0
        for fid, variants in grouped.items():
            if len(variants) < 2:
                skipped += 1
                continue
            ordered = variants[:]
            random.shuffle(ordered)
            for idx, query in enumerate(ordered):
                positive = ordered[(idx + 1) % len(ordered)]
                self.pairs.append((
                    ebm_preprocess_asm(query["asm"], query["opt"], include_settings=True),
                    ebm_preprocess_asm(positive["asm"], positive["opt"], include_settings=True),
                ))
        print(f"Paired samples: {len(self.pairs)} (Total items: {len(self.pairs) * 2})")
        print(f"  {skipped} functions skipped with <2 variants")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


# ---------------------------------------------------------------------------
# DoRA wrapper
# ---------------------------------------------------------------------------

def _wrap_dora(model) -> torch.nn.Module:
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, **DORA_CFG)
    return get_peft_model(model, cfg)


def _save_epoch_checkpoint(
    model,
    tokenizer,
    output_dir: str,
    stage_name: str,
    epoch: int,
    log_fn,
    pooling_head: Optional[torch.nn.Module] = None,
) -> None:
    ckpt_dir = os.path.join(output_dir, f"{stage_name}_epoch_{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    if pooling_head is not None:
        torch.save(pooling_head.state_dict(), os.path.join(ckpt_dir, "pooling_head.pt"))
    log_fn(f"  saved epoch checkpoint → {ckpt_dir}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_qwen(from_dir: Optional[str] = None, device: str = "cuda", tokenizer=None):
    src = from_dir or MODEL_ID
    adapter_config = os.path.join(src, "adapter_config.json") if src else ""
    if from_dir and os.path.isfile(adapter_config):
        with open(adapter_config, "r") as f:
            cfg = json.load(f)
        base_src = cfg.get("base_model_name_or_path") or MODEL_ID
        if os.path.abspath(str(base_src)) == os.path.abspath(str(src)):
            base_src = MODEL_ID
        print(f"  loading Qwen2ForCausalLM base from {base_src}")
        base = _load_qwen(base_src, device=device, tokenizer=tokenizer)
        print(f"  loading PEFT adapter from {src}")
        adapter = PeftModel.from_pretrained(base, src, is_trainable=False)
        return adapter.merge_and_unload()

    print(f"  loading Qwen2ForCausalLM from {src}")
    m = AutoModelForCausalLM.from_pretrained(
        src,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if device.startswith("cuda") else None,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if tokenizer is not None:
        resize_ebm_token_embeddings(m, tokenizer)
    return m


def _resolve_upstream_ckpt(args, run_stages, log_fn) -> Optional[str]:
    if args.s2_ckpt:
        return args.s2_ckpt
    s2_default = os.path.join(args.output_dir, "s2_final")
    if "2" not in run_stages and os.path.isdir(s2_default):
        log_fn(f"  using existing stage-2 checkpoint: {s2_default}")
        return s2_default
    if args.allow_skip_mntp:
        if args.s1_ckpt:
            return args.s1_ckpt
        s1_default = os.path.join(args.output_dir, "s1_final")
        if "1" not in run_stages and os.path.isdir(s1_default):
            log_fn(f"  using existing stage-1 checkpoint: {s1_default}")
            return s1_default
    return None


# ---------------------------------------------------------------------------
# Bench eval embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_bench_texts(model, tokenizer, texts, batch_size, max_length, device, profiler=None):
    """Encode a list of asm strings → last-token pooled embeddings [N, D]."""
    model.eval()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn_2d = enc["attention_mask"].bool()
        attn = enc["attention_mask"].to(device)

        if profiler is not None:
            with profiler:
                out = model(input_ids=input_ids, attention_mask=attn,
                           output_hidden_states=True)
                hidden = out.hidden_states[-1]
                lengths = attn_2d.sum(dim=1) - 1
                lengths = lengths.clamp(min=0).to(device)
                idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.size(-1))
                pooled = hidden.gather(1, idx).squeeze(1)
            profiler.total_samples += len(batch_texts)
        else:
            out = model(input_ids=input_ids, attention_mask=attn,
                       output_hidden_states=True)
            hidden = out.hidden_states[-1]
            lengths = attn_2d.sum(dim=1) - 1
            lengths = lengths.clamp(min=0).to(device)
            idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.size(-1))
            pooled = hidden.gather(1, idx).squeeze(1)

        all_embs.append(pooled.float().cpu())

    return torch.cat(all_embs, dim=0)


def extract_bench_embeddings(model, tokenizer, pairs, batch_size=16,
                              max_length=1024, device="cuda"):
    """Extract separate query/target embeddings for bench eval."""
    query_texts = []
    target_texts = []
    ids = []
    query_opts = []
    target_opts = []

    for pair in pairs:
        query_asm = ebm_preprocess_asm(
            pair["query"]["asm"], pair["query"]["opt"], include_settings=True
        )
        target_asm = ebm_preprocess_asm(
            pair["target"]["asm"], pair["target"]["opt"], include_settings=True
        )
        query_texts.append(INSTRUCT_TEMPLATE + query_asm)
        target_texts.append(target_asm)
        ids.append(pair["id"])
        query_opts.append(pair["query"]["opt"])
        target_opts.append(pair["target"]["opt"])

    profiler = InferenceProfiler(device)

    print("Warmup pass...")
    _ = _encode_bench_texts(model, tokenizer, query_texts[:batch_size],
                            batch_size=batch_size, max_length=max_length,
                            device=device)

    print("Encoding query embeddings...")
    query_embs = _encode_bench_texts(model, tokenizer, query_texts,
                                     batch_size=batch_size, max_length=max_length,
                                     device=device, profiler=profiler)
    print("Encoding target embeddings...")
    target_embs = _encode_bench_texts(model, tokenizer, target_texts,
                                      batch_size=batch_size, max_length=max_length,
                                      device=device, profiler=profiler)

    return {
        "ids": ids,
        "query_opts": query_opts,
        "target_opts": target_opts,
        "query_embeddings": query_embs,
        "target_embeddings": target_embs,
        "stats": profiler.get_stats(),
    }


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, eval_samples, args, log_fn) -> None:
    log_fn("\nEVALUATION")
    gc.collect()
    torch.cuda.empty_cache()
    device = next(model.parameters()).device

    eval_pairs = build_eval_pairs(eval_samples, seed=args.seed)
    result = extract_bench_embeddings(
        model, tokenizer, eval_pairs,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        device=str(device),
    )
    emb_dir = get_embeddings_dir("binarycorp_bench", "qwen_ebm")
    emb_path = os.path.join(emb_dir, "eval_bench_embeddings.pt")
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    torch.save(result, emb_path)
    log_fn(f"  query embeddings {result['query_embeddings'].shape}  →  {emb_path}")
    log_fn(f"  target embeddings {result['target_embeddings'].shape}  →  {emb_path}")

    log_fn("\nPerformance Stats:")
    log_fn(f"  Latency: {result['stats']['avg_ms_per_sample']:.2f} ms/function")
    log_fn(f"  Memory:  {result['stats']['peak_memory_mb']:.1f} MB peak")

    log_fn("\nEvaluating with bench metrics...")
    report = compute_report(result)
    report["source"] = os.path.abspath(emb_path)
    report["model"] = "qwen_ebm_bench_fixed"
    report["data"] = os.path.abspath(args.eval_data)

    report_path = os.path.join(args.output_dir, "eval_bench_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log_fn(f"  report → {report_path}")

    print_report_summary(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen EBM bench training: stages 1 (translation), 2 (MNTP), 3 (contrastive) + eval"
    )
    parser.add_argument("--stages", default="1,2,3",
                        help="e.g. '1,2,3' | '2,3' | '3' | 'eval'")
    parser.add_argument("--s1_ckpt", default=None)
    parser.add_argument("--s2_ckpt", default=None)
    parser.add_argument("--s3_ckpt", default=None)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--train_data", default=_DEFAULT_TRAIN)
    parser.add_argument("--eval_data", default=_DEFAULT_EVAL)
    parser.add_argument("--gpu_id", type=int, default=CFG["gpu_id"])
    parser.add_argument("--max_length", type=int, default=CFG["max_length"])
    parser.add_argument("--seed", type=int, default=CFG["seed"])
    parser.add_argument("--max_grad_norm", type=float, default=CFG["max_grad_norm"])
    parser.add_argument("--warmup_ratio", type=float, default=CFG["warmup_ratio"])
    parser.add_argument("--log_interval", type=int, default=CFG["log_interval"])
    parser.add_argument("--s1_epochs", type=int, default=CFG["s1_epochs"])
    parser.add_argument("--s1_batch", type=int, default=CFG["s1_batch"])
    parser.add_argument("--s1_grad_accum", type=int, default=CFG["s1_grad_accum"])
    parser.add_argument("--s1_lr", type=float, default=CFG["s1_lr"])
    parser.add_argument("--bidir_pairs", action="store_true", default=CFG["bidir_pairs"])
    parser.add_argument("--s2_epochs", type=int, default=CFG["s2_epochs"])
    parser.add_argument("--s2_batch", type=int, default=CFG["s2_batch"])
    parser.add_argument("--s2_grad_accum", type=int, default=CFG["s2_grad_accum"])
    parser.add_argument("--s2_lr", type=float, default=CFG["s2_lr"])
    parser.add_argument("--mask_prob", type=float, default=CFG["mask_prob"])
    parser.add_argument("--s3_epochs", type=int, default=CFG["s3_epochs"])
    parser.add_argument("--s3_batch", type=int, default=CFG["s3_batch"])
    parser.add_argument("--s3_grad_accum", type=int, default=CFG["s3_grad_accum"])
    parser.add_argument("--s3_lr", type=float, default=CFG["s3_lr"])
    parser.add_argument("--temperature", type=float, default=CFG["temperature"])
    parser.add_argument("--eval_batch_size", type=int, default=CFG["eval_batch_size"])
    parser.add_argument("--allow_skip_mntp", action="store_true",
                        help="Allow stage 3 to start from stage 1; off by default for EBM reproduction.")
    parser.add_argument("--wandb_project", default=None,
                        help="W&B project name for remote monitoring (optional).")

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

    _wandb = None
    if args.wandb_project:
        try:
            import wandb as _wandb_mod
            _wandb = _wandb_mod
            _wandb.init(
                project=args.wandb_project,
                name=f"qwen_ebm_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args),
            )
            print(f"wandb run: {_wandb.run.url}")
        except Exception as e:
            print(f"[warn] wandb init failed ({e}) — continuing without remote logging")
            _wandb = None

    def log(msg: str) -> None:
        print(msg)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")
        if _wandb is not None:
            _wandb.log({"log": msg})

    log(f"Qwen EBM  |  {datetime.now()}  |  stages={args.stages}")
    log(f"output_dir={args.output_dir}  device={device}\n")

    run_stages = set(s.strip().lower() for s in args.stages.split(","))

    log(f"Loading tokenizer from {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left",
    )

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
    train_samples = load_jsonl(args.train_data)
    eval_samples = load_jsonl(args.eval_data)
    mask_token_id = add_ebm_special_tokens(tokenizer)
    log(f"  tokenizer size={len(tokenizer)}  [MASK]={mask_token_id}")

    # Stage 1
    if "1" in run_stages:
        log("\nStage 1: Cross-variant translation")
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
                       causal_lm=True,
                       wandb_log=_wandb.log if _wandb else None,
                       on_epoch_end=lambda epoch: _save_epoch_checkpoint(
                           model, tokenizer, args.output_dir, "s1", epoch, log
                       ))

        model = model.merge_and_unload()
        s1_ckpt = os.path.join(args.output_dir, "s1_final")
        model.save_pretrained(s1_ckpt)
        tokenizer.save_pretrained(s1_ckpt)
        log(f"  saved (merged) → {s1_ckpt}")
        args.s1_ckpt = s1_ckpt
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 2
    if "2" in run_stages:
        log("\nStage 2: LLM2Vec MNTP with bidirectional attention")
        s1_src = args.s1_ckpt or os.path.join(args.output_dir, "s1_final")
        if not os.path.isdir(s1_src):
            raise FileNotFoundError(
                f"Stage 2 needs a stage-1 checkpoint, but none was found at {s1_src}. "
                "Run stage 1 first or pass --s1_ckpt."
            )
        base = _load_qwen(s1_src, str(device), tokenizer)
        patch_qwen_bidirectional(base)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        model = _wrap_dora(base)
        model.enable_input_require_grads()

        loader = DataLoader(
            MNTPSampleDataset(train_samples),
            batch_size=args.s2_batch, shuffle=True, num_workers=4, pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
            collate_fn=QwenMNTPCollator(
                tokenizer, mask_token_id, args.max_length, mask_prob=args.mask_prob
            ),
        )
        run_qwen_train(model, loader, args.s2_epochs, args.s2_lr,
                       args.s2_grad_accum, args.max_grad_norm,
                       args.warmup_ratio, args.log_interval, log,
                       causal_lm=False, mntp=True,
                       wandb_log=_wandb.log if _wandb else None,
                       on_epoch_end=lambda epoch: _save_epoch_checkpoint(
                           model, tokenizer, args.output_dir, "s2", epoch, log
                       ))

        model = model.merge_and_unload()
        s2_ckpt = os.path.join(args.output_dir, "s2_final")
        model.save_pretrained(s2_ckpt)
        tokenizer.save_pretrained(s2_ckpt)
        log(f"  saved (merged) → {s2_ckpt}")
        args.s2_ckpt = s2_ckpt
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 3
    if "3" in run_stages:
        log("\nStage 3: cGTE contrastive + last-token pooling")
        upstream = _resolve_upstream_ckpt(args, run_stages, log)
        if upstream is None:
            raise FileNotFoundError(
                "Stage 3 needs a stage-2 checkpoint for EBM reproduction. "
                "Run --stages 1,2,3, pass --s2_ckpt, or use --allow_skip_mntp explicitly."
            )
        base = _load_qwen(upstream, str(device), tokenizer)
        patch_qwen_bidirectional(base)
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
                       temperature=args.temperature,
                       wandb_log=_wandb.log if _wandb else None,
                       on_epoch_end=lambda epoch: _save_epoch_checkpoint(
                           model, tokenizer, args.output_dir, "s3", epoch, log,
                           pooling_head=pooling_head,
                       ))

        model = model.merge_and_unload()
        s3_ckpt = os.path.join(args.output_dir, "s3_final")
        os.makedirs(s3_ckpt, exist_ok=True)
        model.save_pretrained(s3_ckpt)
        tokenizer.save_pretrained(s3_ckpt)
        log(f"  saved (merged) → {s3_ckpt}")
        args.s3_ckpt = s3_ckpt

    # Eval
    if "eval" in run_stages or "3" in run_stages:
        log("\nEval")
        model = _load_qwen(args.s3_ckpt, str(device), tokenizer)
        patch_qwen_bidirectional(model)
        run_eval(model, tokenizer, eval_samples, args, log)

    log("\nDone.")


if __name__ == "__main__":
    main()
