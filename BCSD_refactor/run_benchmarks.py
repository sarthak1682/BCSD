"""Unified benchmark runner for BCSD_refactor.

Ported from BCSD/run_benchmarks.py with the following improvements:
  - Imports from shared/ instead of inlining model code.
  - NovaStudentEmbedder now uses pool_mask (instruction-prefix exclusion)
    consistent with binarycorp3m/nova_student/eval.py.
  - Saves metrics report as JSON (was stdout-only in the original).
  - STUDENT_DIR defaults to ./model_checkpoints/nova_student/student_best
    (lowest val-loss checkpoint) with fallback to student_final.

Usage examples:
  python run_benchmarks.py --model nova_student --data /path/to/test.jsonl
  python run_benchmarks.py --model clap --data /path/to/test.jsonl
  python run_benchmarks.py --model jtrans --data /path/to/test.jsonl \\
      --jtrans_path /path/to/jTrans-finetune
  python run_benchmarks.py --model nova_student --data /path/to/test.jsonl \\
      --student_dir ./model_checkpoints/nova_student/student_best \\
      --save_path ./embeddings/my_run.pt \\
      --report_path ./results/my_run.json
"""

import argparse
import json
import os
import sys
from typing import Optional

import torch

# Ensure shared/ is importable when run from any cwd
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

from shared.data_utils import load_jsonl, get_embeddings_dir
from shared.embedders import CLAPEmbedder, JTransEmbedder, NovaStudentEmbedder
from shared.nova_utils import NOVA_CACHE_DIR, setup_nova_tokenizer
from shared.student_model import LatentAttentionLayer, StudentDistillationModule

# metrics.py lives in binarycorp3m/ (identical EvaluationEngine for 3m-style eval)
sys.path.insert(0, os.path.join(_REPO_ROOT, "binarycorp3m"))
from metrics import EvaluationEngine


def _resolve_student_dir(cli_arg: Optional[str] = None) -> str:
    """Return the student checkpoint directory to load from.

    Priority:
      1. --student_dir CLI argument
      2. ./model_checkpoints/nova_student/student_best  (lowest val-loss)
      3. ./model_checkpoints/nova_student/student_final (last epoch)
    """
    if cli_arg:
        return cli_arg
    for subdir in ("student_best", "student_final"):
        path = os.path.join("model_checkpoints", "nova_student", subdir)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find a student checkpoint. "
        "Pass --student_dir or run distillation first."
    )


def build_nova_student_embedder(student_dir: str, device: str, batch_size: int) -> NovaStudentEmbedder:
    print(f"Loading Nova student from {student_dir}...")
    _, nova_tokenizer, _ = setup_nova_tokenizer()

    config_path = os.path.join(student_dir, "student_config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    pad_id = nova_tokenizer.tokenizer.pad_token_id or 0

    student_model = StudentDistillationModule(
        vocab_size=cfg["vocab_size"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        pad_id=pad_id,
    ).to(device).to(torch.bfloat16)

    lal_head = LatentAttentionLayer(
        hidden_dim=cfg["hidden_dim"],
        num_latents=512,
        num_heads=8,
    ).to(device).to(torch.bfloat16)

    student_model.load_state_dict(
        torch.load(os.path.join(student_dir, "student_model.pt"), map_location="cpu")
    )
    lal_head.load_state_dict(
        torch.load(os.path.join(student_dir, "lal_head.pt"), map_location="cpu")
    )
    student_model.eval()
    lal_head.eval()

    return NovaStudentEmbedder(
        student_model=student_model,
        lal_head=lal_head,
        tokenizer=nova_tokenizer,
        device=device,
        batch_size=batch_size,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run inference + retrieval evaluation for BCSD models."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["clap", "jtrans", "nova_student"],
        help="Which embedder to evaluate.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the evaluation .jsonl file.",
    )
    parser.add_argument(
        "--student_dir",
        default=None,
        help="Path to distilled student checkpoint dir (nova_student only). "
             "Defaults to model_checkpoints/nova_student/student_best.",
    )
    parser.add_argument(
        "--jtrans_path",
        default="/home/ra72yeq/projects/Embedding_Paper/jTrans/models/jTrans-finetune",
        help="Path to jTrans fine-tuned model directory (jtrans only).",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="Print inference progress every N batches (0 to disable).",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save embeddings .pt file. "
             "Defaults to embeddings/<dataset>/<model>/eval_embeddings.pt.",
    )
    parser.add_argument(
        "--report_path",
        default=None,
        help="Path to save JSON metrics report. "
             "Defaults to <save_path>.metrics.json.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading data from {args.data}...")
    dataset = load_jsonl(args.data)

    print(f"Initialising {args.model.upper()} (batch_size={args.batch_size})...")
    if args.model == "clap":
        embedder = CLAPEmbedder(device=device, batch_size=args.batch_size)
    elif args.model == "jtrans":
        embedder = JTransEmbedder(
            model_path=args.jtrans_path, device=device, batch_size=args.batch_size
        )
    else:  # nova_student
        student_dir = _resolve_student_dir(args.student_dir)
        embedder = build_nova_student_embedder(student_dir, device, args.batch_size)

    print("Running inference...")
    results = embedder.run_inference(dataset, progress_every=args.progress_every)

    # Determine save paths
    if args.save_path:
        emb_path = args.save_path
    else:
        dataset_name = os.path.splitext(os.path.basename(args.data))[0]
        emb_dir = get_embeddings_dir(dataset_name, args.model)
        emb_path = os.path.join(emb_dir, "eval_embeddings.pt")

    os.makedirs(os.path.dirname(os.path.abspath(emb_path)), exist_ok=True)
    torch.save(results, emb_path)
    print(f"Embeddings saved to {emb_path}")

    # Evaluate
    engine = EvaluationEngine(device=device)
    report = engine.evaluate(
        results_dict=results,
        pool_sizes=[50, 100, 200, 500, "global"],
        k_list=[1, 5, 10],
        num_trials=100,
    )

    print("\n" + "=" * 50)
    print(f"Performance stats — {args.model.upper()}")
    print(f"  Latency : {results['stats']['avg_ms_per_sample']:.2f} ms / function")
    print(f"  Memory  : {results['stats']['peak_memory_mb']:.1f} MB peak")
    print("=" * 50)

    for pool, metrics in report.items():
        print(f"\n[{pool}]")
        print(f"  NDCG@10:  {metrics['NDCG@10']:.4f}")
        print(f"  Recall@1: {metrics['Recall@1']:.4f}")
        print(f"  MRR:      {metrics['MRR']:.4f}")

    # Save JSON report
    report_path = args.report_path or (emb_path + ".metrics.json")
    report_data = {
        "model": args.model,
        "data": args.data,
        "student_dir": args.student_dir if args.model == "nova_student" else None,
        "embeddings_path": emb_path,
        "latency_ms_per_sample": results["stats"]["avg_ms_per_sample"],
        "peak_memory_mb": results["stats"]["peak_memory_mb"],
        "metrics": report,
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nMetrics report saved to {report_path}")


if __name__ == "__main__":
    main()
