import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath("./PalmTree"))
from palmtree_binary_utils import extract_palmtree_corpus, load_function_records
from palmtree_finetune_utils import load_finetuned_palmtree_encoder
from metrics import EvaluationEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PalmTree function embeddings from extracted records or directly from binaries"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="JSONL file containing function-level PalmTree records",
    )
    parser.add_argument(
        "--binary-input",
        type=Path,
        default=None,
        help="Optional binary file or directory to extract into --data-path before evaluation",
    )
    parser.add_argument("--min-instructions", type=int, default=1)
    parser.add_argument("--instruction-batch-size", type=int, default=256)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("PalmTree/weights/palmtree_function_finetuned.pt"),
        help="Finetuned PalmTree checkpoint. If missing, the base pretrained model is used.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("PalmTree/weights/palmtree_embeddings.pt"),
        help="Where to save the evaluation embeddings",
    )
    return parser.parse_args()


def ensure_eval_data(args: argparse.Namespace) -> None:
    if args.binary_input is not None:
        stats = extract_palmtree_corpus(
            binary_input=args.binary_input,
            output_path=args.data_path,
            min_instructions=args.min_instructions,
            clean=True,
        )
        print(
            "Extracted PalmTree evaluation corpus from binaries: "
            f"{stats['valid_binaries']} valid binaries, "
            f"{stats['emitted_functions']} emitted functions"
        )
        return

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"--data-path does not exist: {args.data_path}. "
            "Provide --binary-input to build it first."
        )


def main() -> None:
    args = parse_args()
    ensure_eval_data(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = args.checkpoint_path if args.checkpoint_path.exists() else None
    if checkpoint_path is None:
        print("Finetuned checkpoint not found. Falling back to the base PalmTree pretrained model.")

    encoder = load_finetuned_palmtree_encoder(
        checkpoint_path=checkpoint_path,
        device=device,
        instruction_batch_size=args.instruction_batch_size,
    )
    encoder.eval()

    records = load_function_records(args.data_path)
    ids = [str(record["id"]) for record in records]
    opts = [str(record.get("opt", "unknown")) for record in records]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    with torch.no_grad():
        embeddings = encoder(records).detach().cpu()
    elapsed_ms = (time.time() - start_time) * 1000.0

    results_dict = {
        "ids": ids,
        "opts": opts,
        "embeddings": embeddings,
        "stats": {
            "avg_ms_per_sample": elapsed_ms / max(1, len(records)),
            "peak_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device == "cuda" else 0.0
            ),
        },
    }

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results_dict, args.save_path)
    print(f"Embeddings saved to {args.save_path}")

    engine = EvaluationEngine(device=device)
    report = engine.evaluate(
        results_dict=results_dict,
        pool_sizes=[50, 100, 200, 500, "global"],
        k_list=[1, 5, 10],
        num_trials=100,
    )

    print("\n" + "=" * 50)
    print(
        "PALMTREE | "
        f"latency = {results_dict['stats']['avg_ms_per_sample']:.2f} ms/function, "
        f"memory = {results_dict['stats']['peak_memory_mb']:.1f} MB peak"
    )
    print("=" * 50)

    for pool, metrics in report.items():
        print(f"\n[{pool}]")
        print(f"NDCG@10:  {metrics['NDCG@10']:.4f}")
        print(f"Recall@1: {metrics['Recall@1']:.4f}")
        print(f"MRR:      {metrics['MRR']:.4f}")


if __name__ == "__main__":
    main()
