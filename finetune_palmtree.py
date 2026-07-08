import argparse
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath("./PalmTree"))
from palmtree_binary_utils import extract_palmtree_corpus, load_function_records
from palmtree_finetune_utils import (
    PalmTreeFunctionEncoder,
    build_positive_pairs,
    contrastive_loss,
    load_palmtree_pretrained,
    save_palmtree_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune PalmTree for function-level retrieval with contrastive loss"
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
        help="Optional binary file or directory to extract into --data-path before training",
    )
    parser.add_argument("--min-instructions", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--instruction-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("PalmTree/weights/palmtree_function_finetuned.pt"),
        help="Output path for the finetuned PalmTree checkpoint",
    )
    return parser.parse_args()


def ensure_training_data(args: argparse.Namespace) -> None:
    if args.binary_input is not None:
        stats = extract_palmtree_corpus(
            binary_input=args.binary_input,
            output_path=args.data_path,
            min_instructions=args.min_instructions,
            clean=True,
        )
        print(
            "Extracted PalmTree corpus from binaries: "
            f"{stats['valid_binaries']} valid binaries, "
            f"{stats['emitted_functions']} emitted functions"
        )
        return

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"--data-path does not exist: {args.data_path}. "
            "Provide --binary-input to build it first."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_training_data(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = load_function_records(args.data_path)
    pairs = build_positive_pairs(records)
    if len(pairs) < 2:
        raise RuntimeError("Need at least two positive pairs to run contrastive finetuning.")

    model, vocab = load_palmtree_pretrained(device=device)
    encoder = PalmTreeFunctionEncoder(
        model=model,
        vocab=vocab,
        device=device,
        instruction_batch_size=args.instruction_batch_size,
        pooling="mean",
    )
    encoder.train()

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    print(f"Loaded {len(records)} records and {len(pairs)} positive pairs.")
    print("Starting PalmTree finetuning...")

    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        epoch_loss = 0.0
        num_steps = 0

        for start in range(0, len(pairs), args.batch_size):
            batch_pairs = pairs[start : start + args.batch_size]
            if len(batch_pairs) < 2:
                continue

            left_records = [pair[0] for pair in batch_pairs]
            right_records = [pair[1] for pair in batch_pairs]

            optimizer.zero_grad()
            left_embs = encoder(left_records)
            right_embs = encoder(right_records)
            loss = contrastive_loss(left_embs, right_embs, temperature=args.temperature)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_steps += 1

        avg_loss = epoch_loss / max(1, num_steps)
        print(f"Epoch {epoch} | loss = {avg_loss:.4f}")

    save_palmtree_checkpoint(
        output_path=args.model_out,
        encoder=encoder,
        metadata={
            "data_path": str(args.data_path),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
        },
    )
    print(f"Finetuned PalmTree checkpoint saved to {args.model_out}")


if __name__ == "__main__":
    main()
