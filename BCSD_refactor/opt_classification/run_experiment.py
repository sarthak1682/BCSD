#!/usr/bin/env python3
"""Orchestrate manifest creation, frozen extraction, probes, and reporting."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .build_manifest import build_manifest
from .report import generate_report


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = Path(
    "/home/ra72yeq/BCSD/nvemb/output_benchset_rebalanced_train_nova.jsonl"
)
DEFAULT_TEST = Path(
    "/home/ra72yeq/BCSD/nvemb/output_benchset_rebalanced_test_nova.jsonl"
)
STRICT_DATA_ROOT = Path("/home/ra72yeq/projects/NovaXLLM2Vec/nvemb")
DEFAULT_CLAP_TRAIN = STRICT_DATA_ROOT / "output_benchset_rebalanced_train_clap_strict.jsonl"
DEFAULT_CLAP_TEST = STRICT_DATA_ROOT / "output_benchset_rebalanced_test_clap_strict.jsonl"
DEFAULT_JTRANS_TRAIN = (
    STRICT_DATA_ROOT / "output_benchset_rebalanced_train_jtrans_strict.jsonl"
)
DEFAULT_JTRANS_TEST = (
    STRICT_DATA_ROOT / "output_benchset_rebalanced_test_jtrans_strict.jsonl"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "opt_classification" / "outputs"
NOVA_INPUT_MODELS = {
    "student",
    "student-pre",
    "ebm-full",
    "ebm-no-causal",
    "nova-raw",
    "ebm-s1",
    "asm2vec",
}


@dataclass(frozen=True)
class ExtractionJob:
    name: str
    model: str
    batch_size: int
    max_length: int
    nova_mask: Optional[str] = None
    pooling: Optional[str] = None


DEPLOYED_JOBS = (
    ExtractionJob("clap", "clap", 128, 256),
    ExtractionJob("ebm-full", "ebm-full", 16, 1024),
    ExtractionJob("ebm-no-causal", "ebm-no-causal", 16, 1024),
    ExtractionJob("student", "student", 64, 1024),
    ExtractionJob("jtrans", "jtrans", 64, 512),
    ExtractionJob("joint-scratch", "joint-scratch", 64, 512),
    ExtractionJob("asm2vec", "asm2vec", 2048, 1024),
)

CONTROL_JOBS = (
    ExtractionJob("clap-raw", "clap-raw", 128, 256),
    ExtractionJob("student-pre", "student-pre", 64, 1024),
    ExtractionJob("jtrans-bench", "jtrans-bench", 64, 512),
    ExtractionJob("nova-raw-native-mean", "nova-raw", 16, 1024, "native", "mean"),
    ExtractionJob(
        "nova-raw-bidirectional-mean",
        "nova-raw",
        16,
        1024,
        "bidirectional",
        "mean",
    ),
    ExtractionJob("ebm-s1-native-mean", "ebm-s1", 16, 1024, "native", "mean"),
    ExtractionJob(
        "ebm-s1-bidirectional-mean",
        "ebm-s1",
        16,
        1024,
        "bidirectional",
        "mean",
    ),
    ExtractionJob(
        "ebm-full-native-mean", "ebm-full", 16, 1024, "native", "mean"
    ),
    ExtractionJob(
        "ebm-full-bidirectional-mean",
        "ebm-full",
        16,
        1024,
        "bidirectional",
        "mean",
    ),
    ExtractionJob(
        "ebm-no-causal-native-mean",
        "ebm-no-causal",
        16,
        1024,
        "native",
        "mean",
    ),
    ExtractionJob(
        "ebm-no-causal-bidirectional-mean",
        "ebm-no-causal",
        16,
        1024,
        "bidirectional",
        "mean",
    ),
)


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _extract_command(
    job: ExtractionJob,
    *,
    manifest: Path,
    output: Path,
    device: str,
    seed: int,
    resume: bool,
    input_jsonls: tuple[Path, Path],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "opt_classification.extract",
        "--manifest",
        str(manifest),
        "--model",
        job.model,
        "--output",
        str(output),
        "--device",
        device,
        "--batch-size",
        str(job.batch_size),
        "--max-length",
        str(job.max_length),
        "--seed",
        str(seed),
    ]
    if job.nova_mask:
        command.extend(["--nova-mask", job.nova_mask])
    if job.pooling:
        command.extend(["--pooling", job.pooling])
    for input_jsonl in input_jsonls:
        command.extend(["--input-jsonl", str(input_jsonl)])
    if resume and output.exists():
        command.append("--resume")
    return command


def _probe_command(
    *,
    manifest: Path,
    embeddings: Path,
    output: Path,
    task: str,
    seed: int,
    bootstrap_samples: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "opt_classification.probe",
        "--manifest",
        str(manifest),
        "--embeddings",
        str(embeddings),
        "--output",
        str(output),
        "--task",
        task,
        "--seed",
        str(seed),
        "--bootstrap-samples",
        str(bootstrap_samples),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", choices=("deployed", "full"), default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--clap-train", type=Path, default=DEFAULT_CLAP_TRAIN)
    parser.add_argument("--clap-test", type=Path, default=DEFAULT_CLAP_TEST)
    parser.add_argument("--jtrans-train", type=Path, default=DEFAULT_JTRANS_TRAIN)
    parser.add_argument("--jtrans-test", type=Path, default=DEFAULT_JTRANS_TEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--skip-asm2vec", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume matching extraction caches and skip completed probe JSON files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.output_root / args.run_id
    manifest = run_dir / "manifest.jsonl"
    manifest_summary = run_dir / "manifest_summary.json"
    embeddings_dir = run_dir / "embeddings"
    probes_dir = run_dir / "probes"

    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use a new --run-id or --resume."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    if not (args.resume and manifest.exists() and manifest_summary.exists()):
        build_manifest(
            args.train,
            args.test,
            manifest,
            manifest_summary,
            val_fraction=args.val_fraction,
            seed=args.seed,
            max_groups=args.max_groups,
            alignment_path_groups=(
                (args.clap_train, args.clap_test),
                (args.jtrans_train, args.jtrans_test),
            ),
        )

    jobs = list(DEPLOYED_JOBS)
    if args.suite == "full":
        jobs.extend(CONTROL_JOBS)
    if args.skip_asm2vec:
        jobs = [job for job in jobs if job.model != "asm2vec"]

    configuration = {
        "run_id": args.run_id,
        "suite": args.suite,
        "seed": args.seed,
        "device": args.device,
        "train": str(args.train.resolve()),
        "test": str(args.test.resolve()),
        "clap_train": str(args.clap_train.resolve()),
        "clap_test": str(args.clap_test.resolve()),
        "jtrans_train": str(args.jtrans_train.resolve()),
        "jtrans_test": str(args.jtrans_test.resolve()),
        "jobs": [job.__dict__ for job in jobs],
    }
    (run_dir / "experiment.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for job in jobs:
        cache_path = embeddings_dir / f"{job.name}.pt"
        if job.model in NOVA_INPUT_MODELS:
            input_jsonls = (args.train, args.test)
        elif job.model.startswith("clap") or job.model == "joint-scratch":
            input_jsonls = (args.clap_train, args.clap_test)
        else:
            input_jsonls = (args.jtrans_train, args.jtrans_test)
        _run(
            _extract_command(
                job,
                manifest=manifest,
                output=cache_path,
                device=args.device,
                seed=args.seed,
                resume=args.resume,
                input_jsonls=input_jsonls,
            )
        )
        for task in ("4way", "o0-vs-o3"):
            result_path = probes_dir / job.name / f"{task}.json"
            if args.resume and result_path.exists():
                print(f"Skipping completed probe: {result_path}", flush=True)
                continue
            _run(
                _probe_command(
                    manifest=manifest,
                    embeddings=cache_path,
                    output=result_path,
                    task=task,
                    seed=args.seed,
                    bootstrap_samples=args.bootstrap_samples,
                )
            )

    generate_report(probes_dir, run_dir)
    print(f"Experiment complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
