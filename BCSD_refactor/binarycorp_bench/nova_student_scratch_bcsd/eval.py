#!/usr/bin/env python3
"""Evaluate a pure BCSD scratch student with the Table I retrieval protocol."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
BCSD_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "binarycorp_bench"))

from binarycorp_bench.eval_bench import (  # noqa: E402
    build_eval_pairs,
    compute_report,
    load_jsonl,
    print_report_summary,
    set_seed,
)
from shared.nova_utils import setup_nova_tokenizer  # noqa: E402
from shared.profiling import InferenceProfiler  # noqa: E402
from shared.student_model import LatentAttentionLayer, StudentDistillationModule  # noqa: E402


DEFAULT_TEST_PATH = BCSD_ROOT / "nvemb" / "output_benchset_rebalanced_test_nova.jsonl"
DEFAULT_STUDENT_DIR = (
    REPO_ROOT / "model_checkpoints" / "nova_student_scratch_bcsd" / "student_best"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "embeddings" / "binarycorp_bench" / "nova_student_scratch_bcsd"
DEFAULT_DISTILLED_REPORT = (
    REPO_ROOT
    / "embeddings"
    / "binarycorp_bench"
    / "nova_student"
    / "eval_student_bench_metrics_20.json"
)
INSTRUCT_TEMPLATE = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-dir", default=str(DEFAULT_STUDENT_DIR))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--distilled-report",
        default=str(DEFAULT_DISTILLED_REPORT),
        help="Current distilled BCSD student report to compare against. Pass '' to skip.",
    )
    parser.add_argument("--eval-tag", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is unavailable")
    return device


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_.-")


def default_eval_tag(student_dir: Path) -> str:
    parent = student_dir.parent.name
    leaf = student_dir.name
    return slug(f"{parent}_{leaf}") or "scratch_bcsd_student"


def safe_torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_checkpoint(student_dir: Path, name: str) -> Path:
    best = student_dir / name.replace(".pt", "_best.pt")
    plain = student_dir / name
    if best.is_file():
        return best
    if plain.is_file():
        return plain
    raise FileNotFoundError(f"Missing checkpoint under {student_dir}: {name}")


def ensure_no_overwrite(paths: Sequence[Path], *, allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing eval output(s):\n"
            f"{formatted}\nUse --eval-tag with a new label or pass --allow-overwrite."
        )


def _full_recall(report: dict[str, Any], key: str) -> float | None:
    value = report.get("full_recall_one_target_per_function", {}).get(key)
    if isinstance(value, dict):
        recall = value.get("recall")
        return None if recall is None else float(recall)
    return None


def retrieval_metric_scalars(report: dict[str, Any]) -> dict[str, float | None]:
    pooled = report.get("pooled_recall", {})
    return {
        "Pool50 R@1": pooled.get("Recall@1_Pool50"),
        "Pool100 R@1": pooled.get("Recall@1_Pool100"),
        "Pool200 R@1": pooled.get("Recall@1_Pool200"),
        "Pool500 R@1": pooled.get("Recall@1_Pool500"),
        "Full R@1": _full_recall(report, "Recall@1"),
        "Full R@5": _full_recall(report, "Recall@5"),
        "Full R@10": _full_recall(report, "Recall@10"),
    }


def load_distilled_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        print(f"distilled report not found; skipping comparison: {path}", flush=True)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_to_distilled(
    pure_report: dict[str, Any],
    distilled_report: dict[str, Any],
) -> dict[str, Any]:
    pure = retrieval_metric_scalars(pure_report)
    distilled = retrieval_metric_scalars(distilled_report)
    rows = []
    for metric_name, pure_value in pure.items():
        distilled_value = distilled.get(metric_name)
        delta = (
            None
            if pure_value is None or distilled_value is None
            else float(distilled_value) - float(pure_value)
        )
        rows.append(
            {
                "metric": metric_name,
                "pure_bcsd": pure_value,
                "distilled_bcsd": distilled_value,
                "distilled_minus_pure": delta,
            }
        )
    primary = next(row for row in rows if row["metric"] == "Full R@1")
    primary_delta = primary["distilled_minus_pure"]
    if primary_delta is None:
        verdict = "Full R@1 comparison unavailable."
    elif primary_delta >= 0:
        verdict = (
            "Nova distillation does not hurt BCSD on Full R@1 "
            f"(distilled >= pure, delta {primary_delta:+.4f})."
        )
    else:
        verdict = (
            "Pure BCSD beats the distilled BCSD student on Full R@1 "
            f"(distilled - pure {primary_delta:+.4f}); distillation may hurt here."
        )
    return {"primary_metric": "Full R@1", "verdict": verdict, "rows": rows}


def _fmt(value: float | None) -> str:
    return "pending" if value is None else f"{float(value):.3f}"


def write_eval_summary(
    output_dir: Path,
    *,
    tag: str,
    report: dict[str, Any],
    comparison: dict[str, Any] | None,
    distilled_report_path: Path | None,
) -> None:
    metrics = retrieval_metric_scalars(report)
    lines = [
        "# Pure BCSD Scratch Student Eval",
        "",
        "## Pure BCSD Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in metrics.items():
        lines.append(f"| {name} | {_fmt(value)} |")

    lines.extend(["", "## Distillation Check", ""])
    if comparison is None:
        lines.append("No distilled BCSD report was available for comparison.")
    else:
        lines.append(comparison["verdict"])
        lines.extend(
            [
                "",
                "| Metric | Pure BCSD | Distilled BCSD | Distilled - Pure |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in comparison["rows"]:
            lines.append(
                "| {metric} | {pure} | {distilled} | {delta} |".format(
                    metric=row["metric"],
                    pure=_fmt(row["pure_bcsd"]),
                    distilled=_fmt(row["distilled_bcsd"]),
                    delta=_fmt(row["distilled_minus_pure"]),
                )
            )
        if distilled_report_path is not None:
            lines.extend(["", f"Distilled report: `{distilled_report_path}`"])

    output_dir.joinpath(f"SUMMARY_{tag}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


@torch.no_grad()
def encode_student_texts(
    texts: list[tuple[str, str]],
    *,
    student: StudentDistillationModule,
    lal_head: LatentAttentionLayer,
    tokenizer,
    pad_id: int,
    device: torch.device,
    batch_size: int,
    max_length: int,
    profiler: InferenceProfiler | None = None,
) -> torch.Tensor:
    all_embs: list[torch.Tensor] = []
    instruct_token_len = len(tokenizer.tokenizer.tokenize(INSTRUCT_TEMPLATE))
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for start in range(0, len(texts), batch_size):
        batch_index = start // batch_size + 1
        batch = texts[start : start + batch_size]
        all_ids: list[Any] = []
        is_query: list[bool] = []
        for text, char_types in batch:
            encoded = tokenizer.encode("", text, char_types)
            ids = encoded["input_ids"][:max_length]
            all_ids.append(ids)
            is_query.append(text.startswith(INSTRUCT_TEMPLATE))
        width = max(len(ids) for ids in all_ids)
        padded = np.full((len(batch), width), pad_id, dtype=np.int64)
        for row, ids in enumerate(all_ids):
            padded[row, : len(ids)] = ids
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        key_padding_mask = input_ids == pad_id
        pool_mask = key_padding_mask.clone()
        for row, query in enumerate(is_query):
            if query:
                pool_mask[row, : min(instruct_token_len, len(all_ids[row]))] = True
        context = profiler if profiler is not None else null_profiler()
        with context:
            hidden = student(input_ids, key_padding_mask=key_padding_mask)
            emb = lal_head(
                hidden,
                key_padding_mask=key_padding_mask,
                pool_mask=pool_mask,
            )
        if profiler is not None:
            profiler.total_samples += len(batch)
        all_embs.append(emb.detach().float().cpu())
        if batch_index == 1 or batch_index % 50 == 0 or batch_index == total_batches:
            print(
                f"encoded batch {batch_index}/{total_batches} "
                f"({min(start + batch_size, len(texts))}/{len(texts)} texts)",
                flush=True,
            )
    return torch.cat(all_embs, dim=0)


class null_profiler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None


def extract_student_embeddings(
    pairs: list[dict[str, Any]],
    *,
    student: StudentDistillationModule,
    lal_head: LatentAttentionLayer,
    tokenizer,
    pad_id: int,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    query_texts: list[tuple[str, str]] = []
    target_texts: list[tuple[str, str]] = []
    ids: list[str] = []
    query_opts: list[str] = []
    target_opts: list[str] = []
    for pair in pairs:
        query_asm = pair["query"]["asm"]
        target_asm = pair["target"]["asm"]
        query_text = INSTRUCT_TEMPLATE + ("\n".join(query_asm) if isinstance(query_asm, list) else query_asm)
        target_text = "\n".join(target_asm) if isinstance(target_asm, list) else target_asm
        query_texts.append(
            (
                query_text,
                "0" * len(INSTRUCT_TEMPLATE) + "1" * (len(query_text) - len(INSTRUCT_TEMPLATE)),
            )
        )
        target_texts.append((target_text, "1" * len(target_text)))
        ids.append(str(pair["id"]))
        query_opts.append(str(pair["query"]["opt"]))
        target_opts.append(str(pair["target"]["opt"]))

    profiler = InferenceProfiler(device)
    warmup = query_texts[: min(batch_size, len(query_texts))]
    if warmup:
        print("warmup pass", flush=True)
        _ = encode_student_texts(
            warmup,
            student=student,
            lal_head=lal_head,
            tokenizer=tokenizer,
            pad_id=pad_id,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    print("encoding query embeddings", flush=True)
    query_embeddings = encode_student_texts(
        query_texts,
        student=student,
        lal_head=lal_head,
        tokenizer=tokenizer,
        pad_id=pad_id,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        profiler=profiler,
    )
    print("encoding target embeddings", flush=True)
    target_embeddings = encode_student_texts(
        target_texts,
        student=student,
        lal_head=lal_head,
        tokenizer=tokenizer,
        pad_id=pad_id,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        profiler=profiler,
    )
    return {
        "ids": ids,
        "query_opts": query_opts,
        "target_opts": target_opts,
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
        "stats": profiler.get_stats(),
    }


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    student_dir = Path(args.student_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = slug(args.eval_tag) if args.eval_tag else default_eval_tag(student_dir)
    embeddings_path = output_dir / f"eval_bench_embeddings_{tag}.pt"
    metrics_path = output_dir / f"eval_student_bench_metrics_{tag}.json"
    summary_path = output_dir / f"SUMMARY_{tag}.md"
    ensure_no_overwrite(
        [embeddings_path, metrics_path, summary_path],
        allow_overwrite=args.allow_overwrite,
    )

    base_tokenizer, nova_tokenizer, _ = setup_nova_tokenizer()
    pad_id = base_tokenizer.pad_token_id or 0
    config = json.loads(student_dir.joinpath("student_config.json").read_text(encoding="utf-8"))
    student = StudentDistillationModule(
        vocab_size=int(config["vocab_size"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        pad_id=pad_id,
    ).to(device=device, dtype=dtype)
    lal_head = LatentAttentionLayer(hidden_dim=int(config["hidden_dim"])).to(
        device=device,
        dtype=dtype,
    )
    student.load_state_dict(
        safe_torch_load(resolve_checkpoint(student_dir, "student_model.pt"), map_location="cpu")
    )
    lal_head.load_state_dict(
        safe_torch_load(resolve_checkpoint(student_dir, "lal_head.pt"), map_location="cpu")
    )
    student.eval()
    lal_head.eval()
    print(f"loaded student from {student_dir} on {device}", flush=True)

    test_samples = load_jsonl(str(args.test_path))
    eval_pairs = build_eval_pairs(test_samples, seed=args.seed)
    results = extract_student_embeddings(
        eval_pairs,
        student=student,
        lal_head=lal_head,
        tokenizer=nova_tokenizer,
        pad_id=pad_id,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    torch.save(results, embeddings_path)
    report = compute_report(results)
    report["source"] = str(embeddings_path.resolve())
    report["model"] = f"nova_student_scratch_bcsd_{tag}"
    report["student_dir"] = str(student_dir.resolve())
    report["data"] = str(Path(args.test_path).resolve())
    distilled_report_path = Path(args.distilled_report) if args.distilled_report else None
    distilled_report = load_distilled_report(distilled_report_path)
    comparison = None
    if distilled_report is not None:
        comparison = compare_to_distilled(report, distilled_report)
        report["comparison_to_distilled_bcsd"] = {
            **comparison,
            "distilled_report": str(distilled_report_path.resolve())
            if distilled_report_path is not None
            else None,
        }
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_eval_summary(
        output_dir,
        tag=tag,
        report=report,
        comparison=comparison,
        distilled_report_path=distilled_report_path,
    )
    print(f"saved embeddings to {embeddings_path}", flush=True)
    print(f"saved metrics to {metrics_path}", flush=True)
    print(f"saved summary to {summary_path}", flush=True)
    print_report_summary(report)
    if comparison is not None:
        print(comparison["verdict"], flush=True)


if __name__ == "__main__":
    main()
