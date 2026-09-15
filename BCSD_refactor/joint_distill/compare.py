#!/usr/bin/env python3
"""Compare the joint student with EBM-full and CLAP retrieval reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from joint_distill.config import StudentConfig, estimate_parameter_count

REPO_ROOT = Path(__file__).resolve().parent.parent
BCSD_ROOT = REPO_ROOT.parent
LOCAL_CLAP_CORE_PARAMETERS = 110_704_896


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-retrieval", required=True)
    parser.add_argument(
        "--teacher-retrieval",
        default=str(
            BCSD_ROOT
            / "model_checkpoints"
            / "nova_ebm_bench"
            / "eval_bench_report.json"
        ),
    )
    parser.add_argument(
        "--clap-retrieval",
        default=str(
            REPO_ROOT
            / "binarycorp_bench"
            / "clap"
            / "eval_clap_bench_final_finetuned_report.json"
        ),
    )
    parser.add_argument("--student-translation", default=None)
    parser.add_argument("--teacher-translation", default=None)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_json(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def recall_one(report: Dict) -> float:
    return float(report["full_recall_one_target_per_function"]["Recall@1"]["recall"])


def pool500(report: Dict) -> float:
    return float(report["pooled_recall"]["Recall@1_Pool500"])


def optional_translation(path: Optional[str]) -> Optional[Dict]:
    return load_json(path) if path else None


def main() -> None:
    args = parse_args()
    student = load_json(args.student_retrieval)
    teacher = load_json(args.teacher_retrieval)
    clap = load_json(args.clap_retrieval)
    config = StudentConfig.from_dict(
        load_json(Path(args.student_checkpoint) / "student_config.json")
    )

    recalls = {
        "joint_student": recall_one(student),
        "ebm_full_teacher": recall_one(teacher),
        "clap": recall_one(clap),
    }
    pool_recalls = {
        "joint_student": pool500(student),
        "ebm_full_teacher": pool500(teacher),
        "clap": pool500(clap),
    }
    student_translation = optional_translation(args.student_translation)
    teacher_translation = optional_translation(args.teacher_translation)

    if recalls["joint_student"] >= max(
        recalls["ebm_full_teacher"], recalls["clap"]
    ):
        retrieval_outcome = "beats_or_matches_teacher_and_clap"
    elif recalls["joint_student"] >= recalls["clap"]:
        retrieval_outcome = "beats_clap_but_not_teacher"
    else:
        retrieval_outcome = "retrieval_under_target"

    translation_outcome = "teacher_baseline_not_supplied"
    if student_translation and teacher_translation:
        student_ce = float(student_translation["cross_entropy"])
        teacher_ce = float(teacher_translation["cross_entropy"])
        translation_outcome = (
            "matches_or_beats_teacher"
            if student_ce <= teacher_ce
            else "below_teacher"
        )

    student_parameters = estimate_parameter_count(config)
    size_ratio = student_parameters / LOCAL_CLAP_CORE_PARAMETERS
    summary = {
        "student_parameters": student_parameters,
        "local_clap_core_parameters_estimate": LOCAL_CLAP_CORE_PARAMETERS,
        "student_to_clap_parameter_ratio": size_ratio,
        "clap_size_class": 0.75 <= size_ratio <= 1.25,
        "global_recall_at_1": recalls,
        "pool500_recall_at_1": pool_recalls,
        "retrieval_outcome": retrieval_outcome,
        "translation_outcome": translation_outcome,
        "student_translation": student_translation,
        "teacher_translation": teacher_translation,
        "note": (
            "Translation metrics are assembly-to-assembly and are not "
            "HumanEval-Decompile Pass@k."
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
