import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from opt_classification.build_manifest import (
    assert_split_disjointness,
    build_manifest,
    build_manifest_rows,
)
from opt_classification.embedders import (
    MODEL_REGISTRY,
    assert_no_explicit_settings,
    list_models,
)
from opt_classification.extract import select_assemblies
from opt_classification.probe import (
    align_embeddings_to_manifest,
    load_embedding_cache,
    run_probe,
)
from opt_classification.report import _paired_deltas
from opt_classification.distill_opt_student.model import (
    build_model_from_config as build_distill_opt_model,
)
from opt_classification.distill_opt_student.train import (
    SCRATCH_OPT_BASELINE_ACC,
    lambda_mse_for_step,
)
from opt_classification.scratch_baseline.model import (
    build_model_from_config as build_scratch_opt_model,
)
from binarycorp_bench.nova_student_scratch_bcsd.eval import (
    compare_to_distilled,
    retrieval_metric_scalars,
)
from binarycorp_bench.nova_student_scratch_bcsd.train import (
    build_function_disjoint_pairs,
)


OPTIMIZATIONS = ("O0", "O1", "O2", "O3")


def quartet(function_id, compiler="gcc", architecture="x86_64", variant="none"):
    return [
        {
            "id": function_id,
            "opt": f"{compiler}_{optimization}_{architecture}_{variant}",
            "asm": "ret",
        }
        for optimization in OPTIMIZATIONS
    ]


class ManifestBuilderTest(unittest.TestCase):
    def test_retains_exact_quartets_and_splits_binary_groups(self):
        train_rows = (
            quartet("binary_a__f1")
            + quartet("binary_a__f2")
            + quartet("binary_b__f1")
            + quartet("binary_c__f1")
            + quartet("incomplete__f1")[:3]
            + quartet("duplicate__f1")
            + [quartet("duplicate__f1")[0]]
        )
        test_rows = quartet("official_test__f1")

        rows, summary = build_manifest_rows(
            train_rows, test_rows, val_fraction=0.34, seed=7
        )

        self.assertEqual(
            set(rows[0]),
            {
                "split",
                "id",
                "opt",
                "compiler",
                "optimization",
                "architecture",
                "variant",
                "binary_group",
            },
        )
        self.assertEqual(summary["sources"]["train"]["retained_quartets"], 4)
        self.assertEqual(summary["sources"]["train"]["rejected_incomplete"], 1)
        self.assertEqual(summary["sources"]["train"]["rejected_duplicate"], 1)
        self.assertEqual({row["split"] for row in rows if row["id"] == "official_test__f1"}, {"test"})

        splits_by_binary = {}
        for row in rows:
            splits_by_binary.setdefault(row["binary_group"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in splits_by_binary.values()))
        self.assertEqual(summary["splits"]["val"]["binary_groups"], 1)
        assert_split_disjointness(rows)

        repeated, _ = build_manifest_rows(
            list(reversed(train_rows)),
            list(reversed(test_rows)),
            val_fraction=0.34,
            seed=7,
        )
        self.assertEqual(rows, repeated)

    def test_smoke_cap_applies_per_official_source(self):
        train_rows = quartet("train_a__f") + quartet("train_b__f")
        test_rows = quartet("test_a__f") + quartet("test_b__f")
        rows, summary = build_manifest_rows(
            train_rows,
            test_rows,
            val_fraction=0,
            seed=2,
            max_groups=1,
        )
        self.assertEqual(summary["sources"]["train"]["selected_binary_groups"], 1)
        self.assertEqual(summary["sources"]["test"]["selected_binary_groups"], 1)
        self.assertEqual(len(rows), 8)

    def test_rejects_cross_source_binary_leakage(self):
        with self.assertRaisesRegex(AssertionError, "Binary leakage"):
            build_manifest_rows(
                quartet("shared_binary__train_function"),
                quartet("shared_binary__test_function"),
                val_fraction=0,
            )

    def test_cross_format_alignment_drops_whole_quartet(self):
        train_rows = quartet("train_a__f") + quartet("train_b__f")
        test_rows = quartet("test_a__f")
        available = {
            (row["id"], row["opt"]) for row in train_rows + test_rows
        }
        available.remove(("train_b__f", "gcc_O2_x86_64_none"))

        rows, summary = build_manifest_rows(
            train_rows,
            test_rows,
            val_fraction=0,
            alignment_key_sets=[available],
        )

        self.assertNotIn("train_b__f", {row["id"] for row in rows})
        self.assertEqual(
            summary["sources"]["train"]["rejected_cross_format_alignment"], 1
        )
        self.assertEqual(
            {row["id"] for row in rows}, {"train_a__f", "test_a__f"}
        )

    def test_writes_manifest_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            test_path = root / "test.jsonl"
            manifest_path = root / "out" / "manifest.jsonl"
            summary_path = root / "out" / "summary.json"
            train_path.write_text(
                "".join(json.dumps(row) + "\n" for row in quartet("train__f")),
                encoding="utf-8",
            )
            test_path.write_text(
                "".join(json.dumps(row) + "\n" for row in quartet("test__f")),
                encoding="utf-8",
            )

            rows, summary = build_manifest(
                train_path,
                test_path,
                manifest_path,
                summary_path,
                val_fraction=0,
            )

            self.assertEqual(len(manifest_path.read_text().splitlines()), len(rows))
            self.assertEqual(json.loads(summary_path.read_text())["seed"], 42)
            self.assertEqual(summary["splits"]["test"]["rows"], 4)


class ExtractionContractTest(unittest.TestCase):
    def test_registry_covers_table_i_and_controls(self):
        expected = {
            "clap",
            "ebm-full",
            "ebm-no-causal",
            "student",
            "jtrans",
            "joint-scratch",
            "asm2vec",
            "clap-raw",
            "nova-raw",
            "ebm-s1",
            "student-pre",
            "jtrans-bench",
        }
        self.assertEqual(set(list_models()), expected)
        self.assertTrue(all(MODEL_REGISTRY[name].checkpoint for name in expected))

    def test_rejects_explicit_settings_but_allows_architecture_evidence(self):
        opt = "gcc_O3_x86_64_none"
        assert_no_explicit_settings(["mov rax, rbx", "ret"], opt)
        with self.assertRaisesRegex(ValueError, "settings-label leakage"):
            assert_no_explicit_settings(
                ["mov rax, rbx", "o3 gcc x86_64 none"], opt
            )
        with self.assertRaisesRegex(ValueError, "settings-label leakage"):
            assert_no_explicit_settings(["optimization: O3", "ret"], opt)

    def test_select_assemblies_aligns_exact_keys_and_detects_missing(self):
        rows = [
            {"id": "binary__f", "opt": "gcc_O0_x86_64_none"},
            {"id": "binary__f", "opt": "gcc_O1_x86_64_none"},
        ]
        index = {
            ("binary__f", "gcc_O0_x86_64_none"): ["ret"],
            ("binary__f", "gcc_O1_x86_64_none"): ["nop", "ret"],
        }
        keys, assemblies = select_assemblies(rows, index)
        self.assertEqual(keys, list(index))
        self.assertEqual(assemblies[1], ["nop", "ret"])
        with self.assertRaisesRegex(KeyError, "absent"):
            select_assemblies(rows, {keys[0]: ["ret"]})


def synthetic_probe_data():
    manifest = []
    cache_rows = []
    split_functions = {
        "train": ["train_a__f", "train_b__f", "train_c__f"],
        "val": ["val_a__f", "val_b__f"],
        "test": ["test_a__f", "test_b__f", "test_c__f"],
    }
    for split, function_ids in split_functions.items():
        for function_number, function_id in enumerate(function_ids):
            for class_number, optimization in enumerate(OPTIMIZATIONS):
                opt = f"gcc_{optimization}_x86_64_none"
                manifest.append(
                    {
                        "split": split,
                        "id": function_id,
                        "opt": opt,
                        "compiler": "gcc",
                        "optimization": optimization,
                        "architecture": "x86_64",
                        "variant": "none",
                        "binary_group": function_id.rsplit("__", 1)[0],
                    }
                )
                embedding = np.zeros(6, dtype=np.float32)
                embedding[class_number] = 8.0
                embedding[4] = function_number * 0.01
                embedding[5] = 1.0
                cache_rows.append((function_id, opt, embedding))

    # Cache order deliberately differs from manifest order.
    cache_rows.reverse()
    cache = {
        "ids": [row[0] for row in cache_rows],
        "opts": [row[1] for row in cache_rows],
        "embeddings": torch.tensor(np.stack([row[2] for row in cache_rows])),
        "metadata": {"model": "synthetic", "pooling": "test"},
    }
    return manifest, cache


class ProbeTest(unittest.TestCase):
    def test_loads_schema_and_aligns_by_exact_id_opt(self):
        manifest, cache = synthetic_probe_data()
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "embeddings.pt"
            torch.save(cache, cache_path)
            loaded = load_embedding_cache(cache_path)

        aligned = align_embeddings_to_manifest(manifest, loaded)
        self.assertEqual(aligned["embeddings"].shape, (32, 6))
        self.assertEqual(aligned["labels"][0], "O0")
        self.assertEqual(loaded["metadata"]["pooling"], "test")

        broken = dict(loaded)
        broken["ids"] = loaded["ids"][:-1]
        with self.assertRaisesRegex(ValueError, "Invalid cache dimensions"):
            align_embeddings_to_manifest(manifest, broken)

    def test_four_way_probe_is_deterministic_and_reports_bootstrap(self):
        manifest, cache = synthetic_probe_data()
        kwargs = {
            "task": "4way",
            "c_grid": [0.01, 0.1, 1.0],
            "seed": 11,
            "bootstrap_samples": 30,
        }
        first = run_probe(manifest, cache, **kwargs)
        second = run_probe(manifest, cache, **kwargs)

        self.assertEqual(first, second)
        self.assertEqual(first["test_metrics"]["accuracy"], 1.0)
        self.assertEqual(first["test_metrics"]["balanced_accuracy"], 1.0)
        self.assertEqual(first["test_metrics"]["macro_f1"], 1.0)
        self.assertEqual(
            first["test_metrics"]["confusion_matrix"],
            (np.eye(4, dtype=int) * 3).tolist(),
        )
        self.assertEqual(first["bootstrap_95"]["clusters"], 3)
        self.assertIn("accuracy", first["bootstrap_95"]["metrics"])
        self.assertIn("test_metrics", first["shuffled_label_sanity"])
        self.assertEqual(len(first["test_predictions"]["ids"]), 12)
        self.assertIsNone(first["test_predictions"]["positive_scores"])

    def test_binary_probe_reports_roc_auc_and_class_recalls(self):
        manifest, cache = synthetic_probe_data()
        result = run_probe(
            manifest,
            cache,
            task="o0-vs-o3",
            c_grid=[0.1, 1.0],
            seed=3,
            bootstrap_samples=20,
        )

        self.assertEqual(result["class_labels"], ["O0", "O3"])
        self.assertEqual(result["test_metrics"]["roc_auc"], 1.0)
        self.assertEqual(
            set(result["test_metrics"]["per_class_recall"]), {"O0", "O3"}
        )
        self.assertIn("roc_auc", result["bootstrap_95"]["metrics"])
        self.assertEqual(len(result["test_predictions"]["positive_scores"]), 6)

    def test_paired_delta_bootstrap_uses_aligned_predictions(self):
        manifest, cache = synthetic_probe_data()
        result = run_probe(
            manifest,
            cache,
            task="4way",
            c_grid=[0.1],
            seed=5,
            bootstrap_samples=10,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ("clap-raw", "clap"):
                model_dir = root / model
                model_dir.mkdir()
                (model_dir / "4way.json").write_text(json.dumps(result))
            deltas = _paired_deltas(root, seed=5)

        self.assertEqual(len(deltas), 3)
        self.assertTrue(
            all(row["delta_after_minus_before"] == 0.0 for row in deltas)
        )
        self.assertTrue(all(row["delta_ci_lower"] == 0.0 for row in deltas))
        self.assertTrue(all(row["delta_ci_upper"] == 0.0 for row in deltas))


class StudentAblationContractTest(unittest.TestCase):
    def test_distilled_opt_model_matches_scratch_architecture(self):
        config = {"vocab_size": 64, "hidden_dim": 32, "num_layers": 1}
        scratch = build_scratch_opt_model(config, pad_id=0, num_latents=4)
        distilled = build_distill_opt_model(config, pad_id=0, num_latents=4)

        self.assertEqual(
            scratch.parameter_counts().as_dict(),
            distilled.parameter_counts().as_dict(),
        )
        input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
        logits, hidden = distilled(
            input_ids,
            key_padding_mask=input_ids.eq(0),
            return_hidden=True,
        )
        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(hidden.shape, (2, 3, 32))

    def test_distilled_opt_lambda_schedule_reaches_endpoints(self):
        self.assertEqual(SCRATCH_OPT_BASELINE_ACC, 0.612)
        self.assertAlmostEqual(
            lambda_mse_for_step(0, 10, start=1.0, end=0.05),
            1.0,
        )
        self.assertAlmostEqual(
            lambda_mse_for_step(9, 10, start=1.0, end=0.05),
            0.05,
        )
        middle = lambda_mse_for_step(5, 10, start=1.0, end=0.05)
        self.assertGreater(middle, 0.05)
        self.assertLess(middle, 1.0)

    def test_pure_bcsd_pairs_are_function_disjoint(self):
        samples = []
        for function_id in ("a", "b", "c", "d"):
            samples.extend(
                [
                    {"id": function_id, "opt": "gcc_O0_x86_64_none", "asm": "ret"},
                    {"id": function_id, "opt": "gcc_O3_x86_64_none", "asm": "nop"},
                ]
            )
        train_pairs, val_pairs, metadata = build_function_disjoint_pairs(
            samples,
            val_split=0.25,
            seed=4,
            max_train_pairs=None,
            max_val_pairs=None,
        )
        self.assertTrue(train_pairs)
        self.assertTrue(val_pairs)
        self.assertTrue(
            {pair[2] for pair in train_pairs}.isdisjoint({pair[2] for pair in val_pairs})
        )
        self.assertTrue(metadata["function_disjoint_validation"])

    def test_pure_bcsd_comparison_verdict_tracks_full_recall(self):
        pure = {
            "pooled_recall": {"Recall@1_Pool50": 0.8},
            "full_recall_one_target_per_function": {
                "Recall@1": {"recall": 0.50},
                "Recall@5": {"recall": 0.70},
                "Recall@10": {"recall": 0.75},
            },
        }
        distilled = {
            "pooled_recall": {"Recall@1_Pool50": 0.82},
            "full_recall_one_target_per_function": {
                "Recall@1": {"recall": 0.51},
                "Recall@5": {"recall": 0.71},
                "Recall@10": {"recall": 0.76},
            },
        }
        self.assertEqual(retrieval_metric_scalars(pure)["Full R@1"], 0.50)
        comparison = compare_to_distilled(pure, distilled)
        self.assertIn("does not hurt BCSD", comparison["verdict"])
        full_r1 = next(
            row for row in comparison["rows"] if row["metric"] == "Full R@1"
        )
        self.assertAlmostEqual(full_r1["distilled_minus_pure"], 0.01)


if __name__ == "__main__":
    unittest.main()
