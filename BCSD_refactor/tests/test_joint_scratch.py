import tempfile
import unittest

import numpy as np
import torch

from joint_scratch.config import ScratchModelConfig, estimate_parameter_count
from joint_scratch.data import (
    JointPair,
    JointScratchCollator,
    build_variant_pairs,
    split_group,
    split_pairs,
)
from joint_scratch.losses import retrieval_recall_at_1, shifted_causal_cross_entropy
from joint_scratch.model import ScratchJointModel
from joint_scratch.train import LossWeights, compute_batch
from opt_classification.joint_scratch_head.model import JointScratchOptClassifier
from shared.losses import contrastive_loss_positive_aware


class FakeBaseTokenizer:
    pad_token_id = 0
    mask_token_id = 63


class FakeNovaTokenizer:
    def __init__(self):
        self.tokenizer = FakeBaseTokenizer()
        self.seen_lengths = []

    def encode(self, input_text, output_text, char_types):
        text = input_text + output_text
        self.seen_lengths.append((len(input_text), len(output_text), len(text)))
        if len(text) != len(char_types):
            raise AssertionError("char type alignment failed")
        ids = np.asarray([(ord(char) % 62) + 1 for char in text], dtype=np.int64)
        labels = np.full(len(text), -100, dtype=np.int64)
        labels[len(input_text) :] = ids[len(input_text) :]
        return {"input_ids": ids, "labels": labels}


def tiny_config():
    return ScratchModelConfig(
        vocab_size=64,
        pad_id=0,
        d_model=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
        max_length=32,
        embedding_dim=8,
    )


def pair(function_id="bin__f", numeric_id=0):
    return JointPair(
        function_id=function_id,
        numeric_id=numeric_id,
        source_asm="source assembly text",
        target_asm="target assembly text",
        source_opt="O0",
        target_opt="O3",
    )


class ScratchModelTest(unittest.TestCase):
    def test_parameter_count_and_default_size(self):
        tiny = ScratchJointModel(tiny_config())
        self.assertEqual(
            sum(parameter.numel() for parameter in tiny.parameters()),
            estimate_parameter_count(tiny_config()),
        )
        count = estimate_parameter_count(ScratchModelConfig())
        self.assertEqual(count, 97_641_984)
        self.assertGreater(count, 90_000_000)
        self.assertLess(count, 115_000_000)

    def test_task_modes_keep_token_alignment(self):
        model = ScratchJointModel(tiny_config()).eval()
        ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
        attention = ids.ne(0)
        lm = model.forward_lm(ids, attention)
        mlm = model.forward_mlm(ids, attention)
        retrieval = model.forward_retrieval(ids, attention, attention)
        self.assertEqual(lm.hidden_states.shape, (2, 4, 16))
        self.assertEqual(lm.logits.shape, (2, 4, 64))
        self.assertEqual(mlm.hidden_states.shape, (2, 4, 16))
        self.assertEqual(mlm.logits.shape, (2, 4, 64))
        self.assertEqual(retrieval.hidden_states.shape, (2, 4, 16))
        self.assertEqual(retrieval.embeddings.shape, (2, 8))

    def test_causal_mode_cannot_observe_future_change(self):
        torch.manual_seed(3)
        model = ScratchJointModel(tiny_config()).eval()
        left = torch.tensor([[1, 2, 3, 4]])
        right = torch.tensor([[1, 2, 9, 4]])
        attention = torch.ones_like(left, dtype=torch.bool)
        left_logits = model.forward_lm(left, attention).logits
        right_logits = model.forward_lm(right, attention).logits
        self.assertTrue(torch.allclose(left_logits[:, :2], right_logits[:, :2], atol=1e-6))

    def test_checkpoint_round_trip(self):
        torch.manual_seed(5)
        model = ScratchJointModel(tiny_config()).eval()
        ids = torch.tensor([[1, 2, 3]])
        expected = model.forward_lm(ids).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = ScratchJointModel.from_pretrained(directory).eval()
            actual = restored.forward_lm(ids).logits
        self.assertTrue(torch.equal(expected, actual))
        self.assertFalse(hasattr(model, "initialize_token_embeddings_from_teacher"))

    def test_freeze_backbone_keeps_only_retrieval_head_trainable(self):
        model = ScratchJointModel(tiny_config())
        model.freeze_backbone()
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(any(name.startswith("pool_attention") for name in trainable))
        self.assertTrue(any(name.startswith("retrieval_projection") for name in trainable))
        self.assertFalse(any(name.startswith("token_embedding") for name in trainable))
        self.assertFalse(any(name.startswith("position_embedding") for name in trainable))
        self.assertFalse(any(name.startswith("encoder") for name in trainable))
        self.assertFalse(model.task_embeddings.requires_grad)

        model.freeze_backbone(train_task_embedding=True)
        self.assertTrue(model.task_embeddings.requires_grad)

    def test_pool_hidden_matches_pool_without_projection(self):
        model = ScratchJointModel(tiny_config()).eval()
        ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
        attention = ids.ne(0)
        hidden = model.forward_retrieval(ids, attention, attention).hidden_states
        pooled = model.pool_hidden(hidden, attention)
        projected = model.retrieval_projection(pooled)
        self.assertEqual(pooled.shape, (2, 16))
        self.assertTrue(torch.allclose(projected, model.pool(hidden, attention), atol=1e-6))

    def test_opt_classifier_freezes_backbone_and_returns_logits(self):
        classifier = JointScratchOptClassifier(ScratchJointModel(tiny_config()), num_classes=4)
        ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
        logits = classifier(ids)
        self.assertEqual(logits.shape, (2, 4))
        trainable = {name for name, p in classifier.named_parameters() if p.requires_grad}
        self.assertIn("opt_classifier.weight", trainable)
        self.assertIn("opt_classifier.bias", trainable)
        self.assertTrue(any(name.startswith("backbone.pool_attention") for name in trainable))
        self.assertFalse(any(name.startswith("backbone.encoder") for name in trainable))
        self.assertFalse(any(name.startswith("backbone.token_embedding") for name in trainable))
        self.assertFalse(any(name.startswith("backbone.retrieval_projection") for name in trainable))


class ScratchDataTest(unittest.TestCase):
    def test_all_variant_pairs_and_binary_disjoint_split(self):
        samples = []
        for binary in ("bin_a", "bin_b", "bin_c", "bin_d"):
            for suffix in ("f1", "f2"):
                fid = f"{binary}__{suffix}"
                samples.extend(
                    [
                        {"id": fid, "opt": "O0", "asm": fid + "_0"},
                        {"id": fid, "opt": "O1", "asm": fid + "_1"},
                        {"id": fid, "opt": "O3", "asm": fid + "_3"},
                    ]
                )
        pairs = build_variant_pairs(samples, seed=3)
        self.assertEqual(len(pairs), 24)
        self.assertEqual(len({item.source_asm for item in pairs}), 24)
        self.assertTrue(all(item.source_asm != item.target_asm for item in pairs))
        train, val = split_pairs(pairs, val_fraction=0.25, seed=7, unit="binary")
        train_groups = {split_group(item.function_id, "binary") for item in train}
        val_groups = {split_group(item.function_id, "binary") for item in val}
        self.assertFalse(train_groups.intersection(val_groups))

    def test_nova_label_tokens_are_rejected(self):
        samples = [
            {"id": "bin_a__f", "opt": "gcc_O0_x86_64_none", "asm": "mov eax, ebx\n<label-1>"},
            {"id": "bin_a__f", "opt": "gcc_O3_x86_64_none", "asm": "ret"},
        ]
        with self.assertRaisesRegex(ValueError, "Nova-style <label-N>"):
            build_variant_pairs(samples, seed=3)

    def test_collator_reserves_target_tokens_when_truncated(self):
        collator = JointScratchCollator(
            FakeNovaTokenizer(), max_length=20, lm_target_fraction=0.5
        )
        batch = collator([pair("bin_a__f", 1), pair("bin_b__f", 2)])
        self.assertLessEqual(batch["lm"]["input_ids"].shape[1], 20)
        self.assertTrue(batch["lm"]["labels"].ne(-100).any(dim=1).all())
        self.assertEqual(batch["retrieval"]["input_ids"].shape[0], 4)
        self.assertEqual(batch["retrieval"]["func_ids"].tolist(), [1, 1, 2, 2])
        self.assertTrue(batch["retrieval"]["pool_mask"].any(dim=1).all())
        self.assertEqual(batch["mlm"]["input_ids"].shape[0], 4)
        self.assertTrue(batch["mlm"]["labels"].ne(-100).any(dim=1).all())

    def test_mlm_collator_masks_only_labeled_tokens(self):
        collator = JointScratchCollator(
            FakeNovaTokenizer(),
            max_length=20,
            lm_target_fraction=0.5,
            mlm_probability=1.0,
        )
        batch = collator([pair("bin_a__f", 1)])
        mlm = batch["mlm"]
        masked = mlm["labels"].ne(-100)
        self.assertTrue(masked.any())
        self.assertTrue(mlm["input_ids"].masked_select(masked).eq(63).all())

    def test_collator_clips_raw_assembly_before_tokenization(self):
        tokenizer = FakeNovaTokenizer()
        collator = JointScratchCollator(
            tokenizer,
            max_length=20,
            lm_target_fraction=0.5,
            raw_chars_per_token=2,
        )
        long_pair = JointPair(
            function_id="bin_a__f",
            numeric_id=1,
            source_asm="\n".join(f"src_{idx:04d} mov eax, ebx" for idx in range(1000)),
            target_asm="\n".join(f"tgt_{idx:04d} add eax, 1" for idx in range(1000)),
            source_opt="O0",
            target_opt="O3",
        )
        batch = collator([long_pair])
        self.assertLessEqual(batch["lm"]["input_ids"].shape[1], 20)
        self.assertLess(max(total for _, _, total in tokenizer.seen_lengths), 500)


class ScratchObjectiveTest(unittest.TestCase):
    def test_joint_optimizer_step_reaches_both_task_tokens(self):
        collator = JointScratchCollator(
            FakeNovaTokenizer(), max_length=24, lm_target_fraction=0.5
        )
        batch = collator([pair("bin_a__f", 1), pair("bin_b__f", 2)])
        model = ScratchJointModel(tiny_config())
        result = compute_batch(
            model,
            batch,
            torch.device("cpu"),
            weights=LossWeights(),
            contrastive_temperature=0.05,
            amp=False,
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertIn("lm_ce", result.metrics)
        self.assertIn("retrieval_contrastive", result.metrics)
        result.loss.backward()
        self.assertGreater(model.task_embeddings.grad[0].abs().sum().item(), 0.0)
        self.assertGreater(model.task_embeddings.grad[1].abs().sum().item(), 0.0)

    def test_alternate_lm_step_reaches_only_lm_task_token(self):
        collator = JointScratchCollator(
            FakeNovaTokenizer(), max_length=24, lm_target_fraction=0.5
        )
        batch = collator([pair("bin_a__f", 1), pair("bin_b__f", 2)])
        model = ScratchJointModel(tiny_config())
        result = compute_batch(
            model,
            batch,
            torch.device("cpu"),
            weights=LossWeights(),
            contrastive_temperature=0.05,
            amp=False,
            task="lm",
        )
        self.assertIn("lm_ce", result.metrics)
        self.assertNotIn("retrieval_contrastive", result.metrics)
        result.loss.backward()
        self.assertGreater(model.task_embeddings.grad[0].abs().sum().item(), 0.0)
        self.assertEqual(model.task_embeddings.grad[1].abs().sum().item(), 0.0)

    def test_alternate_retrieval_step_reaches_only_retrieval_task_token(self):
        collator = JointScratchCollator(
            FakeNovaTokenizer(), max_length=24, lm_target_fraction=0.5
        )
        batch = collator([pair("bin_a__f", 1), pair("bin_b__f", 2)])
        model = ScratchJointModel(tiny_config())
        result = compute_batch(
            model,
            batch,
            torch.device("cpu"),
            weights=LossWeights(),
            contrastive_temperature=0.05,
            amp=False,
            task="retrieval",
        )
        self.assertNotIn("lm_ce", result.metrics)
        self.assertIn("retrieval_contrastive", result.metrics)
        result.loss.backward()
        self.assertEqual(model.task_embeddings.grad[0].abs().sum().item(), 0.0)
        self.assertGreater(model.task_embeddings.grad[1].abs().sum().item(), 0.0)

    def test_alternate_mlm_step_reaches_encoder_task_token(self):
        collator = JointScratchCollator(
            FakeNovaTokenizer(),
            max_length=24,
            lm_target_fraction=0.5,
            mlm_probability=1.0,
        )
        batch = collator([pair("bin_a__f", 1), pair("bin_b__f", 2)])
        model = ScratchJointModel(tiny_config())
        result = compute_batch(
            model,
            batch,
            torch.device("cpu"),
            weights=LossWeights(),
            contrastive_temperature=0.05,
            amp=False,
            task="mlm",
        )
        self.assertNotIn("lm_ce", result.metrics)
        self.assertIn("mlm_ce", result.metrics)
        result.loss.backward()
        self.assertEqual(model.task_embeddings.grad[0].abs().sum().item(), 0.0)
        self.assertGreater(model.task_embeddings.grad[1].abs().sum().item(), 0.0)

    def test_losses_and_global_recall(self):
        logits = torch.randn(2, 5, 11, requires_grad=True)
        labels = torch.tensor([[-100, -100, 3, 4, 5], [-100, 2, 1, 7, 8]])
        loss = shifted_causal_cross_entropy(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        ids = torch.tensor([10, 20])
        self.assertEqual(retrieval_recall_at_1(queries, targets, ids, ids), 1.0)

    def test_positive_aware_contrastive_accepts_bfloat16_embeddings(self):
        embeddings = torch.randn(4, 8, dtype=torch.bfloat16).requires_grad_(True)
        func_ids = torch.tensor([1, 1, 2, 2])
        loss = contrastive_loss_positive_aware(
            embeddings, func_ids, temperature=0.05
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(embeddings.grad)


if __name__ == "__main__":
    unittest.main()
