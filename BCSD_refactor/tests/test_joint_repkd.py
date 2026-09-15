import unittest

import numpy as np
import torch
import torch.nn as nn

from joint_repkd.balance import GradientBalance, teacher_lambda
from joint_repkd.data import RepKDCollator
from joint_repkd.losses import hidden_state_distillation_loss
from joint_repkd.train import compute_stage1, compute_stage2
from joint_distill.data import JointPair
from joint_scratch.config import ScratchModelConfig
from joint_scratch.model import ScratchJointModel


class FakeBaseTokenizer:
    pad_token_id = 0
    mask_token_id = 63

    def encode(self, text):
        # One token per character plus a leading BOS token, mirroring Nova.
        return [1] + [(ord(char) % 61) + 2 for char in text]


class FakeNovaTokenizer:
    def __init__(self):
        self.tokenizer = FakeBaseTokenizer()

    def encode(self, input_text, output_text, char_types):
        full = input_text + output_text
        if len(full) != len(char_types):
            raise AssertionError("char type alignment failed")
        input_ids = []
        labels = []
        for text, is_output in ((input_text, False), (output_text, True)):
            ids = self.tokenizer.encode(text)[1:] if text else []
            input_ids += ids
            labels += ids if is_output else [-100] * len(ids)
        length = len(input_ids)
        mask = np.tril(np.ones((length, length), dtype=np.bool_))
        return {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "labels": np.asarray(labels, dtype=np.int64),
            "nova_attention_mask": mask,
        }


def tiny_config():
    return ScratchModelConfig(
        vocab_size=64,
        pad_id=0,
        d_model=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
        max_length=128,
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


class FakeTeacher:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

    def hidden_states(self, input_ids, attention_mask, nova_attention_mask):
        return torch.zeros(
            input_ids.shape[0], input_ids.shape[1], self.hidden_dim
        )


class HiddenStateKDLossTest(unittest.TestCase):
    def test_exact_value_without_projection(self):
        student = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        teacher = torch.tensor([[[2.0, 4.0], [0.0, 0.0]]])
        mask = torch.tensor([[True, False]])
        loss = hidden_state_distillation_loss(student, teacher, mask, None)
        # Position 0 squared diff = (1-2)^2 + (2-4)^2 = 5; masked out at pos 1.
        # Normalize by N * d_T = 1 * 2 -> 2.5.
        self.assertAlmostEqual(float(loss.item()), 2.5, places=6)

    def test_projection_maps_dims_and_flows_grad(self):
        torch.manual_seed(0)
        student = torch.randn(2, 3, 4, requires_grad=True)
        teacher = torch.randn(2, 3, 8)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        projection = nn.Linear(4, 8)
        loss = hidden_state_distillation_loss(student, teacher, mask, projection)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNotNone(projection.weight.grad)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            hidden_state_distillation_loss(
                torch.zeros(2, 3, 4), torch.zeros(2, 4, 4), torch.ones(2, 3, dtype=torch.bool)
            )


class GradientBalanceTest(unittest.TestCase):
    def test_ratio_is_gt_norm_over_kd_norm(self):
        weight = torch.tensor([2.0], requires_grad=True)
        gt = (weight * weight).sum()  # grad = 2w = 4
        kd = ((weight - 1) ** 2).sum()  # grad = 2(w-1) = 2
        balance = GradientBalance(beta=0.0, epsilon=0.0)
        scale = balance.update("C", gt, kd, [weight])
        self.assertAlmostEqual(scale, 2.0, places=6)

    def test_ema_smooths_updates(self):
        weight = torch.tensor([3.0], requires_grad=True)
        gt = (weight * weight).sum()  # grad 6
        kd = weight.sum()  # grad 1
        balance = GradientBalance(beta=0.5, epsilon=0.0)
        first = balance.update("B", gt, kd, [weight])  # 0.5*1 + 0.5*6 = 3.5
        second = balance.update("B", gt, kd, [weight])  # 0.5*3.5 + 0.5*6 = 4.75
        self.assertAlmostEqual(first, 3.5, places=6)
        self.assertAlmostEqual(second, 4.75, places=6)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            GradientBalance().update("X", torch.tensor(1.0), torch.tensor(1.0), [])


class TeacherLambdaTest(unittest.TestCase):
    def test_teacher_first_anneal(self):
        self.assertAlmostEqual(
            teacher_lambda(0, 100, lambda_high=2.0, lambda_low=0.1, anneal_fraction=0.5),
            2.0,
        )
        self.assertAlmostEqual(
            teacher_lambda(49, 100, lambda_high=2.0, lambda_low=0.1, anneal_fraction=0.5),
            0.138,
            places=5,
        )
        self.assertAlmostEqual(
            teacher_lambda(50, 100, lambda_high=2.0, lambda_low=0.1, anneal_fraction=0.5),
            0.1,
        )
        self.assertAlmostEqual(
            teacher_lambda(90, 100, lambda_high=2.0, lambda_low=0.1, anneal_fraction=0.5),
            0.1,
        )

    def test_invalid_bounds_rejected(self):
        with self.assertRaises(ValueError):
            teacher_lambda(0, 100, lambda_high=0.0)
        with self.assertRaises(ValueError):
            teacher_lambda(0, 100, anneal_fraction=0.0)
        with self.assertRaises(ValueError):
            teacher_lambda(0, 100, anneal_fraction=1.0)


class RepKDCollatorTest(unittest.TestCase):
    def test_stage1_masks_only_assembly_tokens(self):
        collator = RepKDCollator(FakeNovaTokenizer(), max_length=128, mlm_probability=0.5)
        batch = collator([pair("bin_a__f", 1)])
        stage1 = batch["stage1"]
        self.assertEqual(stage1["input_ids"].shape[0], 1)
        self.assertEqual(stage1["labels"].shape, stage1["input_ids"].shape)
        self.assertEqual(stage1["nova_attention_mask"].ndim, 3)
        self.assertEqual(stage1["assembly_mask"].shape, stage1["input_ids"].shape)

        # prefix (type 0), source (type 1), suffix (type 0), target (type 1)
        prefix_len = len("Translate assembly from O0 to O3.\nSource:\n")
        source_len = len("source assembly text")
        suffix_len = len("\nTarget:\n")
        assembly = stage1["assembly_mask"][0].numpy()
        self.assertTrue(assembly[:prefix_len].sum() == 0)
        self.assertTrue(assembly[prefix_len : prefix_len + source_len].all())
        self.assertTrue(
            assembly[prefix_len + source_len : prefix_len + source_len + suffix_len].sum() == 0
        )
        self.assertTrue(assembly[prefix_len + source_len + suffix_len :].all())

        # Every target token must be an assembly token for KD.
        target_mask = stage1["labels"][0].ne(-100)
        self.assertTrue((stage1["assembly_mask"][0] & target_mask).eq(target_mask).all())

    def test_stage2_masks_only_labeled_tokens(self):
        collator = RepKDCollator(FakeNovaTokenizer(), max_length=128, mlm_probability=1.0)
        batch = collator([pair("bin_a__f", 1), pair("bin_b__f", 2)])
        stage2 = batch["stage2"]
        self.assertEqual(stage2["input_ids"].shape[0], 4)
        self.assertEqual(stage2["teacher_input_ids"].shape, stage2["input_ids"].shape)
        self.assertEqual(stage2["labels"].shape, stage2["input_ids"].shape)
        self.assertEqual(stage2["kd_mask"].shape, stage2["input_ids"].shape)
        self.assertEqual(stage2["teacher_nova_attention_mask"].ndim, 3)

        masked = stage2["labels"].ne(-100)
        self.assertTrue(masked.any())
        self.assertTrue(stage2["input_ids"].masked_select(masked).eq(63).all())
        self.assertTrue(stage2["kd_mask"].eq(masked).all())
        # Teacher consumes clean tokens, never the mask id.
        self.assertTrue(stage2["teacher_input_ids"].ne(63).all())


class EndToEndComputeTest(unittest.TestCase):
    def test_stage1_and_stage2_backward_with_balance(self):
        collator = RepKDCollator(FakeNovaTokenizer(), max_length=128, mlm_probability=0.5)
        batch = collator([pair("bin_a__f", 1), pair("bin_b__f", 2)])
        model = ScratchJointModel(tiny_config())
        teacher = FakeTeacher(hidden_dim=8)
        proj_c = nn.Linear(16, 8)
        proj_b = nn.Linear(16, 8)
        balance = GradientBalance(beta=0.9, epsilon=1e-6)
        parameters = list(model.parameters())

        device = torch.device("cpu")
        for projection, mode, compute in (
            (proj_c, "C", compute_stage1),
            (proj_b, "B", compute_stage2),
        ):
            result = compute(model, teacher, projection, batch, device, amp=False)
            self.assertTrue(torch.isfinite(result.gt_loss))
            self.assertTrue(torch.isfinite(result.kd_loss))
            scale = balance.update(mode, result.gt_loss, result.kd_loss, parameters)
            self.assertGreater(scale, 0.0)
            total = result.gt_loss + 0.5 * scale * result.kd_loss
            total.backward()
            grad = sum(p.grad.abs().sum().item() for p in parameters if p.grad is not None)
            self.assertGreater(grad, 0.0)
            model.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main()
