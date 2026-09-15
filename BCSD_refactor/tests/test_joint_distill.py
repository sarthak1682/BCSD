import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from joint_distill.config import StudentConfig, estimate_parameter_count
from joint_distill.data import (
    JointDistillCollator,
    JointPair,
    build_joint_pairs,
    split_function_ids,
)
from joint_distill.losses import (
    embedding_distillation_loss,
    shifted_causal_cross_entropy,
    topk_teacher_kl,
)
from joint_distill.model import TaskConditionedJointStudent


class FakeBaseTokenizer:
    pad_token_id = 0


class FakeNovaTokenizer:
    def __init__(self):
        self.tokenizer = FakeBaseTokenizer()
        self.labels = {29}

    def encode(self, input_text, output_text, char_types):
        text = input_text + output_text
        if len(text) != len(char_types):
            raise AssertionError("char type alignment failed")
        ids = np.asarray([(ord(char) % 28) + 1 for char in text], dtype=np.int64)
        labels = np.full(len(text), -100, dtype=np.int64)
        labels[len(input_text) :] = ids[len(input_text) :]
        return {
            "input_ids": ids,
            "labels": labels,
            "nova_attention_mask": np.tril(
                np.ones((len(text), len(text)), dtype=np.bool_)
            ),
            "no_mask_idx": [],
        }


def tiny_config():
    return StudentConfig(
        vocab_size=32,
        pad_id=0,
        d_model=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
        max_length=32,
        teacher_embedding_dim=8,
    )


class JointStudentTest(unittest.TestCase):
    def test_parameter_estimator_and_default_size_class(self):
        tiny = TaskConditionedJointStudent(tiny_config())
        actual = sum(parameter.numel() for parameter in tiny.parameters())
        self.assertEqual(actual, estimate_parameter_count(tiny_config()))
        default_count = estimate_parameter_count(StudentConfig())
        self.assertGreater(default_count, 90_000_000)
        self.assertLess(default_count, 115_000_000)

    def test_task_prefix_keeps_public_token_alignment(self):
        model = TaskConditionedJointStudent(tiny_config()).eval()
        ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
        attention = ids.ne(0)
        pool = attention.clone()
        lm = model.forward_lm(ids, attention)
        pred = model.forward_pred(ids, attention, pool)
        self.assertEqual(lm.hidden_states.shape, (2, 4, 16))
        self.assertEqual(lm.logits.shape, (2, 4, 32))
        self.assertEqual(pred.hidden_states.shape, (2, 4, 16))
        self.assertEqual(pred.embeddings.shape, (2, 8))

    def test_causal_mode_cannot_observe_a_changed_future_token(self):
        torch.manual_seed(3)
        model = TaskConditionedJointStudent(tiny_config()).eval()
        left = torch.tensor([[1, 2, 3, 4]])
        right = torch.tensor([[1, 2, 9, 4]])
        mask = torch.ones_like(left, dtype=torch.bool)
        left_logits = model.forward_lm(left, mask).logits
        right_logits = model.forward_lm(right, mask).logits
        self.assertTrue(
            torch.allclose(left_logits[:, :2], right_logits[:, :2], atol=1e-6)
        )

    def test_both_task_embeddings_receive_gradient(self):
        model = TaskConditionedJointStudent(tiny_config())
        ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        attention = torch.ones_like(ids, dtype=torch.bool)
        pool = attention.clone()
        lm = model.forward_lm(ids, attention).logits.mean()
        pred = model.forward_pred(ids, attention, pool).embeddings.mean()
        (lm + pred).backward()
        self.assertGreater(model.task_embeddings.grad[0].abs().sum().item(), 0.0)
        self.assertGreater(model.task_embeddings.grad[1].abs().sum().item(), 0.0)

    def test_gradient_checkpointed_shared_encoder_backpropagates(self):
        model = TaskConditionedJointStudent(tiny_config())
        model.gradient_checkpointing_enable()
        ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        attention = torch.ones_like(ids, dtype=torch.bool)
        pool = attention.clone()
        loss = model.forward_lm(ids, attention).logits.mean()
        loss = loss + model.forward_pred(ids, attention, pool).embeddings.mean()
        loss.backward()
        self.assertIsNotNone(model.encoder.layers[0].self_attn.in_proj_weight.grad)
        self.assertGreater(
            model.encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum().item(),
            0.0,
        )

    def test_attention_pool_ignores_excluded_values(self):
        model = TaskConditionedJointStudent(tiny_config()).eval()
        hidden = torch.randn(1, 3, 16)
        mask = torch.tensor([[True, False, True]])
        expected = model.pool(hidden, mask)
        changed = hidden.clone()
        changed[:, 1] = 1_000_000
        actual = model.pool(changed, mask)
        self.assertTrue(torch.allclose(expected, actual))

    def test_checkpoint_round_trip(self):
        torch.manual_seed(2)
        model = TaskConditionedJointStudent(tiny_config()).eval()
        ids = torch.tensor([[1, 2, 3]])
        attention = torch.ones_like(ids, dtype=torch.bool)
        expected = model.forward_lm(ids, attention).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = TaskConditionedJointStudent.from_pretrained(directory).eval()
            actual = restored.forward_lm(ids, attention).logits
        self.assertTrue(torch.equal(expected, actual))


class JointLossTest(unittest.TestCase):
    def test_all_losses_are_finite_and_differentiable(self):
        logits = torch.randn(2, 5, 11, requires_grad=True)
        labels = torch.tensor([[-100, -100, 3, 4, 5], [-100, 2, 1, 7, 8]])
        teacher_values, teacher_indices = torch.topk(
            torch.randn(2, 4, 11), 4, dim=-1
        )
        ce = shifted_causal_cross_entropy(logits, labels)
        kd = topk_teacher_kl(
            logits, labels, teacher_indices, teacher_values, temperature=2.0
        )
        student_embedding = torch.randn(4, 8, requires_grad=True)
        teacher_embedding = torch.randn(4, 8)
        embedding = embedding_distillation_loss(student_embedding, teacher_embedding)
        total = ce + kd + embedding
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(student_embedding.grad)


class JointDataTest(unittest.TestCase):
    def test_function_split_is_disjoint_and_pairs_share_identity(self):
        samples = []
        for function in ("a", "b", "c", "d"):
            samples.extend(
                [
                    {"id": function, "opt": "O0", "asm": function + "0"},
                    {"id": function, "opt": "O3", "asm": function + "3"},
                ]
            )
        train_ids, val_ids = split_function_ids(samples, val_fraction=0.25, seed=7)
        self.assertFalse(set(train_ids).intersection(val_ids))
        pairs = build_joint_pairs(samples, train_ids, seed=7)
        self.assertTrue(pairs)
        self.assertTrue(all(pair.function_id in train_ids for pair in pairs))

    def test_coupled_collator_alignment(self):
        pair = JointPair(
            function_id="f",
            numeric_id=4,
            source_asm="abc",
            target_asm="xyz",
            source_opt="gcc_O0_x86_64_none",
            target_opt="gcc_O3_x86_64_none",
        )
        collator = JointDistillCollator(
            FakeNovaTokenizer(), max_length=256, condition_on_target=True
        )
        batch = collator([pair])
        self.assertEqual(batch["lm"]["input_ids"].shape, batch["lm"]["labels"].shape)
        self.assertTrue(batch["lm"]["labels"].ne(-100).any())
        self.assertEqual(batch["pred"]["input_ids"].shape[0], 2)
        self.assertEqual(batch["pred"]["func_ids"].tolist(), [4, 4])
        self.assertFalse(
            batch["pred"]["pool_mask"][
                0, : collator.instruction_token_count
            ].any()
        )
        self.assertTrue(batch["pred"]["pool_mask"].any(dim=1).all())
        pred_mask = batch["pred"]["nova_attention_mask"]
        self.assertTrue(torch.equal(pred_mask, pred_mask.transpose(1, 2)))


if __name__ == "__main__":
    unittest.main()
