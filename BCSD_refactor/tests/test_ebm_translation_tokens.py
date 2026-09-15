import unittest

import numpy as np

from shared.collators import TranslationCollator
from shared.data_utils import (
    build_ebm_translation_item,
    ebm_setting_block,
)


class _BaseTokenizer:
    pad_token_id = 0


class _RecordingNovaTokenizer:
    def __init__(self):
        self.tokenizer = _BaseTokenizer()
        self.last_input = None
        self.last_output = None

    def encode(self, input_text, output_text, char_types):
        self.last_input = input_text
        self.last_output = output_text
        self.asserted_char_types = char_types
        input_ids = np.arange(1, len(input_text) + len(output_text) + 1)
        labels = np.concatenate((
            np.full(len(input_text), -100),
            np.arange(1, len(output_text) + 1),
        ))
        return {
            "input_ids": input_ids,
            "labels": labels,
            "nova_attention_mask": np.tril(np.ones((len(input_ids), len(input_ids)))),
        }


class EBMTranslationTokenTests(unittest.TestCase):
    def test_setting_blocks_cover_benchmark_and_binarycorp3m(self):
        self.assertEqual(
            ebm_setting_block("gcc_O3_x86_64_flattening"),
            "o3 gcc x86_64 flattening",
        )
        self.assertEqual(ebm_setting_block("O0"), "o0")

    def test_translation_item_contains_source_and_target_settings(self):
        item = build_ebm_translation_item(
            "mov eax, 1", "clang_O1_arm64_none",
            "mov x0, 1", "gcc_O3_x86_64_bogus",
        )
        self.assertEqual(
            item,
            (
                "mov eax, 1\no1 clang arm64 none",
                "mov x0, 1",
                "o3 gcc x86_64 bogus",
            ),
        )

    def test_nova_collator_masks_transition_and_predicts_target(self):
        tokenizer = _RecordingNovaTokenizer()
        collator = TranslationCollator(tokenizer, max_length=256)
        source, target, target_settings = build_ebm_translation_item(
            "source asm", "O0", "target asm", "O3"
        )

        batch = collator([(source, target, target_settings)])

        self.assertEqual(
            tokenizer.last_input,
            "source asm\no0\n<SEP> o3\n",
        )
        self.assertEqual(tokenizer.last_output, "target asm\n<EOS>")
        input_length = len(tokenizer.last_input)
        self.assertTrue((batch["labels"][0, :input_length] == -100).all())
        self.assertTrue((batch["labels"][0, input_length:] != -100).all())


if __name__ == "__main__":
    unittest.main()
