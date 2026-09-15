"""Tokenizer loading that exactly matches the merged EBM-full vocabulary."""

from __future__ import annotations

import sys

from transformers import AutoTokenizer

from shared.nova_utils import MODEL_ID, NOVA_CACHE_DIR


def load_ebm_tokenizer():
    """Load Nova's untouched 32,280-token vocabulary.

    ``shared.nova_utils.setup_nova_tokenizer`` adds ``[MASK]`` for the disabled
    MNTP experiment. EBM-full stages 1 and 3 did not add it, so joint
    distillation must not use that helper.
    """
    if NOVA_CACHE_DIR not in sys.path:
        sys.path.insert(0, NOVA_CACHE_DIR)
    from modeling_nova import NovaTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=NOVA_CACHE_DIR,
        local_files_only=True,
    )
    return base_tokenizer, NovaTokenizer(base_tokenizer)
