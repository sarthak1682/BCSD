"""Load Nova's local assembly tokenizer without loading Nova model weights."""

from __future__ import annotations

import os
import sys

# Keep tokenizer-only jobs on the PyTorch path.  Without these, Transformers can
# discover installed JAX/Flax/TF packages during import; Ctrl-C during that phase
# can get swallowed by XLA cleanup callbacks.
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import AutoTokenizer

from shared.nova_utils import MODEL_ID, NOVA_CACHE_DIR


MLM_MASK_TOKEN = "<|mask|>"


def load_nova_tokenizer():
    """Return the unchanged local Nova vocabulary and structured wrapper."""

    if NOVA_CACHE_DIR not in sys.path:
        sys.path.insert(0, NOVA_CACHE_DIR)
    from modeling_nova import NovaTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=NOVA_CACHE_DIR,
        local_files_only=True,
    )
    if base_tokenizer.mask_token_id is None:
        base_tokenizer.add_special_tokens({"mask_token": MLM_MASK_TOKEN})
    return base_tokenizer, NovaTokenizer(base_tokenizer)
