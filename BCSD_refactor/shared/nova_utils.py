"""Nova model loading and tokenizer setup utilities.

Deduplicated from: run_nova_ebm_stages.py, run_nova_ebm_stages_bench.py,
train_c_bidir_no_mntp.py, eval_c_bidir_no_mntp.py, train_teacher_bench.py,
eval_teacher_bench.py, distill_nova.py, distill_student_bench.py, etc.
"""

import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer


_SNAPSHOT_HASH = "4b4805bac4f13ef8bec678072ef60609ea3b0e77"

def _resolve_nova_cache_dir() -> str:
    """Resolve Nova model snapshot dir, respecting HF_HOME if set.

    Priority:
      1. $HF_HOME/hub/models--lt-asset--nova-1.3b/snapshots/<hash>
      2. ~/.cache/huggingface/hub/models--lt-asset--nova-1.3b/snapshots/<hash>
    """
    hf_home = os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface")
    )
    return os.path.join(
        hf_home, "hub", "models--lt-asset--nova-1.3b",
        "snapshots", _SNAPSHOT_HASH
    )

NOVA_CACHE_DIR = _resolve_nova_cache_dir()
MODEL_ID = "lt-asset/nova-1.3b"


def _ensure_nova_on_path():
    """Add Nova model directory to sys.path if not already present."""
    if NOVA_CACHE_DIR not in sys.path:
        sys.path.insert(0, NOVA_CACHE_DIR)


def setup_nova_tokenizer():
    """Set up Nova tokenizer with [MASK] special token.

    Returns:
        tuple: (base_tokenizer, nova_tokenizer, mask_id)
    """
    _ensure_nova_on_path()
    from modeling_nova import NovaTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=NOVA_CACHE_DIR)
    base_tokenizer.add_special_tokens({'additional_special_tokens': ['[MASK]']})
    mask_id = base_tokenizer.encode('[MASK]')[-1]
    nova_tokenizer = NovaTokenizer(base_tokenizer)

    return base_tokenizer, nova_tokenizer, mask_id


def load_nova(from_dir=None, device_id=0, base_tokenizer=None, torch_dtype=torch.bfloat16):
    """Load NovaForCausalLM, optionally from a checkpoint directory.

    Args:
        from_dir: Directory to load from; defaults to NOVA_CACHE_DIR.
        device_id: GPU device ID for device_map.
        base_tokenizer: If provided, resize embeddings to match.
        torch_dtype: Model dtype (default: bfloat16).

    Returns:
        NovaForCausalLM model.
    """
    _ensure_nova_on_path()
    from modeling_nova import NovaForCausalLM

    src = from_dir or NOVA_CACHE_DIR
    print(f"  loading NovaForCausalLM from {src}")
    model = NovaForCausalLM.from_pretrained(
        src, torch_dtype=torch_dtype, device_map={"": device_id}
    )
    if base_tokenizer is not None:
        model.resize_token_embeddings(len(base_tokenizer))
    return model


def make_bidirectional_nova_mask(nova_mask):
    """Symmetrize Nova attention mask to make blocks bidirectional."""
    return np.maximum(nova_mask, nova_mask.T)
