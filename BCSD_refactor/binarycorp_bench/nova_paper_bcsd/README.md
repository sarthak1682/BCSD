# Nova Paper-Style BCSD Fine-Tuning

This baseline targets the setup Yunru pointed to from the Nova paper, not the
frozen-head ablation.

## Paper Details Mirrored

The Nova paper's BCSD setup says:

- BCSD fine-tuning uses BinaryCorp-3M O0/O3 assembly pairs.
- The model is decoder-only Nova with its hierarchical attention.
- Function embeddings are the average of the last-layer hidden states at all
  Nova instruction-summary `[INST]` tokens. In this tokenizer these are the
  `<label-1>` ... `<label-256>` token IDs.
- Training uses L2 distance between O0 query and O3 candidate embeddings.
- Evaluation normalizes embeddings and ranks with cosine similarity.
- Appendix A.2 reports one epoch, learning rate 2e-5, max length 2048, and
  batch size 64.

This implementation preserves Nova's native hierarchical causal mask. It does
not use `np.maximum(mask, mask.T)`, does not attach the retrieval instruction
prompt, and does not add a learned pooling head.

## Train

Feasible single-GPU version with DoRA adapters:

```bash
CUDA_VISIBLE_DEVICES=0 python -m binarycorp_bench.nova_paper_bcsd.train \
  --device-id 0 \
  --tuning dora \
  --epochs 1 \
  --max-length 2048 \
  --pair-batch-size 8 \
  --grad-accum 8
```

Closest paper-faithful mode, if memory allows full 1.3B fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=0 python -m binarycorp_bench.nova_paper_bcsd.train \
  --device-id 0 \
  --tuning full \
  --epochs 1 \
  --max-length 2048 \
  --pair-batch-size 8 \
  --grad-accum 8
```

The effective batch is `pair-batch-size * grad-accum` O0/O3 function pairs.

## Evaluate

Use the run directory printed by training:

```bash
RUN_DIR=model_checkpoints/nova_paper_bcsd/YYYYMMDD_HHMMSS

CUDA_VISIBLE_DEVICES=0 python -m binarycorp_bench.nova_paper_bcsd.eval \
  --device-id 0 \
  --checkpoint-dir "$RUN_DIR" \
  --checkpoint-kind best
```

Results are written under
`embeddings/binarycorp_bench/nova_paper_bcsd_<tuning>/`.

## Smoke Check

```bash
CUDA_VISIBLE_DEVICES=0 python -m binarycorp_bench.nova_paper_bcsd.train \
  --device-id 0 \
  --tuning dora \
  --epochs 1 \
  --max-train-pairs 16 \
  --max-val-pairs 8 \
  --pair-batch-size 2 \
  --grad-accum 2 \
  --num-workers 0
```

