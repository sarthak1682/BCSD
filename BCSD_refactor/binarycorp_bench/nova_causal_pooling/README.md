# Frozen Native-Causal Nova Pooling

This experiment asks whether Nova's pretrained decoder representations already
support BCSD without converting the backbone into a bidirectional encoder.

## Controlled variants

All variants use the same official `lt-asset/nova-1.3b` weights, native Nova
tokenizer, query/positive construction, assembly-token positions, maximum
length, and bench evaluation protocol.

| variant | trainable component |
|---|---|
| `mean` | none (zero-shot baseline) |
| `linear` | identity-initialized 2048x2048 projection after mean pooling |
| `attention` | scalar attention pooling over assembly token states |

Nova is always frozen and in eval mode. No DoRA/LoRA adapter is attached. The
LM vocabulary projection is skipped because this experiment consumes token
embeddings, not next-token logits.

## Mask contract

Nova divides its attention heads into two groups. The first group receives the
standard Llama causal mask. The second receives Nova's tokenizer-produced
hierarchical mask. In the official implementation that mask is lower triangular
inside each assembly block and connects later block summaries only to earlier
summaries. It is therefore causal despite having more structure than an
ordinary triangular mask.

The collator passes this mask through unchanged except for truncation and batch
padding. It never calls `make_bidirectional_nova_mask` or `np.maximum(mask,
mask.T)`. Every sample is checked at runtime for future-token edges.

## Train the two heads

```bash
CUDA_VISIBLE_DEVICES=0 python -m binarycorp_bench.nova_causal_pooling.train \
  --device-id 0 \
  --epochs 1 \
  --pair-batch-size 8 \
  --grad-accum 4
```

Both heads train from the same frozen backbone forward pass. Their losses and
parameters are independent; combining the losses only avoids running Nova
twice.

## Continue a head

To keep training only the linear mean-probe from a previous run:

```bash
RUN_DIR=model_checkpoints/nova_causal_pooling/YYYYMMDD_HHMMSS

CUDA_VISIBLE_DEVICES=0 python -m binarycorp_bench.nova_causal_pooling.train \
  --device-id 0 \
  --epochs 1 \
  --pair-batch-size 16 \
  --grad-accum 2 \
  --train-heads linear \
  --resume-linear "$RUN_DIR/linear_head_best.pt"
```

The continuation creates a new timestamped run directory. It does not overwrite
the source checkpoint.

## Evaluate

Run each variant separately so its latency and memory measurement is
attributable:

```bash
# Parameter-free zero-shot mean pooling
CUDA_VISIBLE_DEVICES=0 python -m binarycorp_bench.nova_causal_pooling.eval \
  --pooling mean --device-id 0

# Use the directory printed by training
RUN_DIR=model_checkpoints/nova_causal_pooling/YYYYMMDD_HHMMSS

CUDA_VISIBLE_DEVICES=0 python -m binarycorp_bench.nova_causal_pooling.eval \
  --pooling linear --checkpoint-dir "$RUN_DIR" --device-id 0

CUDA_VISIBLE_DEVICES=0 python -m binarycorp_bench.nova_causal_pooling.eval \
  --pooling attention --checkpoint-dir "$RUN_DIR" --device-id 0
```

Each evaluation writes paired embeddings plus the full pool-size and directed
optimization/compiler/architecture/variant report under
`embeddings/binarycorp_bench/nova_causal_<variant>/`.

## Smoke checks

The following limits exercise the real model without starting a full run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m binarycorp_bench.nova_causal_pooling.train \
  --epochs 1 --max-train-pairs 16 --max-val-pairs 8 \
  --pair-batch-size 2 --num-workers 0
```
