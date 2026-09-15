# Joint training from scratch

This is one isolated, teacher-free feasibility experiment. It does not load
Nova weights, EBM-full, CLAP weights, LoRA adapters, or distillation targets.
Nova contributes only its fixed assembly tokenizer; every model parameter is
randomly initialized.

## Objective and task switch

One approximately CLAP-sized model is optimized for both tasks on every update:

```text
L = lambda_lm * next-token cross entropy
  + lambda_retrieval * positive-aware contrastive loss
```

The shared 97,641,984-parameter model switches explicitly:

- `LM`: learned LM task token, causal attention, tied vocabulary head;
- `retrieval`: learned retrieval task token, bidirectional attention,
  attention pooling, and a 768-dimensional embedding head.

This first experiment is not a three-model specialist comparison. It asks the
smaller question: can one randomly initialized shared model learn both signals
at the same time? We look for decreasing held-out LM CE/perplexity, increasing
token accuracy, decreasing contrastive loss, and retrieval Recall@1 above
chance.

## BinaryCorp Bench data

The default data is the CLAP-strict rebalanced multi-compiler BinaryCorp Bench
set:

```text
/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_train_clap_strict.jsonl
```

This uses raw IDA-style instruction lists and avoids the Nova-format
`<label-N>` instruction-summary tokens. The loader rejects those label tokens so
we do not accidentally train on the Nova representation.

Each function has assembly from several compiler, optimization, architecture,
and obfuscation configurations. Every variant is used once as a source and
paired deterministically with another variant of the same function.

Generation is not ambiguous: the LM prompt states both the source and requested
target configurations. The same source/target pair is used as the positive pair
for retrieval. Validation is binary-disjoint by default.

The local structural audit found:

- 395,697 training variants across 36,901 functions and 35 configurations;
- all 36,901 training functions have at least two variants;
- 45,464 held-out variants, of which 45,000 belong to multi-variant functions;
- zero function-ID and zero binary-group overlap between the provided train and
  test files.

Reproduce the audit:

```bash
python -m joint_scratch.audit_data \
  /home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_train_clap_strict.jsonl \
  --test /home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_test_clap_strict.jsonl
```

The provided test file should remain untouched until the recipe and checkpoint
selection rule are frozen.

## Small joint smoke run

The smoke test is one small joint model, not separate full generation and
retrieval models. Run from `BCSD_refactor/`:

```bash
conda run --no-capture-output -n defusion \
  python -m joint_scratch.train \
  --device cpu \
  --epochs 1 \
  --max-train-pairs 64 \
  --max-val-pairs 16 \
  --val-batches 8 \
  --pair-batch-size 2 \
  --grad-accum 1 \
  --num-workers 0 \
  --d-model 32 \
  --layers 1 \
  --heads 4 \
  --ff-dim 64 \
  --embedding-dim 16 \
  --max-length 64 \
  --no-gradient-checkpointing
```

This model has roughly one million parameters. Its purpose is to verify data,
conditioning, masks, both gradients, metrics, and checkpointing before spending
the full-model budget.

## Full joint run

Once the small curve shows that both objectives learn:

```bash
conda run --no-capture-output -n defusion \
  python -m joint_scratch.train
```

Each run writes `run_config.json`, `metrics.jsonl`, `best/`, `final/`, and
`summary.json` below `model_checkpoints/joint_scratch/`.
