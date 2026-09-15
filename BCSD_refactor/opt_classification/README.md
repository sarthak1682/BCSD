# Optimization-classification probe

This experiment tests whether BCSD specialization suppresses optimization-level
information in otherwise frozen function representations.

The relevant conflict is in the positive pairs, not the negative pool. BCSD
training pulls differently compiled versions of the same function together.
Optimization classification asks a probe to distinguish those versions. Keeping
same-function variants out of the negative pool avoids false negatives, but it
does not remove this conflict between the two objectives.

## Protocol

- Retain only matched `(function, compiler, architecture, variant)` groups that
  contain exactly one O0, O1, O2, and O3 instance.
- Use the official project-disjoint benchmark train/test split.
- Split probe train and validation data by binary; test data is never used for
  probe selection.
- Encode assembly only. Synthetic compiler/optimization settings blocks and
  retrieval-query instructions are excluded.
- Freeze every encoder and train an L2-regularized logistic-regression probe.
- Select regularization on validation macro-F1.
- Evaluate four-way O0–O3 classification and binary O0-vs-O3 classification.
- Report accuracy, balanced accuracy, macro-F1, per-class recall, confusion
  matrices, O0/O3 ROC-AUC, shuffled-label sanity checks, and confidence
  intervals clustered by source-function identity.

## Models

The deployed suite contains all Table I checkpoints:

- `clap`
- `ebm-full`
- `ebm-no-causal`
- `student`
- `jtrans`
- `joint-scratch`
- `asm2vec`

Available controls are `clap-raw`, `nova-raw`, `ebm-s1`, `student-pre`, and
`jtrans-bench`. The orchestrated full suite also evaluates Nova checkpoints with
a fixed label-token mean pool under native causal and symmetrized bidirectional
masks. Those fixed-pooling comparisons isolate attention-mask conversion from
BCSD checkpoint training. Asm2Vec has no in-repository pre-BCSD control and is
therefore descriptive only.

## Output layout

Each run is written below `opt_classification/outputs/<run-id>/`:

```text
manifest.jsonl
manifest_summary.json
embeddings/<configuration>.pt
probes/<configuration>/4way.json
probes/<configuration>/o0-vs-o3.json
metrics.csv
paired_deltas.csv
probe_summary.png
report.md
```

Embedding caches include checkpoint, input, mask, pooling, preprocessing, and
manifest hashes. Existing caches are reused only when their provenance matches.

## Running

Run from the repository root in the `defusion` environment:

```bash
mkdir -p opt_classification/outputs/table1_opt_probe && \
CUDA_VISIBLE_DEVICES=1 python -u -m opt_classification.run_experiment \
  --run-id table1_opt_probe \
  --device cuda:0 \
  --suite full \
  --resume 2>&1 | tee -a opt_classification/outputs/table1_opt_probe/run.log
```

Use `--suite deployed` for only the six Table I representations. Asm2Vec performs
ten epochs of test-time function-vector optimization and is expected to be the
slowest adapter. Use a new run ID rather than overwriting an existing run.

## Interpretation

The primary evidence is a paired pre/post change with architecture, tokenizer,
pooling, input rows, and probe protocol held fixed. A lower post-BCSD score then
supports optimization-feature suppression. Absolute rankings across unrelated
architectures are secondary because model capacity, tokenization, pooling, and
pretraining differ.

For Nova, compare:

- raw Nova bidirectional mean → EBM-no-causal bidirectional mean;
- EBM stage-1 bidirectional mean → EBM-full bidirectional mean; and
- native causal mean → bidirectional mean within the same checkpoint.

For the smaller or non-Nova encoders, compare raw CLAP → Table I CLAP,
pre-refinement Student → Table I Student, and upstream jTrans → benchmark-tuned
jTrans. The Table I jTrans row itself is the upstream checkpoint, not the local
benchmark-finetuned artifact.
