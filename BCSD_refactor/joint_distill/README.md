# Compact joint EBM distillation

This package is deliberately separate from the existing teacher, EBM, and
student pipelines. It never modifies their checkpoints or training code.

## Scientific scope

The merged full-EBM `s3_final` model is the frozen teacher for both tasks:

- causal assembly-to-assembly translation, using EBM's native Nova mask;
- BCSD retrieval, using EBM's symmetrized Nova mask and saved attention-pooling
  head.

One compact transformer is trained on both losses every step. It switches only
its attention contract: causal for LM and fully bidirectional for retrieval. Two
learned task embeddings condition the shared weights. The default architecture
has **98,626,304 parameters**, versus an estimated 110.7M core parameters for
the local 12-layer CLAP checkpoint. The LM vocabulary projection is tied to the
student token embedding.

The causal metric is assembly translation cross-entropy/perplexity, token
accuracy, and greedy exact match. It is **not** HumanEval-Decompile Pass@k: the
available datasets have no C source or executable correctness tests.

## Reused, unchanged components

- `shared.nova_utils`: local Nova paths and mask symmetrization (the package's
  loader deliberately keeps EBM-full's untouched 32,280-token vocabulary);
- `shared.data_utils`: JSONL loading and standard bench evaluation pairs;
- `shared.losses`: positive-aware supervised contrastive loss;
- `shared.pooling.AttentionPooling`: exact full-EBM teacher pooling head;
- `binarycorp_bench.eval_bench`: the existing full transfer-matrix scorer;
- `shared.profiling`: latency and peak-memory measurement.

## Train

Run from `BCSD_refactor/` in the existing `defusion` environment:

```bash
CUDA_VISIBLE_DEVICES=0 python -m joint_distill.train
```

The default paths are:

- teacher: `../model_checkpoints/nova_ebm_bench/s3_final`;
- data: `../nvemb/output_benchset_rebalanced_train_nova.jsonl`;
- outputs: `model_checkpoints/joint_distill/<timestamp>/`.

The default reproduces EBM-full's original unconditioned causal input format so
teacher and student see the same distribution. The benchmark has multiple
valid targets, however, so target-configuration conditioning is provided as a
second, stricter generation experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python -m joint_distill.train --condition-on-target
```

Do not compare a conditioned student to an unconditioned teacher report. Run
`eval_teacher` with the same flag for that ablation.

First run a bounded smoke experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python -m joint_distill.train \
  --epochs 1 \
  --max-train-pairs 64 \
  --max-val-pairs 16 \
  --val-batches 2 \
  --pair-batch-size 2 \
  --grad-accum 2 \
  --max-length 256 \
  --output-dir model_checkpoints/joint_distill/smoke
```

The smoke command uses a shorter positional table and therefore creates a
slightly smaller checkpoint. Do not compare its parameter count or retrieval
quality with the full default run.

Each run writes:

- `run_config.json` and `metrics.jsonl`;
- `best_joint/`: lowest weighted validation loss;
- `best_dual_canary/`: improved LM and retrieval validation losses together;
- `final/` and `summary.json`.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python -m joint_distill.eval \
  --checkpoint model_checkpoints/joint_distill/RUN/best_joint \
  --task both
```

This writes compatible retrieval embeddings/report plus a translation report
under a run-specific directory in `embeddings/binarycorp_bench/joint_distill/`.

Evaluate EBM-full under the identical causal protocol before comparing LM
retention:

```bash
CUDA_VISIBLE_DEVICES=0 python -m joint_distill.eval_teacher \
  --output model_checkpoints/joint_distill/RUN/teacher_translation_report.json
```

## Compare with EBM-full and CLAP

```bash
python -m joint_distill.compare \
  --student-checkpoint model_checkpoints/joint_distill/RUN/best_joint \
  --student-retrieval embeddings/binarycorp_bench/joint_distill/RUN_best_joint/retrieval_report.json \
  --student-translation embeddings/binarycorp_bench/joint_distill/RUN_best_joint/translation_report.json \
  --teacher-translation model_checkpoints/joint_distill/RUN/teacher_translation_report.json \
  --output model_checkpoints/joint_distill/RUN/comparison.json
```

The EBM-full and final fine-tuned CLAP reports default to their existing local
paths. A causal teacher report can be added with `--teacher-translation` once
EBM-full has been evaluated on the identical conditioned translation protocol.

## Losses

The coupled objective is configurable and logged component by component:

```text
L = lambda_lm_ce * target cross-entropy
  + lambda_lm_kd * top-k teacher KL
  + lambda_embedding_kd * cosine teacher-embedding loss
  + lambda_contrastive * positive-aware supervised contrastive loss
```

Nova token embeddings initialize the compact token table through a one-time
random projection. No Nova embedding table remains in the deployed student.
