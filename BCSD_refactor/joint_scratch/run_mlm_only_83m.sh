#!/usr/bin/env bash
# Matched RQ2 ablation vs clap_strict_83m_lm_mlm_pretrain10_* :
#   Stage A: MLM only, 10 epochs, same 83M architecture/data/budget
#   Stage C: 1 epoch contrastive retrieval FT
#   Eval: official BinaryCorp bench protocol, pre-CL and post-CL
set -euo pipefail
cd /home/ra72yeq/BCSD/BCSD_refactor
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/ra72yeq/miniconda3/envs/defusion/bin/python}"
DATA="/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_train_clap_strict.jsonl"
TEST="/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_test_clap_strict.jsonl"
PRE="model_checkpoints/joint_scratch/clap_strict_83m_mlm_pretrain10_trainonly_gpu0_seed42"
CL="model_checkpoints/joint_scratch/clap_strict_83m_mlm_pretrain10_cl1_trainonly_gpu0_seed42"

ARCH=(
  --d-model 768
  --layers 8
  --heads 12
  --ff-dim 3072
  --embedding-dim 768
  --max-length 512
  --dropout 0.1
  --mlm-probability 0.15
  --raw-chars-per-token 8
  --seed 42
  --pair-batch-size 64
  --grad-accum 1
  --learning-rate 3e-4
  --weight-decay 0.01
  --warmup-ratio 0.03
  --num-workers 8
  --device cuda:0
  --no-gradient-checkpointing
  --no-validation
)

echo "=== Stage A: MLM-only pretrain (10 epochs) ==="
"$PY" -m joint_scratch.train \
  --data "$DATA" \
  --output-dir "$PRE" \
  --epochs 10 \
  --training-objectives mlm \
  "${ARCH[@]}"

echo "=== Eval pre-CL (MLM backbone, untrained retrieval head) ==="
"$PY" -m joint_scratch.evaluate \
  --checkpoint "$PRE" \
  --split final \
  --protocol bench \
  --data "$TEST" \
  --device cuda:0 \
  --output "$PRE/eval_official_test_final_bench.json"

echo "=== Stage C: contrastive FT (1 epoch) ==="
"$PY" -m joint_scratch.train \
  --data "$DATA" \
  --output-dir "$CL" \
  --init-checkpoint "$PRE/final" \
  --epochs 1 \
  --training-objectives retrieval \
  --training-mode joint \
  "${ARCH[@]}"

echo "=== Eval post-CL ==="
"$PY" -m joint_scratch.evaluate \
  --checkpoint "$CL" \
  --split final \
  --protocol bench \
  --data "$TEST" \
  --device cuda:0 \
  --output "$CL/eval_official_test_final_bench.json"

echo "MLM-only 83M pipeline complete."
echo "  pretrain: $PRE"
echo "  cl1:      $CL"
