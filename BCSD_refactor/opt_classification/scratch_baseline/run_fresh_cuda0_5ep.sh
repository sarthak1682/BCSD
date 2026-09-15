#!/usr/bin/env bash
set -euo pipefail

cd /home/ra72yeq/BCSD/BCSD_refactor
mkdir -p opt_classification/outputs/table1_opt_probe/scratch_student_opt_cuda0_fresh_5ep

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
/home/ra72yeq/miniconda3/envs/defusion/bin/python -u -m opt_classification.scratch_baseline.train \
  --device cuda:0 \
  --require-cuda \
  --output-dir opt_classification/outputs/table1_opt_probe/scratch_student_opt_cuda0_fresh_5ep \
  --epochs 5 \
  --batch-size 16 \
  --grad-accum-steps 2 \
  --num-workers 0 \
  --log-every 100 \
  --save-every-epoch \
  --test-every-epoch \
  --epoch-test-bootstrap-samples 0 \
  --seed 42 \
  2>&1 | tee -a opt_classification/outputs/table1_opt_probe/scratch_student_opt_cuda0_fresh_5ep/train.log
