#!/usr/bin/env bash
set -euo pipefail

/home/ra72yeq/miniconda3/envs/defusion/bin/python -u -m opt_classification.scratch_baseline.train \
  --device auto \
  --require-cuda \
  --epochs 20 \
  --batch-size 16 \
  --grad-accum-steps 2 \
  --num-workers 0 \
  --log-every 100 \
  --save-every-epoch \
  --test-every-epoch \
  --epoch-test-bootstrap-samples 0 \
  --seed 42 \
  --resume
