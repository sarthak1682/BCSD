#!/bin/bash
# =============================================================================
# run_pipeline.sh — Run all 3 training stages of the Nova EBM pipeline
# =============================================================================

export PYTHONPATH="/workspace/BCSD:/workspace/BCSD/BCSD_refactor:$PYTHONPATH"

python3 /workspace/BCSD/BCSD_refactor/binarycorp_bench/nova_ebm/run_stages.py \
    --stages        "1,2,3" \
    --train_data    /workspace/BCSD/nvemb/output_benchset_rebalanced_train_nova.jsonl \
    --eval_data     /workspace/BCSD/nvemb/output_benchset_rebalanced_test_nova.jsonl \
    --output_dir    /workspace/test_output \
    --gpu_id        0 \
    --s1_batch      4 \
    --s1_grad_accum 8 \
    --s1_lr         2e-5 \
    --s1_epochs     1 \
    --s2_batch      4 \
    --s2_grad_accum 8 \
    --s2_lr         2e-4 \
    --s2_epochs     1 \
    --s3_batch      8 \
    --s3_grad_accum 8 \
    --s3_lr         3e-5 \
    --s3_epochs     1 \
    --log_interval  10
