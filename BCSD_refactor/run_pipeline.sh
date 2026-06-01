#!/bin/bash
# =============================================================================
# run_pipeline.sh — Run all 3 training stages of the Nova EBM pipeline
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

GPU_ID="${GPU_ID:-0}"   # override with: GPU_ID=1 bash run_pipeline.sh

# Auto-logging: mirrors all output to logs/run_<timestamp>.log
LOG_FILE="${REPO_ROOT}/logs/run_$(date +%Y%m%d_%H%M%S)_gpu${GPU_ID}.log"
mkdir -p "${REPO_ROOT}/logs"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: ${LOG_FILE}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/BCSD_refactor:${PYTHONPATH:-}"

python3 "${REPO_ROOT}/BCSD_refactor/binarycorp_bench/nova_ebm/run_stages.py" \
    --stages        "1,2,3" \
    --train_data    "${REPO_ROOT}/nvemb/output_benchset_rebalanced_train_nova.jsonl" \
    --eval_data     "${REPO_ROOT}/nvemb/output_benchset_rebalanced_test_nova.jsonl" \
    --output_dir    "${REPO_ROOT}/test_output" \
    --gpu_id        "${GPU_ID}" \
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
    --log_interval  100
