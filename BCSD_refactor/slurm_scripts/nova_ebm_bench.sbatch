#!/bin/bash
# =============================================================================
# SLURM job script — Nova EBM bench (stages 1, 2, 3 + eval)
#
# Submit with:
#   sbatch slurm_scripts/nova_ebm_bench.sbatch
#
# To resume from stage-1 only:
#   sbatch slurm_scripts/nova_ebm_bench.sbatch --stages 2,3 --s1_ckpt /path/to/s1_final
# =============================================================================

#SBATCH --job-name=nova_ebm_bench
#SBATCH --output=logs/nova_ebm_bench_%j.out     # stdout  (%j = job id)
#SBATCH --error=logs/nova_ebm_bench_%j.err      # stderr
#SBATCH --time=24:00:00                          # wall-clock limit — adjust as needed
#SBATCH --partition=gpu                          # partition name — change to match your cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                             # 1 GPU; cluster assigns whichever is available
#SBATCH --mem=64G

# ---------------------------------------------------------------------------
# 0.  Fail fast on any error
# ---------------------------------------------------------------------------
set -euo pipefail

echo "========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Started at:  $(date)"
echo "========================================="

# ---------------------------------------------------------------------------
# 1.  Environment setup
# ---------------------------------------------------------------------------
module purge
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || true   # best-effort; SLURM sets CUDA_VISIBLE_DEVICES automatically

# Activate conda — creates the 'bcsd' env if it doesn't exist yet
source ~/.bashrc
if ! conda env list | grep -q '^bcsd '; then
    echo "Creating conda env 'bcsd'..."
    conda create -y -n bcsd python=3.10
fi
conda activate bcsd

# ---------------------------------------------------------------------------
# 2.  Paths  (auto-detected from script location — no manual editing needed)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"   # BCSD_refactor/slurm_scripts → BCSD
BCSD_REFACTOR="${REPO_ROOT}/BCSD_refactor"
DATA_DIR="${REPO_ROOT}/nvemb"                                  # where the .jsonl files live
OUTPUT_DIR="${REPO_ROOT}/outputs/nova_ebm_bench_${SLURM_JOB_ID}"

TRAIN_DATA="${DATA_DIR}/output_benchset_rebalanced_train_nova.jsonl"
EVAL_DATA="${DATA_DIR}/output_benchset_rebalanced_test_nova.jsonl"

# Make sure the log directory exists (SLURM needs it before the job starts)
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 3.  Python path — so `from shared.xxx import ...` works without installing
# ---------------------------------------------------------------------------
export PYTHONPATH="${BCSD_REFACTOR}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 4.  (Optional) Download dataset if not already present
# ---------------------------------------------------------------------------
if [ ! -f "${TRAIN_DATA}" ] || [ ! -f "${EVAL_DATA}" ]; then
    echo "Dataset not found — downloading..."
    python "${BCSD_REFACTOR}/download_dataset.py" --output_dir "${DATA_DIR}"
fi

# ---------------------------------------------------------------------------
# 5.  Verify GPU assignment
# ---------------------------------------------------------------------------
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set (SLURM handles it)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ---------------------------------------------------------------------------
# 6.  Run training
# ---------------------------------------------------------------------------
echo ""
echo "Starting Nova EBM bench training..."
echo "Output dir: ${OUTPUT_DIR}"
echo ""

python "${BCSD_REFACTOR}/binarycorp_bench/nova_ebm/run_stages.py" \
    --stages       "1,2,3" \
    --train_data   "${TRAIN_DATA}" \
    --eval_data    "${EVAL_DATA}" \
    --output_dir   "${OUTPUT_DIR}" \
    --s1_epochs    1 \
    --s1_batch     4 \
    --s1_grad_accum 8 \
    --s1_lr        2e-5 \
    --s2_epochs    1 \
    --s2_batch     4 \
    --s2_grad_accum 8 \
    --s2_lr        2e-4 \
    --s2_max_steps 1000 \
    --s3_epochs    1 \
    --s3_batch     8 \
    --s3_grad_accum 8 \
    --s3_lr        3e-5 \
    --eval_batch_size 16 \
    --seed         42

echo ""
echo "Job finished at: $(date)"
