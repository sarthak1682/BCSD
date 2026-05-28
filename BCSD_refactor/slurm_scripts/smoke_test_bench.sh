#!/bin/bash
# =============================================================================
# Smoke test for Nova EBM bench — runs 2 steps per stage with tiny batches.
#
# PURPOSE: Validate that the full pipeline (imports, data loading, model init,
#          forward pass, backward pass, saving) works without crashing.
#          Run this on a rented GPU (RunPod / Lambda / Vast.ai) BEFORE
#          submitting the real SLURM job to your supervisor's cluster.
#
# Usage (plain bash — no SLURM needed):
#   cd /path/to/BCSD_refactor
#   bash slurm_scripts/smoke_test_bench.sh
#
# Or pass a custom data dir:
#   bash slurm_scripts/smoke_test_bench.sh /path/to/nvemb
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BCSD_REFACTOR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BCSD_REFACTOR}/.." && pwd)"

DATA_DIR="${1:-${REPO_ROOT}/nvemb}"         # first arg overrides
OUTPUT_DIR="${REPO_ROOT}/smoke_test_output"

TRAIN_DATA="${DATA_DIR}/output_benchset_rebalanced_train_nova.jsonl"
EVAL_DATA="${DATA_DIR}/output_benchset_rebalanced_test_nova.jsonl"

export PYTHONPATH="${BCSD_REFACTOR}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

echo "========================================="
echo "  Nova EBM Bench — Smoke Test"
echo "  $(date)"
echo "========================================="
echo "  BCSD_REFACTOR: ${BCSD_REFACTOR}"
echo "  DATA_DIR:       ${DATA_DIR}"
echo "  OUTPUT_DIR:     ${OUTPUT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Check GPU
# ---------------------------------------------------------------------------
echo "--- GPU check ---"
if python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print('CUDA ok:', torch.cuda.get_device_name(0))"; then
    ok "CUDA available"
else
    warn "No GPU found — will run on CPU (very slow, but still validates imports/logic)"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 2: Check imports
# ---------------------------------------------------------------------------
echo "--- Import check ---"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get("PYTHONPATH", "").split(":")[0])
from shared.data_utils   import set_seed, load_jsonl, parse_bench_opt, asm_to_text, group_samples_by_id
from shared.nova_utils   import NOVA_CACHE_DIR, MODEL_ID
from shared.pooling      import AttentionPooling
from shared.collators    import TranslationCollator, MNTPCollator, PairCollator
from shared.losses       import contrastive_loss_positive_aware
from shared.training     import run_generic_train
from binarycorp_bench.eval_bench import build_eval_pairs, compute_report, print_report_summary
print("All shared imports OK")
PYEOF
ok "Imports passed"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Download dataset if needed
# ---------------------------------------------------------------------------
echo "--- Dataset check ---"
if [ ! -f "${TRAIN_DATA}" ] || [ ! -f "${EVAL_DATA}" ]; then
    echo "Dataset not found at ${DATA_DIR} — downloading..."
    python "${BCSD_REFACTOR}/download_dataset.py" --output_dir "${DATA_DIR}" \
        || fail "Dataset download failed. Check Google Drive permissions."
fi
ok "Dataset present: $(wc -l < "${TRAIN_DATA}") train lines, $(wc -l < "${EVAL_DATA}") eval lines"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Run pipeline with minimal steps
# ---------------------------------------------------------------------------
echo "--- Pipeline smoke run (2 steps per stage, tiny batches) ---"
mkdir -p "${OUTPUT_DIR}"

python "${BCSD_REFACTOR}/binarycorp_bench/nova_ebm/run_stages.py" \
    --stages        "1,2,3" \
    --train_data    "${TRAIN_DATA}" \
    --eval_data     "${EVAL_DATA}" \
    --output_dir    "${OUTPUT_DIR}" \
    --s1_epochs     1  --s1_batch 2  --s1_grad_accum 1 \
    --s2_epochs     1  --s2_batch 2  --s2_grad_accum 1  --s2_max_steps 2 \
    --s3_epochs     1  --s3_batch 2  --s3_grad_accum 1 \
    --eval_batch_size 4 \
    --log_interval  1 \
    --seed          42

echo ""
ok "Pipeline completed without errors"

# ---------------------------------------------------------------------------
# Step 5: Verify outputs exist
# ---------------------------------------------------------------------------
echo "--- Output check ---"
EXPECTED=(
    "${OUTPUT_DIR}/s1_final"
    "${OUTPUT_DIR}/s2_final"
    "${OUTPUT_DIR}/s3_final/pooling_head.pt"
    "${OUTPUT_DIR}/eval_bench_report.json"
)
all_ok=true
for f in "${EXPECTED[@]}"; do
    if [ -e "${f}" ]; then
        ok "${f}"
    else
        warn "Missing: ${f}"
        all_ok=false
    fi
done

echo ""
if $all_ok; then
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  SMOKE TEST PASSED — safe to submit  ${NC}"
    echo -e "${GREEN}======================================${NC}"
else
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}  Smoke test finished with warnings.    ${NC}"
    echo -e "${YELLOW}  Check missing outputs above.          ${NC}"
    echo -e "${YELLOW}========================================${NC}"
    exit 1
fi
