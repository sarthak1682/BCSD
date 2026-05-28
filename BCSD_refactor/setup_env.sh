#!/bin/bash
# =============================================================================
# setup_env.sh — Unified Environment Setup & Dataset Downloader
# =============================================================================
# Purpose: Automates the entire environment setup, package installation,
#          model downloading, and dataset retrieval for the Nova EBM pipeline.
#
# Usage:
#   bash BCSD_refactor/setup_env.sh
# =============================================================================

set -e

# Curated harmonious output colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================================${NC}"
echo -e "${BLUE}           Nova EBM Pipeline — Unified Environment Setup            ${NC}"
echo -e "${BLUE}===================================================================${NC}"

# --- 1. Python Environment Check ---
echo -e "\n${YELLOW}[1/4] Checking Python Environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed or not in PATH.${NC}"
    exit 1
fi
python_ver=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
echo -e "  - Found Python: ${GREEN}${python_ver}${NC}"

# --- 2. Installing Pinned Dependencies ---
echo -e "\n${YELLOW}[2/4] Installing Pinned Dependencies...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

if [ -f "$REQ_FILE" ]; then
    echo -e "  - Installing packages from requirements.txt..."
    pip install -r "$REQ_FILE"
    echo -e "  - Packages ${GREEN}successfully installed${NC}."
else
    echo -e "${RED}Error: requirements.txt not found at ${REQ_FILE}${NC}"
    exit 1
fi

# --- 3. Downloading HuggingFace Nova Model ---
echo -e "\n${YELLOW}[3/4] Fetching HuggingFace Nova Model Asset...${NC}"
python3 -c "
from huggingface_hub import snapshot_download
print('  - Downloading snapshot of lt-asset/nova-1.3b...')
snapshot_download('lt-asset/nova-1.3b')
"
echo -e "  - Nova model snapshot ${GREEN}downloaded successfully${NC}."

# --- 4. Retrieving Bench Datasets ---
echo -e "\n${YELLOW}[4/4] Retrieving Dataset Files...${NC}"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
DATA_DIR="${WORKSPACE_DIR}/nvemb"
mkdir -p "$DATA_DIR"

TRAIN_ID="10UPIRm9jz7QEN7Bva3Jk9UkqzIccACHF"
TEST_ID="1hDzq7aZiuJPvJe09iaE3IKmirWBLpq-C"

TRAIN_FILE="${DATA_DIR}/output_benchset_rebalanced_train_nova.jsonl"
TEST_FILE="${DATA_DIR}/output_benchset_rebalanced_test_nova.jsonl"

echo -e "  - Destination: ${DATA_DIR}"

if [ ! -f "$TRAIN_FILE" ]; then
    echo -e "  - Downloading rebalanced train dataset..."
    gdown "$TRAIN_ID" -O "$TRAIN_FILE"
else
    echo -e "  - Train dataset ${GREEN}already present${NC} (skipping)."
fi

if [ ! -f "$TEST_FILE" ]; then
    echo -e "  - Downloading rebalanced test dataset..."
    gdown "$TEST_ID" -O "$TEST_FILE"
else
    echo -e "  - Test dataset ${GREEN}already present${NC} (skipping)."
fi

# --- Final Health Summary ---
echo -e "\n${BLUE}===================================================================${NC}"
echo -e "${GREEN}                     Environment Setup Complete!                   ${NC}"
echo -e "${BLUE}===================================================================${NC}"
echo -e "\nTo verify and run a Stage 1 training run, execute the following:"
echo -e "  ${YELLOW}export PYTHONPATH=\"${WORKSPACE_DIR}:${WORKSPACE_DIR}/BCSD_refactor:\$PYTHONPATH\"${NC}"
echo -e "  ${YELLOW}python ${SCRIPT_DIR}/binarycorp_bench/nova_ebm/run_stages.py \\${NC}"
echo -e "      ${YELLOW}--stages        \"1\" \\${NC}"
echo -e "      ${YELLOW}--train_data    ${TRAIN_FILE} \\${NC}"
echo -e "      ${YELLOW}--eval_data     ${TEST_FILE} \\${NC}"
echo -e "      ${YELLOW}--output_dir    ${WORKSPACE_DIR}/test_output \\${NC}"
echo -e "      ${YELLOW}--gpu_id        0 \\${NC}"
echo -e "      ${YELLOW}--s1_batch      4 \\${NC}"
echo -e "      ${YELLOW}--s1_grad_accum 8 \\${NC}"
echo -e "      ${YELLOW}--s1_epochs     1 \\${NC}"
echo -e "      ${YELLOW}--log_interval  10${NC}"
echo -e "\n${BLUE}===================================================================${NC}"
