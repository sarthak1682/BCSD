#!/bin/bash
# =============================================================================
# setup_env.sh — Environment Setup & Dataset Downloader
# =============================================================================

set -e

# Setup directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
DATA_DIR="${WORKSPACE_DIR}/nvemb"

# --- 1. Python Environment Check ---
echo "[1/4] Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found."
    exit 1
fi
python_ver=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
echo "  - Found Python: ${python_ver}"

# --- 2. Installing Pinned Dependencies ---
echo "[2/4] Installing dependencies from requirements.txt..."
if [ -f "$REQ_FILE" ]; then
    pip install -r "$REQ_FILE"
else
    echo "Error: requirements.txt not found at ${REQ_FILE}"
    exit 1
fi

# --- 3. Downloading HuggingFace Nova Model ---
echo "[3/4] Fetching HuggingFace Nova Model snapshot..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('lt-asset/nova-1.3b')
"

# --- 4. Retrieving Bench Datasets ---
echo "[4/4] Retrieving dataset files..."
mkdir -p "$DATA_DIR"

TRAIN_ID="10UPIRm9jz7QEN7Bva3Jk9UkqzIccACHF"
TEST_ID="1hDzq7aZiuJPvJe09iaE3IKmirWBLpq-C"

TRAIN_FILE="${DATA_DIR}/output_benchset_rebalanced_train_nova.jsonl"
TEST_FILE="${DATA_DIR}/output_benchset_rebalanced_test_nova.jsonl"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "  - Downloading rebalanced train dataset..."
    gdown "$TRAIN_ID" -O "$TRAIN_FILE"
else
    echo "  - Train dataset already present (skipping)."
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "  - Downloading rebalanced test dataset..."
    gdown "$TEST_ID" -O "$TEST_FILE"
else
    echo "  - Test dataset already present (skipping)."
fi

echo "Setup completed successfully."
