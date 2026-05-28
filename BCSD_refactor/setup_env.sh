#!/bin/bash
# =============================================================================
# setup_env.sh — Minimalist, Silent Environment Setup & Dataset Downloader
# =============================================================================

set -e

# Setup directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
DATA_DIR="${WORKSPACE_DIR}/nvemb"

# --- 1. Python Environment Check ---
echo -n "[1/4] Checking Python environment... "
if ! command -v python3 &> /dev/null; then
    echo "FAILED"
    exit 1
fi
echo "OK"

# --- 2. Installing Pinned Dependencies ---
echo -n "[2/4] Installing packages... "
if [ -f "$REQ_FILE" ]; then
    pip install -q -r "$REQ_FILE"
    echo "OK"
else
    echo "FAILED (requirements.txt missing)"
    exit 1
fi

# --- 3. Downloading HuggingFace Nova Model ---
echo -n "[3/4] Fetching HuggingFace Nova Model... "
python3 -c "
import os
import logging
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
from huggingface_hub import snapshot_download
snapshot_download('lt-asset/nova-1.3b', disable_tqdm=True)
" &>/dev/null
echo "OK"

# --- 4. Retrieving Bench Datasets ---
echo -n "[4/4] Retrieving dataset files... "
mkdir -p "$DATA_DIR"

TRAIN_ID="10UPIRm9jz7QEN7Bva3Jk9UkqzIccACHF"
TEST_ID="1hDzq7aZiuJPvJe09iaE3IKmirWBLpq-C"

TRAIN_FILE="${DATA_DIR}/output_benchset_rebalanced_train_nova.jsonl"
TEST_FILE="${DATA_DIR}/output_benchset_rebalanced_test_nova.jsonl"

if [ ! -f "$TRAIN_FILE" ]; then
    gdown -q "$TRAIN_ID" -O "$TRAIN_FILE"
fi

if [ ! -f "$TEST_FILE" ]; then
    gdown -q "$TEST_ID" -O "$TEST_FILE"
fi
echo "OK"

echo "Setup completed successfully."
