#!/usr/bin/env bash
# Setup script for Kronos — pre-trained financial time series foundation model
# Strategy #12: Stocks + Forex price forecasting
#
# This script:
#   1. Clones the Kronos repository
#   2. Installs Python dependencies
#   3. Downloads pre-trained models from HuggingFace
#   4. Verifies the installation
#
# Usage: bash scripts/setup_kronos.sh [--model mini|small|base]

set -euo pipefail

KRONOS_INSTALL_DIR="${KRONOS_REPO_PATH:-/opt/kronos}"
MODEL_SIZE="${1:-base}"

# Map model size to HuggingFace model name
case "$MODEL_SIZE" in
    --model)
        MODEL_SIZE="${2:-base}"
        ;;
esac

case "$MODEL_SIZE" in
    mini)
        HF_MODEL="NeoQuasar/Kronos-mini"
        echo "[Kronos] Using mini model (4.1M params)"
        ;;
    small)
        HF_MODEL="NeoQuasar/Kronos-small"
        echo "[Kronos] Using small model (24.7M params)"
        ;;
    base)
        HF_MODEL="NeoQuasar/Kronos-base"
        echo "[Kronos] Using base model (102.3M params)"
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE (use mini, small, or base)"
        exit 1
        ;;
esac

HF_TOKENIZER="NeoQuasar/Kronos-Tokenizer-base"

echo "============================================"
echo "  KRONOS SETUP — Strategy #12"
echo "============================================"

# Step 1: Clone repository
echo ""
echo "[1/4] Cloning Kronos repository..."
if [ -d "$KRONOS_INSTALL_DIR" ]; then
    echo "  Directory exists, pulling latest..."
    cd "$KRONOS_INSTALL_DIR" && git pull || true
else
    git clone https://github.com/shiyu-coder/Kronos.git "$KRONOS_INSTALL_DIR"
fi

# Step 2: Install dependencies
echo ""
echo "[2/4] Installing Python dependencies..."
pip install torch transformers huggingface_hub 2>/dev/null || {
    echo "  pip install failed — ensure you are in the correct virtualenv"
    exit 1
}

# Install Kronos repo dependencies if requirements.txt exists
if [ -f "$KRONOS_INSTALL_DIR/requirements.txt" ]; then
    pip install -r "$KRONOS_INSTALL_DIR/requirements.txt" 2>/dev/null || true
fi

# Step 3: Download pre-trained models
echo ""
echo "[3/4] Downloading pre-trained models from HuggingFace..."
python3 -c "
from huggingface_hub import snapshot_download
print('  Downloading model: $HF_MODEL')
snapshot_download('$HF_MODEL', local_dir=None)
print('  Downloading tokenizer: $HF_TOKENIZER')
snapshot_download('$HF_TOKENIZER', local_dir=None)
print('  Models cached successfully.')
"

# Step 4: Verify installation
echo ""
echo "[4/4] Verifying installation..."
python3 -c "
import sys
sys.path.insert(0, '$KRONOS_INSTALL_DIR')
try:
    from kronos import KronosPredictor
    print('  KronosPredictor imported successfully')
except ImportError as e:
    print(f'  WARNING: Could not import KronosPredictor: {e}')
    print('  The model files are downloaded but the import path may need adjustment.')
    print('  Set KRONOS_REPO_PATH=$KRONOS_INSTALL_DIR in your .env')

print()
print('Environment variables to set in .env:')
print(f'  KRONOS_REPO_PATH=$KRONOS_INSTALL_DIR')
print(f'  KRONOS_MODEL_NAME=$HF_MODEL')
print(f'  KRONOS_TOKENIZER_NAME=$HF_TOKENIZER')
print(f'  STRATEGY_KRONOS_ENABLED=true')
"

echo ""
echo "============================================"
echo "  KRONOS SETUP COMPLETE"
echo "============================================"
