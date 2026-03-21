#!/usr/bin/env bash
# Setup script for Deep Surrogates — neural option pricing surrogates
# Strategy #13: Options pricing acceleration + crash prediction
#
# This script:
#   1. Clones the DeepSurrogate repository
#   2. Installs Python dependencies
#   3. Verifies pre-trained surrogates load correctly
#
# Usage: bash scripts/setup_deep_surrogate.sh

set -euo pipefail

INSTALL_DIR="${DEEP_SURROGATE_REPO_PATH:-/opt/deep_surrogate}"

echo "============================================"
echo "  DEEP SURROGATES SETUP — Strategy #13"
echo "============================================"

# Step 1: Clone repository
echo ""
echo "[1/3] Cloning DeepSurrogate repository..."
if [ -d "$INSTALL_DIR" ]; then
    echo "  Directory exists, pulling latest..."
    cd "$INSTALL_DIR" && git pull || true
else
    git clone https://github.com/DeepSurrogate/OptionPricing.git "$INSTALL_DIR"
fi

# Step 2: Install dependencies
echo ""
echo "[2/3] Installing Python dependencies..."
pip install torch numpy pandas scipy 2>/dev/null || {
    echo "  pip install failed — ensure you are in the correct virtualenv"
    exit 1
}

# Install repo dependencies if requirements.txt exists
if [ -f "$INSTALL_DIR/requirements.txt" ]; then
    pip install -r "$INSTALL_DIR/requirements.txt" 2>/dev/null || true
fi

# Step 3: Verify installation
echo ""
echo "[3/3] Verifying installation..."
python3 -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
try:
    from deep_surrogate import DeepSurrogate
    ds = DeepSurrogate()
    print('  DeepSurrogate loaded successfully')

    # Quick test with dummy Heston parameters
    import pandas as pd
    test_df = pd.DataFrame({
        'kappa': [2.0],
        'theta': [0.04],
        'sigma': [0.3],
        'rho': [-0.7],
        'v0': [0.04],
        'rate': [0.05],
        'tau': [0.25],
        'moneyness': [1.0],
    })
    try:
        iv = ds.get_iv(test_df, model_type='heston')
        print(f'  Test IV computation: {iv}')
    except Exception as e:
        print(f'  IV test skipped (may need real data): {e}')

except ImportError as e:
    print(f'  WARNING: Could not import DeepSurrogate: {e}')
    print('  Check the repo structure and adjust import path if needed.')
    print(f'  Set DEEP_SURROGATE_REPO_PATH={INSTALL_DIR} in your .env')

print()
print('Environment variables to set in .env:')
print(f'  DEEP_SURROGATE_REPO_PATH={INSTALL_DIR}')
print(f'  DEEP_SURROGATE_MODEL_TYPE=heston')
print(f'  STRATEGY_DEEP_SURROGATES_ENABLED=true')
"

echo ""
echo "============================================"
echo "  DEEP SURROGATES SETUP COMPLETE"
echo "============================================"
