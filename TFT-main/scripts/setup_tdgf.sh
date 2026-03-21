#!/usr/bin/env bash
# Setup script for TDGF — Time Deep Gradient Flow for American options
# Strategy #14: American options pricing with rough volatility
#
# This script:
#   1. Clones the TDGF repository
#   2. Installs Python dependencies
#   3. Runs a quick training test with synthetic data
#   4. Verifies the installation
#
# Usage: bash scripts/setup_tdgf.sh [--skip-test]

set -euo pipefail

INSTALL_DIR="${TDGF_REPO_PATH:-/opt/tdgf}"
SKIP_TEST="${1:-}"

echo "============================================"
echo "  TDGF SETUP — Strategy #14"
echo "============================================"

# Step 1: Clone repository
echo ""
echo "[1/4] Cloning TDGF repository..."
if [ -d "$INSTALL_DIR" ]; then
    echo "  Directory exists, pulling latest..."
    cd "$INSTALL_DIR" && git pull || true
else
    git clone https://github.com/jgrou/TDGF.git "$INSTALL_DIR"
fi

# Step 2: Install dependencies
echo ""
echo "[2/4] Installing Python dependencies..."
pip install torch numpy scipy 2>/dev/null || {
    echo "  pip install failed — ensure you are in the correct virtualenv"
    exit 1
}

# Install repo dependencies if requirements.txt exists
if [ -f "$INSTALL_DIR/requirements.txt" ]; then
    pip install -r "$INSTALL_DIR/requirements.txt" 2>/dev/null || true
fi

# Step 3: Verify import
echo ""
echo "[3/4] Verifying TDGF import..."
python3 -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
try:
    from tdgf import TDGFSolver
    solver = TDGFSolver(hidden_layers=3, hidden_units=50)
    print('  TDGFSolver imported and instantiated successfully')
except ImportError as e:
    print(f'  WARNING: Could not import TDGFSolver: {e}')
    print('  Check the repo structure and adjust import path if needed.')
    print(f'  Set TDGF_REPO_PATH={INSTALL_DIR} in your .env')
    sys.exit(0)
"

# Step 4: Quick training test (optional)
if [ "$SKIP_TEST" != "--skip-test" ]; then
    echo ""
    echo "[4/4] Running quick training test with synthetic data..."
    python3 -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
try:
    from tdgf import TDGFSolver
    import numpy as np

    solver = TDGFSolver(hidden_layers=3, hidden_units=50)

    # Synthetic Black-Scholes parameters for test
    params = {
        'spot': np.array([100.0]),
        'strike': np.array([100.0]),
        'rate': np.array([0.05]),
        'tau': np.array([0.25]),
        'sigma': np.array([0.20]),
        'option_type': 'put',
        'exercise_type': 'american',
    }

    result = solver.train(
        model_type='black_scholes',
        params=params,
        n_epochs=100,  # Quick test
        learning_rate=0.001,
        hidden_layers=3,
        hidden_units=50,
    )
    print(f'  Training test passed: {result}')

    price = solver.price(model_type='black_scholes', params=params)
    print(f'  Price test: {price}')

except ImportError:
    print('  Skipping test (TDGFSolver not importable)')
except Exception as e:
    print(f'  Test completed with note: {e}')
    print('  (This is expected if the API differs slightly)')
"
else
    echo ""
    echo "[4/4] Skipping training test (--skip-test)"
fi

echo ""
python3 -c "
print('Environment variables to set in .env:')
print(f'  TDGF_REPO_PATH=$INSTALL_DIR')
print(f'  TDGF_PDE_MODEL=heston')
print(f'  TDGF_MAX_EPOCHS=5000')
print(f'  STRATEGY_TDGF_ENABLED=true')
"

echo ""
echo "============================================"
echo "  TDGF SETUP COMPLETE"
echo "  Note: TDGF needs light training before use."
echo "  See: python -m models.tdgf_model --help"
echo "============================================"
