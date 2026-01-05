#!/bin/bash
# Drone Hunter Edge - Termux Setup Script
#
# This script sets up the edge deployment environment on Termux using uv.
#
# Prerequisites:
# - Termux installed from F-Droid (not Play Store)
# - Run: termux-setup-storage (to access shared storage)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -e

echo "=================================="
echo "Drone Hunter Edge - Termux Setup"
echo "=================================="

# Update package manager
echo "[1/6] Updating packages..."
pkg update -y
pkg upgrade -y

# Install required packages
echo "[2/6] Installing system dependencies..."
pkg install -y python clang cmake ninja git curl

# Install uv if not present
echo "[3/6] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment and install dependencies
echo "[4/6] Creating virtual environment..."
uv venv .venv
source .venv/bin/activate

echo "[5/6] Installing Python packages..."
uv pip install numpy pillow onnxruntime

# Optional: NCNN (uncomment if needed)
# uv pip install ncnn

echo "[6/6] Verifying installation..."
python -c "
import numpy as np
from PIL import Image
import onnxruntime as ort

print(f'NumPy: {np.__version__}')
print(f'Pillow: {Image.__version__}')
print(f'ONNX Runtime: {ort.__version__}')
print(f'ONNX Providers: {ort.get_available_providers()}')
print('All packages installed successfully!')
"

# Create models directory
mkdir -p models

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo ""
echo "2. Copy your models to the 'models/' directory:"
echo "   - nanodet_drone.onnx (detector)"
echo "   - policy.onnx (trained agent)"
echo "   - normalization.json (observation stats)"
echo ""
echo "3. Run the simulation:"
echo "   python run_simulation.py --detector models/nanodet_drone.onnx \\"
echo "                           --policy models/policy.onnx \\"
echo "                           --normalization models/normalization.json"
echo ""
echo "4. Or run in oracle mode (no detector):"
echo "   python run_simulation.py --oracle --policy models/policy.onnx"
echo ""
