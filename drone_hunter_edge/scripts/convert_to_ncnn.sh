#!/bin/bash
# Convert ONNX models to NCNN format for mobile deployment.
#
# Prerequisites:
# - ncnn tools installed (onnx2ncnn, ncnnoptimize)
# - Install from: https://github.com/Tencent/ncnn
#
# Usage:
#   ./convert_to_ncnn.sh models/nanodet_drone.onnx
#   ./convert_to_ncnn.sh models/policy.onnx

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.onnx> [output_prefix]"
    echo ""
    echo "Example:"
    echo "  $0 models/nanodet_drone.onnx"
    echo "  $0 models/policy.onnx models/policy_ncnn"
    exit 1
fi

INPUT_ONNX="$1"
OUTPUT_PREFIX="${2:-${INPUT_ONNX%.onnx}}"

# Check if input exists
if [ ! -f "$INPUT_ONNX" ]; then
    echo "Error: Input file not found: $INPUT_ONNX"
    exit 1
fi

# Check for onnx2ncnn
if ! command -v onnx2ncnn &> /dev/null; then
    echo "Error: onnx2ncnn not found."
    echo ""
    echo "Install ncnn tools:"
    echo "  git clone https://github.com/Tencent/ncnn.git"
    echo "  cd ncnn && mkdir build && cd build"
    echo "  cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_BUILD_TOOLS=ON .."
    echo "  make -j\$(nproc)"
    echo "  sudo make install"
    exit 1
fi

echo "Converting: $INPUT_ONNX"
echo "Output prefix: $OUTPUT_PREFIX"
echo ""

# Step 1: Convert ONNX to NCNN
echo "[1/3] Converting ONNX to NCNN..."
onnx2ncnn "$INPUT_ONNX" "${OUTPUT_PREFIX}.param" "${OUTPUT_PREFIX}.bin"

# Step 2: Optimize the model
echo "[2/3] Optimizing NCNN model..."
if command -v ncnnoptimize &> /dev/null; then
    ncnnoptimize "${OUTPUT_PREFIX}.param" "${OUTPUT_PREFIX}.bin" \
                 "${OUTPUT_PREFIX}_opt.param" "${OUTPUT_PREFIX}_opt.bin" 1

    # Replace with optimized version
    mv "${OUTPUT_PREFIX}_opt.param" "${OUTPUT_PREFIX}.param"
    mv "${OUTPUT_PREFIX}_opt.bin" "${OUTPUT_PREFIX}.bin"
else
    echo "Warning: ncnnoptimize not found, skipping optimization"
fi

# Step 3: Show model info
echo "[3/3] Model info:"
echo "  Param file: ${OUTPUT_PREFIX}.param"
echo "  Bin file:   ${OUTPUT_PREFIX}.bin"

# Show file sizes
PARAM_SIZE=$(du -h "${OUTPUT_PREFIX}.param" | cut -f1)
BIN_SIZE=$(du -h "${OUTPUT_PREFIX}.bin" | cut -f1)
echo "  Param size: $PARAM_SIZE"
echo "  Bin size:   $BIN_SIZE"

echo ""
echo "Conversion complete!"
echo ""
echo "To use with NCNN backend:"
echo "  python run_simulation.py --detector ${OUTPUT_PREFIX}.param --backend ncnn"
