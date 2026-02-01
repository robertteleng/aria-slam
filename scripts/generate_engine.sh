#!/bin/bash
# Generate TensorRT engine for current GPU
# Usage: ./generate_engine.sh [model_name]
# Example: ./generate_engine.sh yolo26s
#
# Generates: models/yolo26s_sm75.engine  (for RTX 2060)
#            models/yolo26s_sm120.engine (for RTX 5060 Ti)

set -e

MODEL=${1:-yolo26s}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

mkdir -p "$MODELS_DIR"

# Detect GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if [ -z "$GPU_NAME" ]; then
    echo "Error: No NVIDIA GPU detected"
    exit 1
fi

# Detect compute capability (e.g., 7.5 -> 75, 12.0 -> 120)
SM_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
SM=$(echo "$SM_RAW" | tr -d '.')

echo "=== TensorRT Engine Generator ==="
echo "GPU: $GPU_NAME"
echo "Compute Capability: SM $SM ($SM_RAW)"
echo "Model: $MODEL"

# Check if ONNX exists
ONNX_FILE="$MODELS_DIR/${MODEL}.onnx"
if [ ! -f "$ONNX_FILE" ]; then
    echo ""
    echo "ONNX file not found: $ONNX_FILE"
    echo "Downloading $MODEL from Ultralytics..."
    pip install ultralytics -q 2>/dev/null || true
    python3 -c "
from ultralytics import YOLO
model = YOLO('${MODEL}.pt')
model.export(format='onnx', imgsz=640, simplify=True)
import shutil
shutil.move('${MODEL}.onnx', '$ONNX_FILE')
"
fi

# Find trtexec
TRTEXEC=""
if [ -f "$HOME/libs/TensorRT-RTX-1.3.0.35/bin/trtexec" ]; then
    # TensorRT-RTX for Blackwell (SM 120)
    TRTEXEC="$HOME/libs/TensorRT-RTX-1.3.0.35/bin/trtexec"
elif command -v trtexec &>/dev/null; then
    TRTEXEC="trtexec"
elif [ -f "/usr/src/tensorrt/bin/trtexec" ]; then
    # Jetson
    TRTEXEC="/usr/src/tensorrt/bin/trtexec"
else
    # Search in common locations
    for dir in "$HOME/libs/TensorRT"*/bin /usr/local/tensorrt/bin; do
        if [ -f "$dir/trtexec" ]; then
            TRTEXEC="$dir/trtexec"
            break
        fi
    done
fi

if [ -z "$TRTEXEC" ]; then
    echo "Error: trtexec not found. Install TensorRT or set path manually."
    exit 1
fi

echo "Using: $TRTEXEC"

# Engine file with SM suffix
ENGINE_FILE="$MODELS_DIR/${MODEL}_sm${SM}.engine"
echo ""
echo "Generating engine: $ENGINE_FILE"

# Generate engine (new syntax for TensorRT 10.x)
$TRTEXEC \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    --fp16 \
    --memPoolSize=workspace:4096MiB

# Create symlink to generic name for backwards compatibility
GENERIC_ENGINE="$MODELS_DIR/${MODEL}.engine"
rm -f "$GENERIC_ENGINE"
ln -sf "${MODEL}_sm${SM}.engine" "$GENERIC_ENGINE"

echo ""
echo "=== Done ==="
echo "Engine saved: $ENGINE_FILE"
echo "Symlink: $GENERIC_ENGINE -> ${MODEL}_sm${SM}.engine"
echo ""
echo "Benchmark:"
$TRTEXEC --loadEngine="$ENGINE_FILE" --warmUp=500 --iterations=100 2>&1 | grep -E "Throughput|Latency.*mean" || true
