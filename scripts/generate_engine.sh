#!/bin/bash
# Generate TensorRT engine for current GPU
# Usage: ./generate_engine.sh [model_name]
# Example: ./generate_engine.sh yolo26s

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

echo "=== TensorRT Engine Generator ==="
echo "GPU: $GPU_NAME"
echo "Model: $MODEL"

# Detect compute capability
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
echo "Compute Capability: SM $SM"

# Check if ONNX exists, if not download
ONNX_FILE="$MODELS_DIR/${MODEL}.onnx"
if [ ! -f "$ONNX_FILE" ]; then
    echo "Downloading $MODEL.onnx from Ultralytics..."
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
if command -v trtexec &>/dev/null; then
    TRTEXEC="trtexec"
elif [ -f "$HOME/libs/TensorRT-RTX-1.3.0.35/bin/tensorrt_rtx" ]; then
    # Blackwell (SM 120) uses TensorRT-RTX
    TRTEXEC="$HOME/libs/TensorRT-RTX-1.3.0.35/bin/tensorrt_rtx"
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

# Generate engine
ENGINE_FILE="$MODELS_DIR/${MODEL}.engine"
echo "Generating engine: $ENGINE_FILE"

# TensorRT-RTX uses different flags
if [[ "$TRTEXEC" == *"tensorrt_rtx"* ]]; then
    $TRTEXEC \
        --onnx="$ONNX_FILE" \
        --saveEngine="$ENGINE_FILE" \
        --memPoolSize=workspace:4096M
else
    $TRTEXEC \
        --onnx="$ONNX_FILE" \
        --saveEngine="$ENGINE_FILE" \
        --fp16 \
        --workspace=4096
fi

echo ""
echo "=== Done ==="
echo "Engine saved: $ENGINE_FILE"
echo ""
echo "Benchmark:"
$TRTEXEC --loadEngine="$ENGINE_FILE" --warmUp=500 --iterations=100 2>&1 | grep -E "Throughput|Latency.*mean"
