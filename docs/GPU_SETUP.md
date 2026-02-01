# GPU Setup Guide

This guide documents CUDA/TensorRT setup for different NVIDIA GPUs and the multi-GPU workflow.

## Supported GPUs

| GPU Series | Compute Capability | CUDA Arch | Min CUDA | TensorRT |
|------------|-------------------|-----------|----------|----------|
| RTX 2060/2070/2080 | 7.5 | sm75 | 10.0 | Standard |
| RTX 3060/3070/3080 | 8.6 | sm86 | 11.1 | Standard |
| Jetson Orin Nano | 8.7 | sm87 | 11.4 | Standard |
| RTX 4060/4070/4080 | 8.9 | sm89 | 11.8 | Standard |
| RTX 5060 Ti/5070/5080/5090 | 12.0 | sm120 | **12.8** | **TensorRT-RTX** |

---

## Multi-GPU Workflow

If you work on multiple machines (e.g., RTX 2060 + RTX 5060 Ti + Jetson):

**Portable (commit to git):**
- `models/yolo26s.onnx` - ONNX is universal

**Machine-specific (regenerate):**
- `models/yolo26s_sm75.engine` - RTX 2060
- `models/yolo26s_sm87.engine` - Jetson Orin Nano
- `models/yolo26s_sm120.engine` - RTX 5060 Ti

**Workflow on new machine:**
```bash
git pull                           # Get ONNX
./scripts/generate_engine.sh       # Auto-detect GPU, create engine + symlink
cd build && cmake .. && make -j$(nproc)
./aria_slam
```

The symlink `yolo26s.engine → yolo26s_sm{XX}.engine` ensures code works without modification.

---

## Blackwell-Specific Issues (RTX 50 Series)

## Overview

Blackwell GPUs require bleeding-edge software versions. As of January 2025, standard installations will fail with cryptic errors.

**Tested configuration:**
- GPU: RTX 5060 Ti (SM 120)
- Ubuntu 24.04
- CUDA 12.8
- TensorRT-RTX 1.3
- OpenCV 4.9.0 (compiled from source)

---

## Problem 1: CUDA Version

**Symptom:**
```
nvcc fatal: Unsupported gpu architecture 'compute_120'
```

**Cause:** CUDA versions before 12.8 don't support SM 120 (Blackwell).

**Solution:**
```bash
# Remove old CUDA
sudo apt remove --purge cuda* nvidia-cuda*

# Install CUDA 12.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-8

# Add to ~/.zshrc or ~/.bashrc
export PATH=/usr/local/cuda-12.8/bin:$PATH
```

---

## Problem 2: TensorRT Doesn't Support Blackwell

**Symptom:**
```
[TRT] Unsupported SM: 120
```
or
```
CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage...
```
followed by crashes.

**Cause:** Standard TensorRT (even version 10.14) does not support Blackwell GPUs. NVIDIA has a separate product called **TensorRT-RTX** specifically for RTX GPUs including Blackwell.

**Solution:**

1. Download TensorRT-RTX from NVIDIA:
   - Go to https://developer.nvidia.com/tensorrt/download
   - Select "TensorRT-RTX" (not regular TensorRT)
   - Download: `TensorRT-RTX-1.3.0.35-Linux-x86_64-cuda-12.9-Release-external.tar.gz`

2. Install:
```bash
cd ~/libs
tar -xzf ~/Downloads/TensorRT-RTX-1.3.0.35-Linux-x86_64-cuda-12.9-Release-external.tar.gz

# Add to ~/.zshrc
export LD_LIBRARY_PATH="$HOME/libs/TensorRT-RTX-1.3.0.35/lib:$LD_LIBRARY_PATH"
```

3. Update CMakeLists.txt:
```cmake
# TensorRT-RTX (for Blackwell support)
set(TensorRT_DIR "/home/user/libs/TensorRT-RTX-1.3.0.35")
set(TensorRT_INCLUDE_DIRS "${TensorRT_DIR}/include")
set(TensorRT_LIBS "${TensorRT_DIR}/lib")

link_directories(${TensorRT_LIBS})

target_link_libraries(your_target
    tensorrt_rtx  # Note: tensorrt_rtx, not nvinfer
    cudart
)
```

---

## Problem 3: OpenCV CUDA Kernel Errors

**Symptom:**
```
OpenCV(4.9.0) error: (-217:Gpu API call) no kernel image is available for execution on the device
```

**Cause:** Pre-built OpenCV or OpenCV compiled for older GPU architectures won't work.

**Solution:** Recompile OpenCV with `CUDA_ARCH_BIN=12.0`:

```bash
cd ~/libs/opencv/build

# Clean previous build
rm -rf *

# Configure for Blackwell (SM 120 = arch 12.0)
cmake .. \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=~/libs/opencv_cuda \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=OFF \
    -DENABLE_FAST_MATH=ON \
    -DCUDA_FAST_MATH=ON \
    -DWITH_CUBLAS=ON \
    -DCUDA_ARCH_BIN=12.0 \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
    -DOPENCV_EXTRA_MODULES_PATH=~/libs/opencv_contrib/modules \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF

# Compile (takes ~20-30 min)
make -j$(nproc) && make install
```

---

## Problem 4: TensorRT Engine Incompatible

**Symptom:**
```
[TRT] The engine was built with a different compute capability
```
or engine loads but inference produces garbage/crashes.

**Cause:** TensorRT engines are GPU-specific. An engine built for SM 7.5 (RTX 2060) won't work on SM 120 (RTX 5060 Ti).

**Solution:** Use the project script (auto-detects GPU and TensorRT version):

```bash
./scripts/generate_engine.sh yolo26s

# Creates:
#   models/yolo26s_sm120.engine  (for your GPU)
#   models/yolo26s.engine → symlink
```

---

## Jetson Orin Nano Setup

The Jetson Orin Nano has a powerful GPU (SM 8.7) but a weak CPU (6-core ARM Cortex-A78AE).

**JetPack includes everything:**
- CUDA, TensorRT, OpenCV CUDA pre-installed
- No need to compile OpenCV or download TensorRT

**Critical considerations:**
- **H13 (Multithreading) is essential** - CPU bottleneck requires async pipeline
- Loop closure g2o optimization blocks main thread → must run in separate thread
- YOLO inference should run async to not block ORB processing

**Setup:**
```bash
# JetPack 6.x comes with everything
# Just generate the engine for SM 8.7
./scripts/generate_engine.sh yolo26s

# Creates: models/yolo26s_sm87.engine
```

**Performance expectations:**
- GPU: Comparable to RTX 2060 for inference
- CPU: Much slower than desktop → multithreading critical
- Target: 15-20 FPS (vs 30+ on desktop)

---

## Summary: Complete Environment Setup

```bash
# ~/.zshrc additions for Blackwell

# CUDA 12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH

# Aria SLAM libs (TensorRT-RTX + OpenCV CUDA)
export LD_LIBRARY_PATH="$HOME/libs/TensorRT-RTX-1.3.0.35/lib:$HOME/libs/opencv_cuda/lib:$LD_LIBRARY_PATH"
```

## Verification Commands

```bash
# Check CUDA version
nvcc --version

# Check GPU info
nvidia-smi

# Test CUDA compilation for Blackwell
echo '__global__ void k(){}' > test.cu && nvcc -arch=sm_120 test.cu -o test && rm test test.cu && echo "SM 120 OK"

# Test OpenCV CUDA
python3 -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"

# Test aria_slam
cd ~/Projects/aria/aria-slam/build
./aria_slam
```

---

*Last updated: February 2026*
