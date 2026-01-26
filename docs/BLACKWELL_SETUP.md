# Blackwell GPU Setup Guide (RTX 50 Series)

This guide documents the specific setup requirements for NVIDIA Blackwell architecture GPUs (RTX 5060 Ti, 5070, 5080, 5090) with compute capability SM 120.

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

**Cause:** TensorRT engines are GPU-specific. An engine built for compute 7.5 (RTX 2060) won't work on SM 120 (RTX 5060 Ti).

**Solution:** Regenerate the engine with TensorRT-RTX:

```bash
# Use trtexec from TensorRT-RTX
~/libs/TensorRT-RTX-1.3.0.35/bin/trtexec \
    --onnx=models/yolo26s.onnx \
    --saveEngine=models/yolo26s.engine \
    --fp16

# Expected output:
# [01/25/2025-...] Latency: min = 3.5 ms, max = 4.2 ms, mean = 3.6 ms
# Throughput: ~280 FPS
```

---

## Summary: Complete Environment Setup

```bash
# ~/.zshrc additions for Blackwell

# CUDA 12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH

# Aria SLAM libs (TensorRT-RTX + OpenCV CUDA)
export LD_LIBRARY_PATH="$HOME/libs/TensorRT-RTX-1.3.0.35/lib:$HOME/libs/opencv_cuda/lib:$LD_LIBRARY_PATH"
```

## GPU Architecture Reference

| GPU Series | Compute Capability | CUDA Arch | Min CUDA Version |
|------------|-------------------|-----------|------------------|
| RTX 2060/2070/2080 | 7.5 | 75 | 10.0 |
| RTX 3060/3070/3080 | 8.6 | 86 | 11.1 |
| RTX 4060/4070/4080 | 8.9 | 89 | 11.8 |
| RTX 5060 Ti/5070/5080/5090 | 12.0 (SM 120) | 120 | **12.8** |

---

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

*Last updated: January 2025*
*Hardware: RTX 5060 Ti (Blackwell SM 120)*
