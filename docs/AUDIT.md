# ARIA SLAM - Technical Overview

**Actualizado:** H01-H14 completados
**Version:** Fase 1 completa

---

## Stack Tecnologico

| Tecnologia | Version | Proposito |
|------------|---------|-----------|
| C++ | 17 | Lenguaje principal |
| OpenCV | 4.9.0 (CUDA) | Vision, features, matching |
| CUDA Toolkit | 12.6+ | GPU computing |
| TensorRT | 10.7+ | YOLO inference |
| Eigen | 3.3+ | Algebra lineal |
| g2o | Sistema | Pose graph optimization |

---

## Librerias Utilizadas

### OpenCV 4.9.0 (CUDA)

```cpp
#include <opencv2/opencv.hpp>          // Core
#include <opencv2/cudafeatures2d.hpp>  // GPU ORB
#include <opencv2/cuda.hpp>            // GpuMat
```

**Uso:**
- `cv::cuda::ORB` - Feature detection GPU
- `cv::cuda::DescriptorMatcher` - Matching GPU
- `cv::cuda::GpuMat` - Memoria GPU
- `cv::findEssentialMat` - Geometria epipolar
- `cv::recoverPose` - Descomposicion R, t

### CUDA Toolkit

```cpp
#include <cuda_runtime_api.h>
```

**Uso:**
- `cudaStream_t` - Streams paralelos
- `cudaMalloc/cudaFree` - Memoria GPU
- `cudaMemcpyAsync` - Transferencias async
- `cudaStreamSynchronize` - Sincronizacion

### TensorRT 10.7

```cpp
#include <NvInfer.h>
```

**Uso:**
- `IRuntime` - Runtime TensorRT
- `ICudaEngine` - Motor optimizado
- `IExecutionContext` - Contexto de inferencia
- `enqueueV3` - Inferencia async

### Eigen 3.3+

```cpp
#include <Eigen/Dense>
```

**Uso:**
- `Eigen::Vector3d` - Posicion, velocidad
- `Eigen::Matrix3d` - Rotacion
- `Eigen::Matrix4d` - Pose SE3
- `Eigen::Quaterniond` - Orientacion
- `Eigen::Matrix<double, 15, 15>` - EKF covarianza

### g2o

```cpp
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
```

**Uso:**
- `VertexSE3` - Poses de keyframes
- `EdgeSE3` - Restricciones odometria/loop
- `OptimizationAlgorithmLevenberg` - Solver

---

## Arquitectura de Clases

### Frame (H02, H05)

```cpp
class Frame {
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::cuda::GpuMat gpu_descriptors;
};
```

### TRTInference (H06)

```cpp
class TRTInference {
    nvinfer1::IExecutionContext* context_;
    void* buffers_[2];
    cudaStream_t stream_;

    void detectAsync(cv::Mat, cudaStream_t);
    std::vector<Detection> getDetections();
};
```

### IMU (H08)

```cpp
class IMU {
    Eigen::Matrix<double, 15, 1> state_;  // pos, vel, quat, bias_a, bias_g
    Eigen::Matrix<double, 15, 15> P_;     // Covarianza

    void predict(ImuMeasurement);
    void updateVO(Pose);
};
```

### LoopClosure (H09, H10, H14)

```cpp
class LoopClosure {
    std::vector<KeyFrame> keyframes_;
    cv::cuda::GpuMat gpu_descriptors_;  // H14: GPU database
    g2o::SparseOptimizer optimizer_;

    bool detect(KeyFrame& query);
    void optimizePoseGraph();
};
```

### EuRoCReader (H07)

```cpp
class EuRoCReader {
    std::vector<ImageEntry> images_;
    std::vector<ImuEntry> imu_data_;
    std::vector<GroundTruth> ground_truth_;

    bool getNextSynchronized(cv::Mat&, std::vector<ImuMeasurement>&);
};
```

---

## Pipeline Principal (main.cpp)

```
1. Capturar frame
2. Lanzar en paralelo (H11):
   - Stream 1: ORB GPU
   - Stream 2: YOLO TensorRT
3. Sincronizar streams
4. Filtrar keypoints en objetos dinamicos (H06)
5. Matching con frame anterior
6. Estimar pose (Essential Matrix + RANSAC)
7. Acumular trayectoria
8. Visualizar
```

---

## Performance

| Operacion | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| ORB (1000 pts) | 50ms | 10ms | 5x |
| YOLO inference | 100ms | 5ms | 20x |
| BF Match | 15ms | 2ms | 7.5x |
| **Pipeline total** | ~165ms | ~17ms | **~10x** |

**FPS:** ~60-80 con CUDA streams

---

## Archivos del Proyecto

```
src/
  main.cpp          # Pipeline principal
  Frame.cpp         # ORB GPU wrapper
  TRTInference.cpp  # YOLO TensorRT
  IMU.cpp           # EKF sensor fusion
  LoopClosure.cpp   # Loop detection + pose graph
  EuRoCReader.cpp   # Dataset reader
  Mapper.cpp        # 3D mapping

include/
  Frame.hpp
  TRTInference.hpp
  IMU.hpp
  LoopClosure.hpp
  EuRoCReader.hpp
  Mapper.hpp

models/
  yolo26s.engine    # TensorRT engine

docs/
  milestones/       # H01-H14 AUDIT docs
  PIPELINE_DIAGRAM.md
  STUDY_GUIDE.md
  BLACKWELL_SETUP.md
```

---

## Milestones

| # | Nombre | Estado |
|---|--------|--------|
| H01 | Setup | Completado |
| H02 | Feature Extraction | Completado |
| H03 | Feature Matching | Completado |
| H04 | Pose Estimation | Completado |
| H05 | OpenCV CUDA | Completado |
| H06 | TensorRT YOLO | Completado |
| H07 | EuRoC Dataset | Completado |
| H08 | Sensor Fusion | Completado |
| H09 | Loop Closure | Completado |
| H10 | Pose Graph | Completado |
| H11 | CUDA Streams | Completado |
| H12 | Clean Architecture | Design doc |
| H13 | Multithreading | Completado |
| H14 | GPU Loop Closure | Completado |
