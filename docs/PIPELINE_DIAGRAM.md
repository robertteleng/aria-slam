# Diagrama del Pipeline aria-slam

**Actualizado:** H01-H14 completados

---

## Vista General

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ARIA-SLAM PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌─────────────────────────────────────────────────────┐    │
│  │  Video  │───>│              PARALLEL GPU PROCESSING                │    │
│  │  Input  │    │  ┌─────────────┐         ┌─────────────┐            │    │
│  └─────────┘    │  │ Stream 1    │         │ Stream 2    │            │    │
│                 │  │ ORB CUDA    │         │ YOLO TRT    │            │    │
│                 │  │ (features)  │         │ (objects)   │            │    │
│                 │  └──────┬──────┘         └──────┬──────┘            │    │
│                 └─────────┼───────────────────────┼───────────────────┘    │
│                           │                       │                         │
│                           └───────────┬───────────┘                         │
│                                       ▼                                     │
│                         ┌─────────────────────────┐                         │
│                         │   DYNAMIC FILTERING     │                         │
│                         │ (remove features on     │                         │
│                         │  moving objects)        │                         │
│                         └────────────┬────────────┘                         │
│                                      ▼                                      │
│                         ┌─────────────────────────┐                         │
│                         │      MATCHING           │                         │
│                         │  (GPU BFMatcher +       │                         │
│                         │   Lowe's ratio test)    │                         │
│                         └────────────┬────────────┘                         │
│                                      ▼                                      │
│                         ┌─────────────────────────┐                         │
│                         │   POSE ESTIMATION       │                         │
│                         │  (Essential Matrix +    │                         │
│                         │   RANSAC)               │                         │
│                         └────────────┬────────────┘                         │
│                                      ▼                                      │
│                         ┌─────────────────────────┐                         │
│                         │   TRAJECTORY            │──────> Visualization    │
│                         │   ACCUMULATION          │                         │
│                         └─────────────────────────┘                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Detallado por Frame

```
================================================================================
                           FRAME N (entrada)
================================================================================

                           +-------------+
                           |   cv::Mat   |
                           |    frame    |
                           +------+------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v                                                 v
+------------------+                             +------------------+
|  CUDA STREAM 1   |                             |  CUDA STREAM 2   |
|   (stream_orb)   |                             |   (stream_yolo)  |
+------------------+                             +------------------+
         |                                                 |
         v                                                 v
+------------------+                             +------------------+
|  Frame::Frame()  |                             | yolo->detectAsync|
+------------------+                             +------------------+
         |                                                 |
         v                                                 v
+------------------+                             +------------------+
| 1. BGR -> Gray   |                             | 1. Resize 640x640|
| 2. CPU -> GPU    |                             | 2. BGR -> RGB    |
| 3. ORB detect    |                             | 3. Normalize 0-1 |
|    (async)       |                             | 4. HWC -> CHW    |
+------------------+                             | 5. CPU -> GPU    |
         |                                       | 6. TRT inference |
         |                                       |    (async)       |
         |                                       | 7. GPU -> CPU    |
         |                                       +------------------+
         |                                                 |
         +------------------------+------------------------+
                                  |
                                  v
                     +------------------------+
                     | cudaStreamSynchronize  |
                     |   (esperar ambos)      |
                     +------------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v                                                 v
+------------------+                             +------------------+
| downloadResults()|                             | getDetections()  |
+------------------+                             +------------------+
         |                                                 |
         v                                                 v
+------------------+                             +------------------+
| - keypoints[]    |                             | - Detection[]    |
| - descriptors    |                             |   - box (x,y,w,h)|
| - gpu_descriptors|                             |   - confidence   |
+------------------+                             |   - class_id     |
                                                 +------------------+


================================================================================
                     DYNAMIC OBJECT FILTERING (H06)
================================================================================

+------------------+      +------------------+
|    keypoints     |      |   detections     |
|    (features)    |      | (YOLO bboxes)    |
+--------+---------+      +---------+--------+
         |                          |
         +------------+-------------+
                      |
                      v
             +------------------+
             | isInDynamicObject|
             |                  |
             | DYNAMIC_CLASSES: |
             | - person (0)     |
             | - bicycle (1)    |
             | - car (2)        |
             | - motorcycle (3) |
             | - bus (5)        |
             | - truck (7)      |
             | - bird (14)      |
             | - cat (15)       |
             | - dog (16)       |
             +--------+---------+
                      |
         +------------+-------------+
         |                          |
         v                          v
+------------------+      +------------------+
|  STATIC features |      | FILTERED features|
|  (used for pose) |      |  (discarded)     |
+------------------+      +------------------+


================================================================================
                         MATCHING (si hay frame previo)
================================================================================

    Frame N-1                                    Frame N
+------------------+                      +------------------+
| gpu_descriptors  |                      | gpu_descriptors  |
+--------+---------+                      +---------+--------+
         |                                          |
         +----------------+    +--------------------+
                          |    |
                          v    v
                  +------------------+
                  |   BFMatcher GPU  |
                  |   knnMatch(k=2)  |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  |   Lowe's Ratio   |
                  |   Test (0.75)    |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  |  good_matches[]  |
                  +------------------+


================================================================================
                         POSE ESTIMATION (si matches >= 8)
================================================================================

+------------------+     +------------------+
|  pts1 (frame N-1)|     |  pts2 (frame N)  |
|  [Point2f, ...]  |     |  [Point2f, ...]  |
+--------+---------+     +---------+--------+
         |                         |
         +------------+------------+
                      |
                      v
            +------------------+
            |    Camera K      |
            | [fx  0  cx]      |
            | [ 0 fy  cy]      |
            | [ 0  0   1]      |
            +--------+---------+
                     |
                     v
            +------------------+
            | findEssentialMat |
            |    + RANSAC      |
            +--------+---------+
                     |
                     v
            +------------------+
            |  Essential Mat E |
            |    (3x3)         |
            +--------+---------+
                     |
                     v
            +------------------+
            |   recoverPose    |
            +--------+---------+
                     |
         +-----------+-----------+
         |                       |
         v                       v
+------------------+    +------------------+
|   Rotation R     |    |  Translation t   |
|     (3x3)        |    |     (3x1)        |
+------------------+    +------------------+


================================================================================
                           POSE ACCUMULATION
================================================================================

Estado anterior:                 Movimiento relativo:
+------------------+            +------------------+
|  position (3x1)  |            |      t (3x1)     |
|  rotation (3x3)  |            |      R (3x3)     |
+--------+---------+            +---------+--------+
         |                                |
         +--------------------------------+
                      |
                      v
            +------------------+
            | position +=      |
            |   rotation * t   |
            |                  |
            | rotation =       |
            |   R * rotation   |
            +--------+---------+
                     |
                     v
            +------------------+
            | Nueva posicion   |
            | en coord. mundo  |
            +------------------+


================================================================================
                           VISUALIZACION
================================================================================

+------------------+     +------------------+     +------------------+
|    Trajectory    |     |     Display      |     |    Estadisticas  |
|                  |     |                  |     |                  |
|  o               |     | +--[person]--+   |     |  FPS: 80         |
|   o              |     | |  FILTERED  |   |     |  Matches: 245    |
|    o             |     | +------------+   |     |  Objects: 3      |
|     o  <- pos    |     |    * * *         |     |  Filtered: 12    |
|                  |     |   *     *        |     |                  |
+------------------+     +------------------+     +------------------+
```

---

## Timeline de Ejecucion (1 frame)

```
Tiempo (ms)   0    2    4    6    8   10   12   14
              |----|----|----|----|----|----|----|

CPU:          [CAP]                    [FILT][MATCH][POSE][VIS]
               2ms                      1ms   2ms    1ms   1ms

GPU Stream 1: ..... [----ORB----]
                         10ms

GPU Stream 2: ..... [--YOLO--]
                       5ms

              |<---- paralelo ---->|
                                   ^
                                   |
                            cudaStreamSync

Total: ~14ms = ~70 FPS
```

---

## Arquitectura de Componentes Implementados (H01-H14)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ARIA-SLAM SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         FRONTEND (Real-time)                           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │ │
│  │  │    Frame     │  │ TRTInference │  │    IMU       │                  │ │
│  │  │   (H02-H05)  │  │   (H06)      │  │   (H08)      │                  │ │
│  │  │              │  │              │  │              │                  │ │
│  │  │ - ORB CUDA   │  │ - YOLO26s    │  │ - Preintegr. │                  │ │
│  │  │ - GPU desc   │  │ - TensorRT   │  │ - EKF fusion │                  │ │
│  │  │ - Async      │  │ - Async      │  │ - Bias est.  │                  │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │ │
│  │         │                 │                 │                          │ │
│  │         └────────────┬────┴─────────────────┘                          │ │
│  │                      ▼                                                 │ │
│  │         ┌──────────────────────────┐                                   │ │
│  │         │    CUDA Streams (H11)    │                                   │ │
│  │         │    Parallel Execution    │                                   │ │
│  │         └────────────┬─────────────┘                                   │ │
│  └──────────────────────┼─────────────────────────────────────────────────┘ │
│                         │                                                   │
│  ┌──────────────────────┼─────────────────────────────────────────────────┐ │
│  │                      ▼       BACKEND (Optimization)                    │ │
│  │         ┌──────────────────────────┐                                   │ │
│  │         │    LoopClosure (H09)     │                                   │ │
│  │         │    - Descriptor match    │                                   │ │
│  │         │    - Geometric verify    │                                   │ │
│  │         │    - GPU matching (H14)  │                                   │ │
│  │         └────────────┬─────────────┘                                   │ │
│  │                      │                                                 │ │
│  │                      ▼                                                 │ │
│  │         ┌──────────────────────────┐                                   │ │
│  │         │    Mapper (H10)          │                                   │ │
│  │         │    - g2o pose graph      │                                   │ │
│  │         │    - SE3 vertices        │                                   │ │
│  │         │    - Levenberg-Marquardt │                                   │ │
│  │         └──────────────────────────┘                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        DATA LAYER                                      │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │ │
│  │  │ EuRoCReader  │  │   KeyFrame   │  │   MapPoint   │                  │ │
│  │  │   (H07)      │  │   Database   │  │   Cloud      │                  │ │
│  │  │              │  │              │  │              │                  │ │
│  │  │ - Image sync │  │ - Poses      │  │ - 3D points  │                  │ │
│  │  │ - IMU interp │  │ - Covisib.   │  │ - PLY export │                  │ │
│  │  │ - GT poses   │  │ - Observ.    │  │              │                  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Flujo de Datos entre Clases

```
+------------------+
|     main.cpp     |
+--------+---------+
         |
         | crea
         v
+------------------+     +------------------+     +------------------+
|  cv::cuda::ORB   |     |  TRTInference    |     |   IMU (H08)      |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         | usa                    | usa                    | usa
         v                        v                        v
+------------------+     +------------------+     +------------------+
|      Frame       |     |    Detection     |     |  ImuMeasurement  |
|                  |     |                  |     |                  |
| - image          |     | - box            |     | - timestamp      |
| - keypoints[]    |     | - confidence     |     | - accel (3D)     |
| - descriptors    |     | - class_id       |     | - gyro (3D)      |
| - gpu_descriptors|     +------------------+     +------------------+
+--------+---------+
         |
         | almacena
         v
+------------------+
|   prev_frame     |
| (frame anterior) |
+------------------+
         |
         | opcional: promover a
         v
+------------------+     +------------------+
|    KeyFrame      |────>|   LoopClosure    |
|    (H09-H10)     |     |    (H09, H14)    |
|                  |     |                  |
| - pose SE3       |     | - detectar loops |
| - covisibility   |     | - GPU matching   |
| - observations   |     | - geom. verify   |
+--------+---------+     +--------+---------+
         |                        |
         | conecta                | detecta loop
         v                        v
+------------------+     +------------------+
|  Pose Graph      |<────| LoopCandidate    |
|    (g2o)         |     |                  |
|                  |     | - query_id       |
| - VertexSE3      |     | - match_id       |
| - EdgeSE3        |     | - relative_pose  |
| - optimize()     |     | - matches        |
+------------------+     +------------------+
```

---

## Memoria GPU

```
VRAM (~500MB total)
+--------------------------------------------------+
|                                                  |
|  ORB Buffers (~100MB)                            |
|  +--------------------------------------------+  |
|  | GpuMat gpu_img (grayscale)                 |  |
|  | GpuMat gpu_keypoints                       |  |
|  | GpuMat gpu_descriptors                     |  |
|  +--------------------------------------------+  |
|                                                  |
|  TensorRT Buffers (~400MB)                       |
|  +--------------------------------------------+  |
|  | buffers_[0] - Input  (3x640x640 float)     |  |
|  | buffers_[1] - Output (1x300x6 float)       |  |
|  | TRT Engine weights                         |  |
|  +--------------------------------------------+  |
|                                                  |
|  Loop Closure (H14) (~50MB)                      |
|  +--------------------------------------------+  |
|  | gpu_descriptors_ database                  |  |
|  | (all keyframe descriptors on GPU)          |  |
|  +--------------------------------------------+  |
|                                                  |
+--------------------------------------------------+
```

---

## Referencia Rapida de Archivos

```
src/
  main.cpp          <- Pipeline principal (H11 CUDA streams)
  Frame.cpp         <- ORB GPU wrapper (H02, H05)
  TRTInference.cpp  <- YOLO TensorRT (H06)
  IMU.cpp           <- EKF sensor fusion (H08)
  LoopClosure.cpp   <- Loop detection + pose graph (H09, H10, H14)
  EuRoCReader.cpp   <- Dataset reader (H07)
  Mapper.cpp        <- 3D mapping (H10)

include/
  Frame.hpp
  TRTInference.hpp
  IMU.hpp
  LoopClosure.hpp
  EuRoCReader.hpp
  Mapper.hpp

Lineas clave en main.cpp:
  29-40   DYNAMIC_CLASSES (objetos a filtrar)
  43-50   isInDynamicObject() (filtrado dinamico)
  100-104 Crear CUDA streams
  129-136 Lanzar ORB y YOLO en paralelo
  139-140 Sincronizar streams
  162-172 Matching con filtrado dinamico
  176-198 Pose estimation
```

---

## Milestones Completados

| Milestone | Descripcion | Archivo Principal |
|-----------|-------------|-------------------|
| H01 | Setup del proyecto | CMakeLists.txt |
| H02 | Feature extraction | Frame.cpp |
| H03 | Feature matching | main.cpp |
| H04 | Pose estimation | main.cpp |
| H05 | OpenCV CUDA | Frame.cpp |
| H06 | TensorRT YOLO | TRTInference.cpp |
| H07 | EuRoC Dataset | EuRoCReader.cpp |
| H08 | Sensor Fusion (IMU) | IMU.cpp |
| H09 | Loop Closure | LoopClosure.cpp |
| H10 | Pose Graph (g2o) | LoopClosure.cpp, Mapper.cpp |
| H11 | CUDA Streams | main.cpp |
| H12 | Clean Architecture | (Design doc) |
| H13 | Multithreading | (Thread patterns) |
| H14 | GPU Loop Closure | LoopClosure.cpp |
