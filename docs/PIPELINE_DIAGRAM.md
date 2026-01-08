# Diagrama del Pipeline aria-slam

## Vista General (30,000 pies)

```
+-------+     +-------+     +-------+     +-------+
| Video | --> | Frame | --> | Match | --> | Pose  | --> Trayectoria
+-------+     +-------+     +-------+     +-------+
                 |
                 +--> YOLO --> Detecciones
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
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+------------------+                                   +------------------+
|  CUDA STREAM 1   |                                   |  CUDA STREAM 2   |
|    (stream_orb)  |                                   |   (stream_yolo)  |
+------------------+                                   +------------------+
         |                                                       |
         v                                                       v
+------------------+                                   +------------------+
|  Frame::Frame()  |                                   | yolo->detectAsync|
+------------------+                                   +------------------+
         |                                                       |
         v                                                       v
+------------------+                                   +------------------+
| 1. BGR -> Gray   |                                   | 1. Resize 640x640|
| 2. CPU -> GPU    |                                   | 2. BGR -> RGB    |
| 3. ORB detect    |                                   | 3. Normalize 0-1 |
|    (async)       |                                   | 4. HWC -> CHW    |
+------------------+                                   | 5. CPU -> GPU    |
         |                                             | 6. TRT inference |
         |                                             |    (async)       |
         |                                             | 7. GPU -> CPU    |
         |                                             +------------------+
         |                                                       |
         +---------------------------+---------------------------+
                                     |
                                     v
                        +------------------------+
                        | cudaStreamSynchronize  |
                        |   (esperar ambos)      |
                        +------------------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+------------------+                                   +------------------+
| downloadResults()|                                   | getDetections()  |
+------------------+                                   +------------------+
         |                                                       |
         v                                                       v
+------------------+                                   +------------------+
| - keypoints[]    |                                   | - Detection[]    |
| - descriptors    |                                   |   - box (x,y,w,h)|
| - gpu_descriptors|                                   |   - confidence   |
+------------------+                                   |   - class_id     |
                                                       +------------------+


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
|    Trajectory    |     |     Display      |     |       FPS        |
|                  |     |                  |     |                  |
|  o               |     | +--[person]--+   |     |  FPS: 80         |
|   o              |     | |            |   |     |  Matches: 245    |
|    o             |     | +------------+   |     |  Objects: 3      |
|     o  <- pos    |     |    * * *         |     |                  |
|                  |     |   *     *        |     |                  |
+------------------+     +------------------+     +------------------+

```

---

## Timeline de Ejecucion (1 frame)

```
Tiempo (ms)   0    2    4    6    8   10   12   14
              |----|----|----|----|----|----|----|

CPU:          [CAP]                    [MATCH][POSE][VIS]
               2ms                       2ms   1ms   1ms

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

## Flujo de Datos entre Clases

```
+------------------+
|     main.cpp     |
+--------+---------+
         |
         | crea
         v
+------------------+     +------------------+
|  cv::cuda::ORB   |     |  TRTInference    |
+--------+---------+     +--------+---------+
         |                        |
         | usa                    | usa
         v                        v
+------------------+     +------------------+
|      Frame       |     |    Detection     |
|                  |     |                  |
| - image          |     | - box            |
| - keypoints[]    |     | - confidence     |
| - descriptors    |     | - class_id       |
| - gpu_descriptors|     +------------------+
+------------------+

         |
         | almacena
         v
+------------------+
|   prev_frame     |
| (frame anterior) |
+------------------+
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
|  | buffers_[1] - Output (84x8400 float)       |  |
|  | TRT Engine weights                         |  |
|  +--------------------------------------------+  |
|                                                  |
+--------------------------------------------------+
```

---

## Referencia Rapida de Archivos

```
src/
  main.cpp          <- Pipeline principal (este archivo es el CORE)
  Frame.cpp         <- Wrapper de features (ORB)
  TRTInference.cpp  <- Wrapper de YOLO (TensorRT)

include/
  Frame.hpp         <- Definicion de Frame
  TRTInference.hpp  <- Definicion de TRTInference

Lineas clave en main.cpp:
  74-77   Crear CUDA streams
  103-110 Lanzar ORB y YOLO en paralelo
  112-114 Sincronizar streams
  127-139 Matching con ratio test
  142-164 Pose estimation
```

---

**Siguiente:** Leer main.cpp linea por linea.
