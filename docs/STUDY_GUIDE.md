# Guia de Estudio: aria-slam

**Actualizado:** H01-H14 completados

---

## 1. Que es SLAM?

**SLAM** = Simultaneous Localization and Mapping

Responde dos preguntas a la vez:
- **Donde estoy?** (Localization)
- **Como es el entorno?** (Mapping)

```
Problema del huevo y la gallina:
- Para saber donde estoy, necesito un mapa
- Para hacer un mapa, necesito saber donde estoy
- SLAM resuelve ambos simultaneamente
```

### Aplicaciones
- Robots autonomos
- Drones
- Coches autonomos
- AR/VR (realidad aumentada)
- **Tu proyecto:** Navegacion para personas con discapacidad visual

---

## 2. Visual Odometry (VO)

**Odometria Visual** = Estimar movimiento de la camara usando solo imagenes.

### Pipeline basico:

```
Frame N-1          Frame N
    |                  |
    v                  v
[Features]  -----> [Features]
    |                  |
    +---> [Match] <----+
             |
             v
    [Calcular Movimiento]
             |
             v
      Pose (R, t)
```

### Conceptos clave:

#### Features (Caracteristicas)
Puntos "interesantes" en la imagen que podemos detectar y seguir.

```
Imagen original:        Features detectados:
+----------------+      +----------------+
|    ___         |      |    *           |
|   /   \   __   |  =>  |   * *     *    |
|  |     | |  |  |      |  *     *  * *  |
|   \___/  |__|  |      |   * *    **    |
+----------------+      +----------------+
```

**ORB** (Oriented FAST and Rotated BRIEF):
- FAST: Detecta esquinas rapido
- BRIEF: Descriptor binario (256 bits)
- Oriented: Invariante a rotacion

#### Matching (Emparejamiento)
Encontrar el mismo punto en dos frames diferentes.

```
Frame N-1:  *A    *B    *C
             \    |    /
              \   |   /
               \  |  /
Frame N:        *A' *B' *C'
```

**Lowe's Ratio Test:** Si el mejor match es mucho mejor que el segundo, es confiable.
```cpp
if (match[0].distance < 0.75 * match[1].distance) {
    // Match confiable
}
```

#### Pose Estimation (Estimacion de Pose)
Calcular rotacion (R) y traslacion (t) entre frames.

```
Essential Matrix (E):
- Codifica la relacion geometrica entre dos vistas
- Se calcula con minimo 8 puntos (RANSAC para outliers)
- Se descompone en R y t
```

```cpp
cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC);
cv::recoverPose(E, pts1, pts2, K, R, t);
```

---

## 3. GPU y CUDA

### Por que GPU?

```
CPU: 8-16 cores potentes (tareas secuenciales)
GPU: 1000+ cores simples (tareas paralelas)

 ORB en 1000 keypoints:
- CPU: procesa 1 por 1 = lento
- GPU: procesa 1000 a la vez = rapido
```

### CUDA Basico

```cpp
// CPU (Host)              // GPU (Device)
float* h_data;             float* d_data;
h_data = malloc(...);      cudaMalloc(&d_data, ...);

// Copiar CPU -> GPU
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Ejecutar en GPU
kernel<<<blocks, threads>>>(d_data);

// Copiar GPU -> CPU
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
```

### CUDA Streams (H11)

Permiten ejecutar operaciones GPU en paralelo.

```
Sin streams (secuencial):
|--ORB 10ms--|--YOLO 5ms--| = 15ms

Con streams (paralelo):
Stream 1: |--ORB 10ms--------|
Stream 2: |--YOLO 5ms--|
                              = 10ms (maximo de ambos)
```

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Lanzar en paralelo (no bloquea)
operacion1_async(stream1);
operacion2_async(stream2);

// Esperar ambos
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

---

## 4. TensorRT (H06)

### Que es?
Optimizador de NVIDIA para inferencia de redes neuronales.

```
PyTorch Model (.pt)
       |
       v
    ONNX (.onnx)
       |
       v
TensorRT Engine (.engine)  <-- Optimizado para tu GPU especifica
       |
       v
   Inferencia rapida
```

### Optimizaciones que hace:
- **Fusion de capas:** Conv + BatchNorm + ReLU = 1 operacion
- **Precision reducida:** FP32 -> FP16 (2x mas rapido, misma precision)
- **Kernel auto-tuning:** Elige el mejor kernel para tu GPU

### En aria-slam:
```cpp
// Cargar engine
TRTInference yolo("yolo26s.engine");

// Inferencia async
yolo.detectAsync(frame, stream);

// Obtener resultados
auto detections = yolo.getDetections(0.5f, 0.45f);
//                                   conf   nms
```

---

## 5. Filtrado de Objetos Dinamicos (H06)

### El Problema
Objetos en movimiento (personas, coches) generan matches incorrectos.

```
Frame N-1:  Persona caminando -->  Frame N: Persona se movio
     *A                                 *A'  (match incorrecto!)
```

El punto A esta en la persona, no en el fondo estatico. El match A-A' no refleja movimiento de camara.

### La Solucion
YOLO detecta objetos dinamicos, filtramos keypoints dentro de sus bounding boxes.

```cpp
const std::set<int> DYNAMIC_CLASSES = {
    0,   // person
    2,   // car
    3,   // motorcycle
    5,   // bus
    7,   // truck
    // ... otros objetos moviles
};

bool isInDynamicObject(const cv::Point2f& pt,
                       const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        if (DYNAMIC_CLASSES.count(det.class_id) &&
            det.box.contains(pt)) {
            return true;  // Keypoint esta en objeto dinamico
        }
    }
    return false;
}
```

### Pipeline con Filtrado

```
1. ORB detecta keypoints
2. YOLO detecta objetos dinamicos
3. kNN matching entre frames
4. Ratio test filtra matches ambiguos
5. isInDynamicObject filtra keypoints en personas/coches
6. Solo matches en objetos ESTATICOS se usan para pose
```

---

## 6. Sensor Fusion con IMU (H08)

### Por que IMU?
- Vision falla con motion blur, poca luz
- IMU es rapido (100-1000 Hz) vs camara (30 Hz)
- Combinados = robusto

### Estado del EKF (Extended Kalman Filter)

```cpp
// Estado de 15 dimensiones
Eigen::Matrix<double, 15, 1> state_;
// [0-2]:  position     (x, y, z)
// [3-5]:  velocity     (vx, vy, vz)
// [6-9]:  quaternion   (qw, qx, qy, qz)
// [10-12]: accel_bias  (bax, bay, baz)
// [13-14]: gyro_bias   (bgx, bgy, bgz)
```

### Ciclo Predict-Update

```
IMU (1000 Hz):          Vision (30 Hz):
     |                       |
     v                       v
  predict()               update()
     |                       |
     v                       v
[Estado predicho]  -->  [Estado corregido]
```

```cpp
// Predict: usa IMU para avanzar estado
void predict(const ImuMeasurement& imu) {
    // Integrar aceleracion y gyro
    // Propagar incertidumbre
}

// Update: usa vision para corregir
void updateVO(const Pose& vo_pose) {
    // Calcular innovacion (diferencia)
    // Kalman gain
    // Corregir estado
}
```

---

## 7. Loop Closure (H09, H10, H14)

### El Problema: Drift
Visual odometry acumula error con el tiempo.

```
Trayectoria real:     Trayectoria estimada (con drift):
    +----+                +----+
    |    |                |     \
    |    |                |      \
    +----+                +-------+
    (cerrado)             (no cierra!)
```

### La Solucion: Loop Closure
Detectar cuando volvemos a un lugar visitado y corregir.

```
1. Crear keyframe cada N frames
2. Comparar descriptores con keyframes anteriores
3. Si match score > threshold -> loop detectado!
4. Optimizar trayectoria para cerrar el loop
```

### Bag of Words (BoW)
Representacion compacta de una imagen para busqueda rapida.

```cpp
// Cada keyframe tiene un "vocabulario" de features
KeyFrame {
    vector<KeyPoint> keypoints;
    Mat descriptors;
    float bow_score;  // Para comparacion rapida
};
```

### Loop Detection en GPU (H14)

```cpp
class LoopClosure {
    cv::cuda::GpuMat gpu_database_;  // Descriptores en GPU

    bool detect(const KeyFrame& query) {
        // Match query vs database en GPU
        // Mucho mas rapido que CPU
    }
};
```

---

## 8. Pose Graph Optimization (H10)

### Que es?
Optimizacion global de todas las poses usando restricciones.

### Grafo de Poses

```
Nodos: Poses de keyframes
       KF0 ---- KF1 ---- KF2 ---- KF3
        |                          |
        +-------- loop edge -------+

Edges:
  - Odometry edges (entre frames consecutivos)
  - Loop edges (cuando detectamos loop closure)
```

### g2o (General Graph Optimization)

```cpp
g2o::SparseOptimizer optimizer;

// Agregar vertices (poses)
for (auto& kf : keyframes) {
    auto* v = new g2o::VertexSE3();
    v->setEstimate(kf.pose);
    optimizer.addVertex(v);
}

// Agregar edges (restricciones)
// Odometry edge: KF[i] -> KF[i+1]
auto* e = new g2o::EdgeSE3();
e->setMeasurement(relative_pose);
optimizer.addEdge(e);

// Optimizar
optimizer.optimize(10);  // 10 iteraciones
```

### Resultado
Todas las poses se ajustan para satisfacer todas las restricciones.

```
Antes de optimizacion:     Despues:
    +----+                 +----+
    |     \                |    |
    |      \    -->        |    |
    +-------+              +----+
   (loop no cierra)       (loop cerrado!)
```

---

## 9. Pipeline Completo de aria-slam

```
                    +------------------+
                    |   Video/EuRoC    |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
     +--------+--------+           +--------+--------+
     |  CUDA Stream 1  |           |  CUDA Stream 2  |
     |    ORB GPU      |           |   YOLO GPU      |
     +--------+--------+           +--------+--------+
              |                             |
              v                             v
        [keypoints]                  [detections]
        [descriptors]                 (boxes)
              |                             |
              +----------+------------------+
                         |
                         v
              +----------+----------+
              | Dynamic Filtering   |
              | (remove pts on      |
              |  moving objects)    |
              +----------+----------+
                         |
                         v
              +----------+----------+
              |   GPU Matching      |
              |   + Ratio Test      |
              +----------+----------+
                         |
                         v
              +----------+----------+
              |  Essential Matrix   |
              |    + RANSAC         |
              +----------+----------+
                         |
         +---------------+---------------+
         |                               |
         v                               v
+--------+--------+             +--------+--------+
|   IMU Fusion    |             |  Loop Closure   |
|   (EKF update)  |             |  (detect loops) |
+--------+--------+             +--------+--------+
         |                               |
         v                               v
+--------+--------+             +--------+--------+
| Pose Accumulate |             | Pose Graph Opt  |
| pos += R * t    |             |    (g2o)        |
+--------+--------+             +--------+--------+
         |                               |
         +---------------+---------------+
                         |
                         v
              +----------+----------+
              |   Visualization     |
              | trajectory + boxes  |
              +---------------------+
```

---

## 10. Clases Principales

### Frame (H02, H05)
Representa un frame procesado con sus features.

```cpp
class Frame {
    cv::Mat image;              // Imagen original
    vector<KeyPoint> keypoints; // Puntos detectados
    cv::Mat descriptors;        // Descriptores CPU
    cv::cuda::GpuMat gpu_descriptors; // Descriptores GPU

    // Constructor con stream para async
    Frame(image, orb_gpu, stream);
};
```

### TRTInference (H06)
Wrapper para YOLO con TensorRT.

```cpp
class TRTInference {
    nvinfer1::IExecutionContext* context_;
    void* buffers_[2];  // [0]=input, [1]=output

    void detectAsync(image, stream);  // Lanza inferencia
    vector<Detection> getDetections(); // Obtiene resultados
};
```

### IMU (H08)
Extended Kalman Filter para fusion de sensores.

```cpp
class IMU {
    Eigen::Matrix<double, 15, 1> state_;   // Estado
    Eigen::Matrix<double, 15, 15> P_;      // Covarianza

    void predict(ImuMeasurement);  // Paso de prediccion
    void updateVO(Pose);           // Correccion con vision
};
```

### LoopClosure (H09, H10, H14)
Deteccion de loops y optimizacion de pose graph.

```cpp
class LoopClosure {
    vector<KeyFrame> keyframes_;
    cv::cuda::GpuMat gpu_database_;  // H14: GPU
    g2o::SparseOptimizer optimizer_;

    bool detect(KeyFrame& query);
    void optimizePoseGraph();
};
```

### EuRoCReader (H07)
Lector de dataset EuRoC para benchmarking.

```cpp
class EuRoCReader {
    vector<ImageEntry> images_;
    vector<ImuEntry> imu_data_;
    vector<GroundTruth> ground_truth_;

    bool getNextSynchronized(Mat&, vector<ImuMeasurement>&);
};
```

---

## 11. Flujo de Memoria GPU

```
CPU (Host)                          GPU (Device)
==========                          ============

cv::Mat frame
    |
    | cudaMemcpyHostToDevice
    v
                        ------>     GpuMat gpu_img
                                        |
                        +---------------+---------------+
                        |                               |
                        v                               v
                    [ORB kernel]                [YOLO kernel]
                        |                               |
                        v                               v
                GpuMat gpu_keypoints           float* yolo_output
                GpuMat gpu_descriptors
                        |                               |
                        +---------------+---------------+
                                        |
    |                                   |
    | cudaMemcpyDeviceToHost    <------
    v
vector<KeyPoint> keypoints
cv::Mat descriptors
vector<Detection> detections
```

**H2D** = Host to Device (CPU -> GPU)
**D2H** = Device to Host (GPU -> CPU)

---

## 12. Glosario Completo

| Termino | Significado |
|---------|-------------|
| SLAM | Localizacion y Mapeo Simultaneo |
| VO | Visual Odometry - movimiento desde imagenes |
| ORB | Detector de features (FAST + BRIEF) |
| Feature | Punto interesante en imagen |
| Descriptor | Vector que describe un feature (256 bits en ORB) |
| Matching | Encontrar mismo punto en 2 frames |
| Essential Matrix | Relacion geometrica entre vistas |
| RANSAC | Algoritmo para ignorar outliers |
| CUDA Stream | Cola de operaciones GPU paralelas |
| TensorRT | Optimizador de inferencia NVIDIA |
| H2D / D2H | Host to Device / Device to Host |
| IMU | Inertial Measurement Unit (acelerometro + giroscopio) |
| EKF | Extended Kalman Filter |
| Loop Closure | Detectar cuando volvemos a lugar visitado |
| Pose Graph | Grafo de poses con restricciones |
| g2o | Libreria para graph optimization |
| KeyFrame | Frame seleccionado para loop closure |
| Drift | Error acumulado en odometria |
| NMS | Non-Maximum Suppression (filtrar detecciones) |
| Bounding Box | Rectangulo que encierra objeto detectado |

---

## 13. Preguntas de Auto-Evaluacion

### Basico (H01-H04)

1. **Por que usamos GPU para ORB?**
   > Porque ORB procesa miles de puntos independientes - ideal para paralelismo GPU.

2. **Que hace findEssentialMat?**
   > Calcula la matriz que relaciona puntos correspondientes entre dos vistas.

3. **Que es el ratio test de Lowe?**
   > Filtro para matches: si el mejor es mucho mejor que el segundo, es confiable.

4. **Como se acumula la pose?**
   > position += rotation * t; rotation = R * rotation;

### Intermedio (H05-H08)

5. **Por que CUDA streams mejoran el rendimiento?**
   > Permiten que ORB y YOLO ejecuten en paralelo en la GPU.

6. **Que optimizaciones hace TensorRT?**
   > Fusion de capas, precision reducida (FP16), kernel auto-tuning.

7. **Por que filtramos keypoints en objetos dinamicos?**
   > Porque matches en objetos moviles no reflejan movimiento de camara.

8. **Que hace el IMU en SLAM?**
   > Predice pose entre frames de camara, mejora robustez ante motion blur.

### Avanzado (H09-H14)

9. **Que es loop closure y por que importa?**
   > Detectar cuando volvemos a un lugar visitado para corregir drift acumulado.

10. **Que es pose graph optimization?**
    > Ajustar todas las poses para satisfacer restricciones de odometria y loops.

11. **Por que usar GPU para loop closure database?**
    > Matching contra miles de keyframes es mas rapido en paralelo.

12. **Cual es la ventaja de EKF sobre simple averaging?**
    > EKF pondera segun incertidumbre - confía mas en sensores precisos.

---

## 14. Milestones del Proyecto

| # | Nombre | Tema Principal |
|---|--------|----------------|
| H01 | Setup | Configuracion inicial |
| H02 | Feature Extraction | ORB detector |
| H03 | Feature Matching | BF Matcher + ratio test |
| H04 | Pose Estimation | Essential Matrix + RANSAC |
| H05 | OpenCV CUDA | GPU acceleration |
| H06 | TensorRT YOLO | Object detection + filtering |
| H07 | EuRoC Dataset | Benchmark data |
| H08 | Sensor Fusion | IMU + EKF |
| H09 | Loop Closure | Place recognition |
| H10 | Pose Graph | g2o optimization |
| H11 | CUDA Streams | Parallel GPU ops |
| H12 | Clean Architecture | Design patterns |
| H13 | Multithreading | CPU parallelism |
| H14 | GPU Loop Closure | GPU database matching |

---

**Siguiente paso:** Leer los documentos AUDIT de cada milestone en `docs/milestones/` para detalles de implementacion.
