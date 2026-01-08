# Guia de Estudio: aria-slam

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

### CUDA Streams

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

## 4. TensorRT

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
TRTInference yolo("yolov12s.engine");

// Inferencia
auto detections = yolo.detect(frame, 0.5f, 0.45f);
//                            imagen  conf   nms
```

---

## 5. Pipeline de aria-slam

```
                    +------------------+
                    |   Video Input    |
                    +--------+---------+
                             |
                             v
+------------------+    +----+----+    +------------------+
|   CUDA Stream 1  |    |  Frame  |    |   CUDA Stream 2  |
|                  |    +---------+    |                  |
|  +------------+  |         |         |  +------------+  |
|  |  ORB GPU   |  | <-------+-------> |  | YOLO GPU   |  |
|  | (features) |  |                   |  | (objects)  |  |
|  +-----+------+  |                   |  +-----+------+  |
|        |         |                   |        |         |
+--------+---------+                   +--------+---------+
         |                                      |
         v                                      v
   +-----------+                         +-----------+
   | keypoints |                         | detections|
   |descriptors|                         |  (boxes)  |
   +-----+-----+                         +-----------+
         |
         v
+------------------+
|     Matching     |
| (frame N-1 vs N) |
+--------+---------+
         |
         v
+------------------+
|  Essential Mat   |
|   + RANSAC       |
+--------+---------+
         |
         v
+------------------+
|   Recover Pose   |
|     (R, t)       |
+--------+---------+
         |
         v
+------------------+
| Accumulate Pose  |
| position += R*t  |
+--------+---------+
         |
         v
+------------------+
|   Visualization  |
| trajectory + det |
+------------------+
```

---

## 6. Clases Principales

### Frame
Representa un frame procesado con sus features.

```cpp
class Frame {
    cv::Mat image;              // Imagen original
    vector<KeyPoint> keypoints; // Puntos detectados
    cv::Mat descriptors;        // Descriptores (para matching)
    cv::cuda::GpuMat gpu_descriptors; // En GPU (para matching rapido)
};
```

**Constructor GPU:**
```cpp
Frame(image, orb_gpu, stream);
// 1. Convierte a grayscale
// 2. Sube imagen a GPU
// 3. Detecta features (async si hay stream)
// 4. Baja resultados a CPU (lazy si hay stream)
```

### TRTInference
Wrapper para YOLO con TensorRT.

```cpp
class TRTInference {
    // Motor TensorRT
    nvinfer1::IExecutionContext* context_;

    // Buffers GPU
    void* buffers_[2];  // [0]=input, [1]=output

    // Metodos
    void detectAsync(image, stream);  // Lanza inferencia
    vector<Detection> getDetections(); // Obtiene resultados
};
```

---

## 7. Flujo de Memoria

```
CPU (Host)                          GPU (Device)
==========                          ============

cv::Mat frame
    |
    | cudaMemcpyHostToDevice
    v
                        ------>     GpuMat gpu_img
                                        |
                                        v
                                    [ORB kernel]
                                        |
                                        v
                                    GpuMat gpu_keypoints
                                    GpuMat gpu_descriptors
    |                                   |
    | cudaMemcpyDeviceToHost    <------
    v
vector<KeyPoint> keypoints
cv::Mat descriptors
```

**H2D** = Host to Device (CPU -> GPU)
**D2H** = Device to Host (GPU -> CPU)

---

## 8. Pose Accumulation

Cada frame da movimiento relativo. Acumulamos para posicion global.

```cpp
// Movimiento relativo (frame a frame)
cv::Mat R, t;  // Rotacion y traslacion

// Acumular en coordenadas mundo
position = position + rotation * t;  // Trasladar
rotation = R * rotation;              // Rotar
```

```
Frame 0 -> Frame 1: move (1,0,0)
Frame 1 -> Frame 2: move (0,1,0)
Frame 2 -> Frame 3: move (1,0,0)

Trayectoria acumulada:
(0,0,0) -> (1,0,0) -> (1,1,0) -> (2,1,0)
```

---

## 9. Glosario Rapido

| Termino | Significado |
|---------|-------------|
| SLAM | Localizacion y Mapeo Simultaneo |
| VO | Visual Odometry - movimiento desde imagenes |
| ORB | Detector de features (FAST + BRIEF) |
| Feature | Punto interesante en imagen |
| Descriptor | Vector que describe un feature |
| Matching | Encontrar mismo punto en 2 frames |
| Essential Matrix | Relacion geometrica entre vistas |
| RANSAC | Algoritmo para ignorar outliers |
| CUDA Stream | Cola de operaciones GPU paralelas |
| TensorRT | Optimizador de inferencia NVIDIA |
| H2D | Host to Device (CPU -> GPU) |
| D2H | Device to Host (GPU -> CPU) |

---

## 10. Preguntas de Auto-Evaluacion

1. **Por que usamos GPU para ORB?**
   > Porque ORB procesa miles de puntos independientes - ideal para paralelismo GPU.

2. **Que hace findEssentialMat?**
   > Calcula la matriz que relaciona puntos correspondientes entre dos vistas.

3. **Por que CUDA streams mejoran el rendimiento?**
   > Permiten que ORB y YOLO ejecuten en paralelo en la GPU.

4. **Que es el ratio test de Lowe?**
   > Filtro para matches: si el mejor es mucho mejor que el segundo, es confiable.

5. **Como se acumula la pose?**
   > position += rotation * t; rotation = R * rotation;

---

**Siguiente paso:** Ver el diagrama del pipeline completo, luego leer main.cpp linea por linea.
