# main.cpp - Explicacion Linea por Linea

## Estructura General

```
main.cpp tiene ~215 lineas:
- Lineas 1-41:   Includes y constantes
- Lineas 43-78:  Inicializacion
- Lineas 96-208: Loop principal (1 iteracion = 1 frame)
- Lineas 210-215: Cleanup
```

---

## Seccion 1: Includes y Constantes (1-41)

```cpp
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>  // ORB GPU
#include <cuda_runtime_api.h>           // CUDA streams
#include <chrono>                       // Medir tiempo
#include "Frame.hpp"                    // Nuestra clase Frame
#include "TRTInference.hpp"             // Nuestra clase YOLO
```

```cpp
// 80 clases de COCO (dataset donde se entreno YOLO)
const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", ...
};
```

**Por que importa:** Necesitamos OpenCV CUDA para ORB en GPU, y cuda_runtime para crear streams.

---

## Seccion 2: Inicializacion (43-78)

### Verificar CUDA (47-52)
```cpp
int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
if (cuda_devices == 0) {
    std::cerr << "Error: No CUDA devices found!" << std::endl;
    return -1;
}
```
**Que hace:** Verifica que hay GPU NVIDIA disponible.

### Abrir Video (54-58)
```cpp
cv::VideoCapture cap("../test.mp4");
if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video." << std::endl;
    return -1;
}
```
**Que hace:** Abre el video de prueba. `cap >> frame` leera frames.

### Crear Detectores (60-72)
```cpp
// ORB en GPU
cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create();

// Matcher en GPU (distancia Hamming para descriptores binarios)
cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
    cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

// YOLO con TensorRT
std::unique_ptr<TRTInference> yolo;
try {
    yolo = std::make_unique<TRTInference>("../models/yolov12s.engine");
} catch (const std::exception& e) {
    std::cerr << "Warning: YOLO disabled - " << e.what() << std::endl;
}
```
**Que hace:**
- `ORB::create()` - Detector de features en GPU
- `BFMatcher(NORM_HAMMING)` - Matcher para descriptores binarios (ORB usa bits)
- `TRTInference` - Carga el motor YOLO optimizado

### Crear CUDA Streams (74-78)
```cpp
cudaStream_t stream_orb, stream_yolo;
cudaStreamCreate(&stream_orb);
cudaStreamCreate(&stream_yolo);
```
**Que hace:** Crea dos "colas" independientes para la GPU.
- `stream_orb` - Cola para operaciones ORB
- `stream_yolo` - Cola para operaciones YOLO

**Por que importa:** Permite que ORB y YOLO ejecuten EN PARALELO.

### Variables de Estado (80-94)
```cpp
// Frame anterior (para matching)
std::unique_ptr<Frame> prev_frame;

// Matriz de camara (parametros intrinsicos)
double fx = 700, fy = 700;        // Distancia focal
double cx = 640 / 2.0;            // Centro X
double cy = 360 / 2.0;            // Centro Y
cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

// Pose acumulada (donde estamos en el mundo)
cv::Mat position = cv::Mat::zeros(3, 1, CV_64F);  // (x, y, z)
cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);    // Identidad inicial

// Canvas para dibujar trayectoria
cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);
```

**Matriz K:**
```
K = | fx  0  cx |   fx,fy = distancia focal (zoom)
    |  0 fy  cy |   cx,cy = punto principal (centro optico)
    |  0  0   1 |
```

---

## Seccion 3: Loop Principal (96-208)

### Inicio del Loop (96-101)
```cpp
while (true) {
    auto t1 = std::chrono::high_resolution_clock::now();  // Timer inicio

    cv::Mat frame;
    cap >> frame;                    // Leer siguiente frame
    if (frame.empty()) break;        // Fin del video
```

### LANZAR ORB Y YOLO EN PARALELO (103-110)
```cpp
// Stream 1: ORB (async - no bloquea)
Frame current_frame(frame, orb, stream_orb);

// Stream 2: YOLO (async - no bloquea)
if (yolo) {
    yolo->detectAsync(frame, stream_yolo);
}
```
**Que hace:**
1. `Frame(frame, orb, stream_orb)` - Lanza ORB en stream 1
2. `detectAsync(frame, stream_yolo)` - Lanza YOLO en stream 2
3. AMBOS ejecutan en la GPU SIMULTANEAMENTE

**IMPORTANTE:** Estas llamadas RETORNAN INMEDIATAMENTE. El trabajo GPU esta en progreso.

### SINCRONIZAR STREAMS (112-114)
```cpp
cudaStreamSynchronize(stream_orb);
cudaStreamSynchronize(stream_yolo);
```
**Que hace:** Espera a que AMBAS operaciones terminen.

**Timeline:**
```
Linea 105: lanza ORB       |------ ORB GPU ------|
Linea 109: lanza YOLO      |-- YOLO GPU --|
Linea 113: sync ORB                              ^ espera aqui
Linea 114: sync YOLO       (ya termino, no espera)
```

### OBTENER RESULTADOS (116-123)
```cpp
// Bajar resultados de ORB desde GPU
current_frame.downloadResults();

// Obtener detecciones YOLO (ya estan en CPU despues del sync)
std::vector<Detection> detections;
if (yolo) {
    detections = yolo->getDetections(0.5f, 0.45f);
    //                              conf   nms
}
```
**Que hace:**
- `downloadResults()` - Copia keypoints y descriptors de GPU a CPU
- `getDetections(0.5, 0.45)` - Postprocesa: filtra por confianza 50%, NMS 45%

### MATCHING (125-139)
```cpp
std::vector<cv::DMatch> good_matches;

if (prev_frame &&
    !prev_frame->gpu_descriptors.empty() &&
    !current_frame.gpu_descriptors.empty()) {

    // kNN match (k=2 para ratio test)
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(prev_frame->gpu_descriptors,
                      current_frame.gpu_descriptors,
                      knn_matches, 2);

    // Lowe's ratio test
    for (auto& knn : knn_matches) {
        if (knn.size() >= 2 && knn[0].distance < 0.75 * knn[1].distance) {
            good_matches.push_back(knn[0]);
        }
    }
}
```
**Que hace:**
1. `knnMatch(..., 2)` - Para cada descriptor en frame N-1, encuentra los 2 mas cercanos en frame N
2. Ratio test: Si el mejor es MUCHO mejor que el segundo (< 75%), es confiable

**Por que ratio test:**
```
Caso bueno: distances = [10, 100]  -> 10/100 = 0.1 < 0.75 OK
Caso malo:  distances = [50, 60]   -> 50/60 = 0.83 > 0.75 RECHAZADO
```

### POSE ESTIMATION (141-164)
```cpp
if (good_matches.size() >= 8) {
    // Extraer puntos correspondientes
    std::vector<cv::Point2f> pts1, pts2;
    for (auto& m : good_matches) {
        pts1.push_back(prev_frame->keypoints[m.queryIdx].pt);
        pts2.push_back(current_frame.keypoints[m.trainIdx].pt);
    }

    // Essential Matrix con RANSAC
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC);

    // Descomponer en R y t
    cv::Mat R, t;
    cv::recoverPose(E, pts1, pts2, K, R, t);

    // Acumular pose
    position = position + rotation * t;
    rotation = R * rotation;

    // Dibujar en trayectoria
    int x = (int)(position.at<double>(0) * 100) + 300;
    int y = (int)(position.at<double>(2) * 100) + 300;
    cv::circle(trajectory, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
}
```

**Por que >= 8 matches:**
- Essential Matrix tiene 5 grados de libertad
- Necesita minimo 5 puntos (pero 8 es mas robusto con RANSAC)

**findEssentialMat:**
- Encuentra matriz E tal que: `pts2^T * E * pts1 = 0`
- RANSAC ignora outliers (matches incorrectos)

**recoverPose:**
- Descompone E en 4 soluciones posibles
- Elige la que tiene puntos DELANTE de la camara

**Acumular pose:**
```cpp
position = position + rotation * t;  // Mover en direccion actual
rotation = R * rotation;              // Actualizar orientacion
```

### VISUALIZACION (166-203)
```cpp
// Mostrar trayectoria
cv::imshow("Trajectory", trajectory);

// Dibujar matches o keypoints
cv::Mat display;
if (prev_frame && !good_matches.empty()) {
    cv::drawMatches(...);  // Dibuja lineas entre matches
} else {
    cv::drawKeypoints(...);  // Solo puntos verdes
}

// Dibujar detecciones YOLO
for (const auto& det : detections) {
    cv::rectangle(display, det.box, cv::Scalar(0, 0, 255), 2);
    std::string label = COCO_CLASSES[det.class_id] + " " +
                       std::to_string((int)(det.confidence * 100)) + "%";
    cv::putText(display, label, ...);
}

// Guardar frame actual para siguiente iteracion
prev_frame = std::make_unique<Frame>(current_frame);

// Mostrar FPS
auto t2 = std::chrono::high_resolution_clock::now();
double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
cv::putText(display, "FPS: " + std::to_string((int)(1000.0 / ms)), ...);
```

### EXIT Y CLEANUP (206-215)
```cpp
    char key = cv::waitKey(1);
    if (key == 'q') break;
}

// Liberar CUDA streams
cudaStreamDestroy(stream_orb);
cudaStreamDestroy(stream_yolo);

return 0;
```

---

## Resumen del Loop

```
1. Leer frame              cap >> frame
2. Lanzar ORB (async)      Frame(frame, orb, stream_orb)
3. Lanzar YOLO (async)     yolo->detectAsync(frame, stream_yolo)
4. Sincronizar             cudaStreamSynchronize(...)
5. Obtener resultados      downloadResults(), getDetections()
6. Match con anterior      matcher->knnMatch() + ratio test
7. Estimar pose            findEssentialMat + recoverPose
8. Acumular posicion       position += rotation * t
9. Visualizar              imshow()
10. Guardar frame          prev_frame = current_frame
```

---

## Lineas Clave para Memorizar

| Linea | Codigo | Proposito |
|-------|--------|-----------|
| 74-77 | `cudaStreamCreate` | Crear streams paralelos |
| 105 | `Frame(frame, orb, stream_orb)` | ORB async |
| 109 | `yolo->detectAsync(frame, stream_yolo)` | YOLO async |
| 113-114 | `cudaStreamSynchronize` | Esperar GPU |
| 132 | `matcher->knnMatch` | Matching GPU |
| 135 | `knn[0].distance < 0.75 * knn[1].distance` | Ratio test |
| 150 | `findEssentialMat` | Geometria epipolar |
| 154 | `recoverPose` | Extraer R, t |
| 157-158 | `position += rotation * t` | Acumular pose |

---

**Siguiente:** Compilar y ejecutar para ver el resultado visual.
