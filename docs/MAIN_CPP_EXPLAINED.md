# main.cpp - Explicacion Linea por Linea

**Actualizado:** H01-H14 (265 lineas)

---

## Estructura General

```
main.cpp (~265 lineas):
- Lineas 1-66:    Includes, constantes, DYNAMIC_CLASSES
- Lineas 68-121:  Inicializacion (CUDA, video, ORB, YOLO, streams)
- Lineas 122-257: Loop principal
- Lineas 259-264: Cleanup
```

---

## Seccion 1: Includes y Constantes (1-66)

### Includes (18-27)

```cpp
#include <iostream>
#include <memory>
#include <set>                          // Para DYNAMIC_CLASSES
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>   // ORB GPU
#include <cuda_runtime_api.h>           // CUDA streams
#include <chrono>                       // Medir tiempo
#include "Frame.hpp"
#include "TRTInference.hpp"
```

### Clases Dinamicas (29-40)

```cpp
const std::set<int> DYNAMIC_CLASSES = {
    0,   // person
    1,   // bicycle
    2,   // car
    3,   // motorcycle
    5,   // bus
    6,   // train
    7,   // truck
    14,  // bird
    15,  // cat
    16,  // dog
};
```

**Por que:** Objetos que se mueven no sirven para odometria visual. Si un keypoint esta sobre una persona caminando, ese match sera incorrecto.

### Funcion de Filtrado (43-50)

```cpp
bool isInDynamicObject(const cv::Point2f& pt, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        if (DYNAMIC_CLASSES.count(det.class_id) && det.box.contains(pt)) {
            return true;
        }
    }
    return false;
}
```

**Que hace:** Dado un punto 2D y las detecciones YOLO, retorna `true` si el punto esta dentro de un bounding box de objeto dinamico.

### COCO Classes (53-66)

```cpp
const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", ...  // 80 clases
};
```

**Para que:** Convertir class_id (numero) a nombre legible para visualizacion.

---

## Seccion 2: Inicializacion (68-121)

### Modo Headless (69-70)

```cpp
bool headless = (argc > 1 && std::string(argv[1]) == "--headless");
```

**Para que:** Ejecutar sin ventanas (CI/CD, benchmarks, SSH).

### Verificar CUDA (73-78)

```cpp
int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
if (cuda_devices == 0) {
    std::cerr << "Error: No CUDA devices found!" << std::endl;
    return -1;
}
```

### Crear Detectores (86-98)

```cpp
// ORB en GPU
cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create();

// Matcher en GPU (distancia Hamming para descriptores binarios)
cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
    cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

// YOLO con TensorRT
std::unique_ptr<TRTInference> yolo;
try {
    yolo = std::make_unique<TRTInference>("../models/yolo26s.engine");
} catch (const std::exception& e) {
    std::cerr << "Warning: YOLO disabled - " << e.what() << std::endl;
}
```

### Crear CUDA Streams (100-104)

```cpp
cudaStream_t stream_orb, stream_yolo;
cudaStreamCreate(&stream_orb);
cudaStreamCreate(&stream_yolo);
std::cout << "H11: CUDA streams created for parallel ORB + YOLO" << std::endl;
```

**Por que streams:**
- Sin streams: ORB espera, luego YOLO espera = 15ms
- Con streams: ORB y YOLO ejecutan en paralelo = 10ms (max de ambos)

### Matriz de Camara (109-113)

```cpp
double fx = 700, fy = 700;        // Distancia focal
double cx = 640 / 2.0;            // Centro X
double cy = 360 / 2.0;            // Centro Y
cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
```

**Matriz K:**
```
K = | fx  0  cx |   fx,fy = distancia focal (zoom)
    |  0 fy  cy |   cx,cy = punto principal (centro optico)
    |  0  0   1 |
```

### Estado Inicial (116-120)

```cpp
cv::Mat position = cv::Mat::zeros(3, 1, CV_64F);  // (x, y, z) = origen
cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);    // Identidad = sin rotacion
cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);  // Canvas negro
```

---

## Seccion 3: Loop Principal (122-257)

### Leer Frame (125-127)

```cpp
cv::Mat frame;
cap >> frame;
if (frame.empty()) break;
```

### H11: Lanzar ORB y YOLO en Paralelo (129-136)

```cpp
// Stream 1: ORB feature extraction (async)
Frame current_frame(frame, orb, stream_orb);

// Stream 2: YOLO object detection (async)
if (yolo) {
    yolo->detectAsync(frame, stream_yolo);
}
```

**IMPORTANTE:** Estas llamadas RETORNAN INMEDIATAMENTE. El trabajo GPU esta en progreso.

### H11: Sincronizar Streams (139-140)

```cpp
cudaStreamSynchronize(stream_orb);
cudaStreamSynchronize(stream_yolo);
```

**Timeline:**
```
Linea 131: lanza ORB       |------ ORB GPU ------|
Linea 135: lanza YOLO      |-- YOLO GPU --|
Linea 139: sync ORB                              ^ espera aqui
Linea 140: sync YOLO       (ya termino, no espera)
```

### Obtener Resultados (143-149)

```cpp
current_frame.downloadResults();

std::vector<Detection> detections;
if (yolo) {
    detections = yolo->getDetections(0.5f, 0.45f);
    //                              conf   nms
}
```

### Matching con Filtrado Dinamico (152-173)

```cpp
std::vector<cv::DMatch> good_matches;
int filtered_count = 0;

if (prev_frame && ...) {
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(prev_frame->gpu_descriptors,
                      current_frame.gpu_descriptors, knn_matches, 2);

    for (auto& knn : knn_matches) {
        if (knn.size() >= 2 && knn[0].distance < 0.75 * knn[1].distance) {
            // Filter out keypoints on dynamic objects
            cv::Point2f pt1 = prev_frame->keypoints[knn[0].queryIdx].pt;
            cv::Point2f pt2 = current_frame.keypoints[knn[0].trainIdx].pt;

            if (!isInDynamicObject(pt1, detections) &&
                !isInDynamicObject(pt2, detections)) {
                good_matches.push_back(knn[0]);
            } else {
                filtered_count++;
            }
        }
    }
}
```

**Filtrado dinamico:**
1. kNN match encuentra 2 mejores candidatos
2. Lowe's ratio test filtra matches ambiguos
3. `isInDynamicObject` filtra keypoints sobre personas/coches/etc
4. Solo los matches en objetos estaticos se usan para pose

### Pose Estimation (176-198)

```cpp
if (good_matches.size() >= 8) {
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
- Minimo 5 puntos, pero 8 es mas robusto con RANSAC

**Acumular pose:**
```cpp
position = position + rotation * t;  // Mover en direccion actual
rotation = R * rotation;              // Actualizar orientacion
```

### Visualizacion (207-244)

```cpp
if (!headless) {
    cv::imshow("Trajectory", trajectory);

    // Dibujar matches
    cv::Mat display;
    cv::drawMatches(prev_frame->image, prev_frame->keypoints,
                   current_frame.image, current_frame.keypoints,
                   good_matches, display);

    // Dibujar detecciones YOLO
    for (const auto& det : detections) {
        cv::rectangle(display, det.box, cv::Scalar(0, 0, 255), 2);
        std::string label = COCO_CLASSES[det.class_id] + " " +
                           std::to_string((int)(det.confidence * 100)) + "%";
        cv::putText(display, label, ...);
    }

    // Estadisticas
    cv::putText(display, "FPS: " + std::to_string((int)(1000.0 / ms)), ...);
    cv::putText(display, "Matches: " + std::to_string(good_matches.size()), ...);
    cv::putText(display, "Objects: " + std::to_string(detections.size()), ...);
    cv::putText(display, "Filtered: " + std::to_string(filtered_count), ...);
}
```

### Modo Headless (245-256)

```cpp
} else {
    static int frame_count = 0;
    frame_count++;
    if (frame_count % 50 == 0) {
        std::cout << "Frame " << frame_count
                  << " | FPS: " << (int)(1000.0 / ms)
                  << " | Matches: " << good_matches.size()
                  << " | Objects: " << detections.size()
                  << " | Filtered: " << filtered_count << std::endl;
    }
}
```

---

## Seccion 4: Cleanup (259-264)

```cpp
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
6. Match con filtrado      knnMatch + ratio test + isInDynamicObject
7. Estimar pose            findEssentialMat + recoverPose
8. Acumular posicion       position += rotation * t
9. Visualizar              imshow() o stdout
10. Guardar frame          prev_frame = current_frame
```

---

## Lineas Clave para Memorizar

| Linea | Codigo | Proposito |
|-------|--------|-----------|
| 29-40 | `DYNAMIC_CLASSES` | Objetos a filtrar |
| 43-50 | `isInDynamicObject` | Filtrar keypoints |
| 100-103 | `cudaStreamCreate` | Crear streams paralelos |
| 131 | `Frame(frame, orb, stream_orb)` | ORB async |
| 135 | `yolo->detectAsync(...)` | YOLO async |
| 139-140 | `cudaStreamSynchronize` | Esperar GPU |
| 159 | `matcher->knnMatch` | Matching GPU |
| 162 | `knn[0].distance < 0.75 * knn[1].distance` | Ratio test |
| 166-167 | `!isInDynamicObject(...)` | Filtrado dinamico |
| 184 | `findEssentialMat` | Geometria epipolar |
| 188 | `recoverPose` | Extraer R, t |
| 191-192 | `position += rotation * t` | Acumular pose |
