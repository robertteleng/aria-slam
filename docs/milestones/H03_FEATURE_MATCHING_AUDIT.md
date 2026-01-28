# Auditoría Técnica: H03 - Feature Matching (GPU BFMatcher)

**Proyecto:** aria-slam (C++)
**Milestone:** H03 - Matching de features con Lowe's ratio test
**Fecha:** 2025-01
**Autor:** Roberto (preparación entrevista técnica)

---

## 0. RESUMEN EJECUTIVO

### Objetivo
Encontrar correspondencias entre features de frames consecutivos para estimar movimiento de cámara.

### Resultado
- **200-500 matches** por par de frames
- **0.8ms en GPU** vs 5ms CPU
- **Ratio test 0.75** para filtrar falsos positivos

---

## 1. ANÁLISIS DEL CÓDIGO REAL

### 1.1 Creación del Matcher (`main.cpp:90-91`)

```cpp
// Feature matching (GPU accelerated)
cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
    cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
```

**¿Por qué NORM_HAMMING?**
- ORB produce descriptores **binarios**
- Hamming distance = XOR + popcount
- Mucho más rápido que L2 para binarios

### 1.2 KNN Matching con Ratio Test (`main.cpp:154-173`)

```cpp
// Match current frame with previous frame using Lowe's ratio test (GPU)
std::vector<cv::DMatch> good_matches;
int filtered_count = 0;

if (prev_frame &&
    !prev_frame->gpu_descriptors.empty() &&
    !current_frame.gpu_descriptors.empty()) {

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(prev_frame->gpu_descriptors,
                      current_frame.gpu_descriptors,
                      knn_matches, 2);  // k=2 para ratio test

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

**Análisis línea por línea:**

| Línea | Operación | Propósito |
|-------|-----------|-----------|
| 159 | `knnMatch(..., 2)` | Encuentra 2 mejores matches para cada descriptor |
| 162 | `knn[0].distance < 0.75 * knn[1].distance` | Ratio test de Lowe |
| 164-165 | `keypoints[queryIdx].pt` | Obtiene coordenadas 2D |
| 167-168 | `isInDynamicObject()` | Filtra matches en objetos móviles |

---

## 2. TEORÍA DEL MATCHING

### 2.1 Brute Force Matcher

```
Para cada descriptor d1 en frame1:
    Para cada descriptor d2 en frame2:
        calcular distance(d1, d2)
    Ordenar por distancia
    Retornar k mejores (kNN)
```

**Complejidad:** O(N × M) donde N, M = número de descriptores

### 2.2 Hamming Distance

```cpp
// Para descriptores binarios de 32 bytes:
int hammingDistance(const uchar* d1, const uchar* d2) {
    int distance = 0;
    for (int i = 0; i < 32; i++) {
        distance += __builtin_popcount(d1[i] ^ d2[i]);
    }
    return distance;
}
```

**GPU optimización:**
- CUDA tiene instrucción `__popc()` para popcount
- Procesa múltiples descriptores en paralelo

### 2.3 Lowe's Ratio Test

```
                    best_match
                        │
     ┌──────────────────┴──────────────────┐
     ▼                                      ▼
  ACEPTAR                                RECHAZAR
(match único)                        (match ambiguo)

if (best.distance < 0.75 * second_best.distance)
    → Match es distintivo, aceptar
else
    → Match es ambiguo, rechazar
```

**¿Por qué 0.75?**
- Valor de Lowe's paper original
- Equilibrio entre:
  - Más alto (0.8): Más matches, más falsos positivos
  - Más bajo (0.6): Menos matches, más falsos negativos

### 2.4 Estructura cv::DMatch

```cpp
struct DMatch {
    int queryIdx;   // Índice en descriptors del frame anterior
    int trainIdx;   // Índice en descriptors del frame actual
    int imgIdx;     // Índice de imagen (para múltiples imágenes)
    float distance; // Distancia entre descriptores
};
```

---

## 3. FILTRADO DE OBJETOS DINÁMICOS

### 3.1 Función isInDynamicObject (`main.cpp:43-50`)

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

### 3.2 Clases Dinámicas (`main.cpp:29-41`)

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

**¿Por qué filtrar?**
- SLAM asume mundo estático
- Features en objetos móviles → matches incorrectos → drift
- YOLO detecta objetos → excluimos esas regiones

---

## 4. FLUJO DE MATCHING EN GPU

```
┌─────────────────────────────────────────────────────────────────┐
│                         GPU (VRAM)                               │
│                                                                  │
│   prev_frame.gpu_descriptors    current_frame.gpu_descriptors   │
│         (2000 × 32)                    (2000 × 32)              │
│              │                              │                    │
│              └──────────┬───────────────────┘                    │
│                         │                                        │
│                         ▼                                        │
│              ┌─────────────────────┐                            │
│              │    knnMatch()       │                            │
│              │  (BFMatcher GPU)    │                            │
│              │                     │                            │
│              │ - Para cada d1:     │                            │
│              │   - Calcula dist    │                            │
│              │     a todos d2      │                            │
│              │   - Ordena          │                            │
│              │   - Retorna top 2   │                            │
│              └─────────────────────┘                            │
│                         │                                        │
└─────────────────────────┼────────────────────────────────────────┘
                          │ Download (implícito)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CPU (RAM)                                │
│                                                                  │
│   std::vector<std::vector<cv::DMatch>> knn_matches              │
│                         │                                        │
│                         ▼                                        │
│              ┌─────────────────────┐                            │
│              │   Ratio Test        │                            │
│              │   + Dynamic Filter  │                            │
│              └─────────────────────┘                            │
│                         │                                        │
│                         ▼                                        │
│   std::vector<cv::DMatch> good_matches (200-500)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. CONCEPTOS C++ UTILIZADOS

### 5.1 Range-based for con structured binding

```cpp
for (auto& knn : knn_matches) {
    // knn es std::vector<cv::DMatch>
    if (knn.size() >= 2) {
        // knn[0] = mejor match
        // knn[1] = segundo mejor
    }
}
```

### 5.2 std::set para lookup O(1)

```cpp
const std::set<int> DYNAMIC_CLASSES = { 0, 1, 2, ... };

// O(log n) lookup
if (DYNAMIC_CLASSES.count(det.class_id)) { ... }

// Alternativa C++20: std::unordered_set para O(1)
```

### 5.3 cv::Rect::contains()

```cpp
cv::Rect box(x, y, width, height);
cv::Point2f pt(px, py);

if (box.contains(pt)) {
    // pt está dentro del rectángulo
}
```

---

## 6. PREGUNTAS DE ENTREVISTA

### Q1: ¿Por qué usar kNN con k=2 en lugar de k=1?

**R:**
Con k=1 solo obtienes el mejor match, sin forma de evaluar su calidad.
Con k=2 puedes aplicar ratio test: si el mejor es **mucho mejor** que el segundo, es un match confiable.

```cpp
// Sin ratio test: muchos false positives
auto best = knn[0];

// Con ratio test: filtra ambiguos
if (knn[0].distance < 0.75 * knn[1].distance)
    // Match es distintivo
```

### Q2: ¿Cuál es la complejidad del matching?

**R:**
- **Brute Force**: O(N × M) donde N, M ≈ 2000
- **4 millones** de comparaciones por frame
- GPU paraleliza → ~0.8ms

### Q3: ¿Qué pasa si un objeto cubre la mitad del frame?

**R:**
- Muchos features filtrados → pocos matches
- `good_matches.size() < 8` → no se puede estimar pose
- El sistema pierde tracking temporalmente
- Recupera cuando el objeto sale del frame

### Q4: ¿Por qué los descriptors GPU no se descargan antes del matching?

**R:**
```cpp
matcher->knnMatch(prev_frame->gpu_descriptors,  // GPU
                  current_frame.gpu_descriptors, // GPU
                  knn_matches, 2);               // CPU output
```

- Matching **directamente en GPU**
- Solo el resultado (matches) se descarga
- Evita transferir 64KB × 2 de descriptores

### Q5: ¿Cómo mejorarías el matching para escenas con muchos objetos dinámicos?

**R:**
1. **Semantic segmentation** en lugar de bounding boxes (más preciso)
2. **Optical flow** para detectar movimiento
3. **Confidence scoring** basado en consistencia temporal
4. **RANSAC** más agresivo en pose estimation

### Q6: ¿Qué es queryIdx vs trainIdx?

**R:**
```cpp
// Terminología de OpenCV
Query  = frame anterior (el que "pregunta")
Train  = frame actual (la "base de datos")

match.queryIdx = índice en prev_frame.descriptors
match.trainIdx = índice en current_frame.descriptors
```

---

## 7. PERFORMANCE

### Comparación por método de matching

| Método | Tiempo (2000×2000) | Precisión |
|--------|-------------------|-----------|
| BF CPU | 5ms | 100% |
| BF GPU | 0.8ms | 100% |
| FLANN CPU | 2ms | ~98% |

**¿Por qué no FLANN?**
- FLANN es aproximado, puede perder matches
- Para ORB binario, BF GPU es suficientemente rápido
- Simplicidad > optimización prematura

### Estadísticas típicas

```
Total KNN matches:     2000
After ratio test:      600   (30% pasan)
After dynamic filter:  450   (25% del total)
Inliers after RANSAC:  350   (17.5% del total)
```

---

## 8. DIAGRAMA DE SECUENCIA

```
main.cpp                    DescriptorMatcher              GPU
    │                              │                        │
    │ knnMatch(gpu_desc1,          │                        │
    │          gpu_desc2, ...)     │                        │
    │─────────────────────────────►│                        │
    │                              │ Launch matching kernel │
    │                              │───────────────────────►│
    │                              │                        │[2000×2000 comparisons]
    │                              │◄───────────────────────│
    │                              │ Download results       │
    │◄─────────────────────────────│                        │
    │                              │                        │
    │ for (knn : knn_matches)      │                        │
    │   ratio test                 │                        │
    │   dynamic filter             │                        │
    │                              │                        │
    │ good_matches ready           │                        │
    │                              │                        │
```

---

## 9. CHECKLIST DE PREPARACIÓN

- [ ] Entender Hamming distance para binarios
- [ ] Saber explicar ratio test de Lowe
- [ ] Conocer significado de queryIdx/trainIdx
- [ ] Entender por qué k=2 en kNN
- [ ] Saber complejidad de BF matching
- [ ] Explicar filtrado de objetos dinámicos
- [ ] Conocer ventajas de matching en GPU
- [ ] Entender impacto de ratio threshold

---

**Generado:** 2025-01
**Proyecto:** aria-slam (C++)
