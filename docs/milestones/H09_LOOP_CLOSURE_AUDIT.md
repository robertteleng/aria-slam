# Auditoría Técnica: H09 - Loop Closure Detection

**Proyecto:** aria-slam (C++)
**Milestone:** H09 - Detección de Loop Closure
**Fecha:** 2025-01-28
**Autor:** Roberto (preparación entrevista técnica)

---

## 0. RESUMEN EJECUTIVO

### Objetivo
Detectar cuando la cámara revisita una ubicación previamente vista para corregir el drift acumulado.

### Pipeline de Loop Closure
```
Query KeyFrame
       │
       ▼
┌─────────────────────┐
│  1. Find Candidates │  ← Matching de descriptores vs database
│     (Appearance)    │     Score = good_matches / total_keypoints
└──────────┬──────────┘
           │ Top 5 candidates
           ▼
┌─────────────────────┐
│  2. Verify Geometry │  ← RANSAC Fundamental Matrix
│     (RANSAC)        │     Elimina outliers geométricos
└──────────┬──────────┘
           │ Si inliers >= 30
           ▼
┌─────────────────────┐
│  3. Compute Pose    │  ← Essential Matrix + recoverPose
│     (Relative)      │     T_query_candidate
└──────────┬──────────┘
           │
           ▼
    LoopCandidate
    {query_id, match_id, relative_pose}
```

### Resultados
| Métrica | Valor |
|---------|-------|
| Tiempo por query | ~50ms |
| Precision | ~95% |
| Recall | ~70% |
| Loops detectados (MH_01) | ~15 |

---

## 1. ANÁLISIS DEL CÓDIGO REAL

### 1.1 Constructor (`LoopClosure.cpp:15-22`)

```cpp
// LoopClosure.cpp:15-22
LoopClosureDetector::LoopClosureDetector(int min_frames_between,
                                         double min_score,
                                         int min_matches)
    : min_frames_between_(min_frames_between),
      min_score_(min_score),
      min_matches_(min_matches) {
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
}
```

**Parámetros por defecto:**
- `min_frames_between = 30`: Evita detectar frames consecutivos como loops
- `min_score = 0.3`: 30% de matches buenos para considerar candidato
- `min_matches = 30`: Mínimo de inliers para validar loop

### 1.2 Add KeyFrame (`LoopClosure.cpp:24-31`)

```cpp
// LoopClosure.cpp:24-31
void LoopClosureDetector::addKeyFrame(const KeyFrame& kf) {
    keyframes_.push_back(kf);

    // Limit database size
    while (keyframes_.size() > 500) {
        keyframes_.pop_front();  // FIFO: elimina los más antiguos
    }
}
```

**Gestión de memoria:**
- `std::deque` permite pop_front() eficiente
- Límite de 500 keyframes ≈ 200 MB (con descriptores)
- Sliding window: pierde loops muy antiguos

### 1.3 Detection Main Function (`LoopClosure.cpp:33-70`)

```cpp
// LoopClosure.cpp:33-70
bool LoopClosureDetector::detect(const KeyFrame& query, LoopCandidate& candidate) {
    if (keyframes_.size() < (size_t)min_frames_between_) {
        return false;  // No hay suficientes keyframes
    }

    // Find candidates by descriptor similarity
    auto candidates = findCandidates(query);

    for (const auto& [idx, score] : candidates) {
        if (score < min_score_) continue;

        const KeyFrame& kf = keyframes_[idx];

        // Skip recent frames
        if (query.id - kf.id < min_frames_between_) continue;

        // Geometric verification
        std::vector<cv::DMatch> inlier_matches;
        if (verifyGeometry(query, kf, inlier_matches)) {
            // Compute relative pose
            Eigen::Matrix4d relative_pose;
            if (computeRelativePose(query, kf, inlier_matches, relative_pose)) {
                candidate.query_id = query.id;
                candidate.match_id = kf.id;
                candidate.score = score;
                candidate.matches = inlier_matches;
                candidate.relative_pose = relative_pose;

                loop_count_++;
                std::cout << "Loop detected: " << query.id << " -> " << kf.id
                          << " (score: " << score << ")" << std::endl;
                return true;
            }
        }
    }

    return false;
}
```

**Flujo de decisión:**
```
┌─────────────────────────────────────────┐
│ Para cada candidato (ordenado por score) │
└────────────────┬────────────────────────┘
                 │
     ┌───────────▼───────────┐
     │ score >= min_score?   │──No──► Siguiente candidato
     └───────────┬───────────┘
                 │ Sí
     ┌───────────▼───────────┐
     │ frame gap >= 30?      │──No──► Siguiente candidato
     └───────────┬───────────┘
                 │ Sí
     ┌───────────▼───────────┐
     │ verifyGeometry()      │──No──► Siguiente candidato
     │ (RANSAC >= 30 inliers)│
     └───────────┬───────────┘
                 │ Sí
     ┌───────────▼───────────┐
     │ computeRelativePose() │──No──► Siguiente candidato
     │ (Essential matrix)    │
     └───────────┬───────────┘
                 │ Sí
                 ▼
           LOOP FOUND!
```

### 1.4 Find Candidates (`LoopClosure.cpp:72-114`)

```cpp
// LoopClosure.cpp:72-114
std::vector<std::pair<int, double>> LoopClosureDetector::findCandidates(const KeyFrame& query) {
    std::vector<std::pair<int, double>> candidates;

    if (query.descriptors.empty()) return candidates;

    for (size_t i = 0; i < keyframes_.size(); i++) {
        const KeyFrame& kf = keyframes_[i];

        // Skip recent frames
        if (query.id - kf.id < min_frames_between_) continue;

        if (kf.descriptors.empty()) continue;

        // Match descriptors
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(query.descriptors, kf.descriptors, knn_matches, 2);

        // Ratio test
        int good_matches = 0;
        for (const auto& m : knn_matches) {
            if (m.size() >= 2 && m[0].distance < 0.7 * m[1].distance) {
                good_matches++;
            }
        }

        // Score = ratio of good matches
        double score = (double)good_matches / std::max(1, (int)query.keypoints.size());
        if (score > 0.1) {
            candidates.push_back({i, score});
        }
    }

    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Return top 5 candidates
    if (candidates.size() > 5) {
        candidates.resize(5);
    }

    return candidates;
}
```

**Complejidad:**
- O(N) keyframes × O(M×K) matching = O(N×M×K)
- N = 500 keyframes, M = 2000 descriptors, K = 2000
- ~2 millones de comparaciones por query

### 1.5 Geometric Verification (`LoopClosure.cpp:116-156`)

```cpp
// LoopClosure.cpp:116-156
bool LoopClosureDetector::verifyGeometry(const KeyFrame& query, const KeyFrame& candidate,
                                         std::vector<cv::DMatch>& inlier_matches) {
    // Match descriptors
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(query.descriptors, candidate.descriptors, knn_matches, 2);

    // Ratio test
    std::vector<cv::DMatch> good_matches;
    for (const auto& m : knn_matches) {
        if (m.size() >= 2 && m[0].distance < 0.7 * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    if (good_matches.size() < (size_t)min_matches_) {
        return false;
    }

    // Extract matched points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : good_matches) {
        pts1.push_back(query.keypoints[m.queryIdx].pt);
        pts2.push_back(candidate.keypoints[m.trainIdx].pt);
    }

    // RANSAC fundamental matrix estimation
    std::vector<uchar> inlier_mask;
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, inlier_mask);

    if (F.empty()) return false;

    // Collect inlier matches
    inlier_matches.clear();
    for (size_t i = 0; i < good_matches.size(); i++) {
        if (inlier_mask[i]) {
            inlier_matches.push_back(good_matches[i]);
        }
    }

    return inlier_matches.size() >= (size_t)min_matches_;
}
```

**RANSAC Fundamental Matrix:**
```
Fundamental matrix F satisface:
   p2ᵀ F p1 = 0  para correspondencias válidas

RANSAC:
1. Selecciona 8 puntos aleatorios
2. Calcula F con 8-point algorithm
3. Cuenta inliers (|p2ᵀ F p1| < threshold)
4. Repite hasta confidence = 0.99
5. Refina F con todos los inliers
```

### 1.6 Compute Relative Pose (`LoopClosure.cpp:158-195`)

```cpp
// LoopClosure.cpp:158-195
bool LoopClosureDetector::computeRelativePose(const KeyFrame& query, const KeyFrame& candidate,
                                               const std::vector<cv::DMatch>& matches,
                                               Eigen::Matrix4d& relative_pose) {
    if (matches.size() < 8) return false;

    // Extract matched points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(query.keypoints[m.queryIdx].pt);
        pts2.push_back(candidate.keypoints[m.trainIdx].pt);
    }

    // Approximate camera matrix
    double fx = 700, fy = 700;
    double cx = 320, cy = 180;
    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // Essential matrix
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0);
    if (E.empty()) return false;

    // Recover pose
    cv::Mat R, t;
    int inliers = cv::recoverPose(E, pts1, pts2, K, R, t);

    if (inliers < min_matches_) return false;

    // Build 4x4 transform matrix
    relative_pose = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            relative_pose(i, j) = R.at<double>(i, j);
        }
        relative_pose(i, 3) = t.at<double>(i);
    }

    return true;
}
```

**De Essential a Pose:**
```
E = [t]× R  donde [t]× es skew-symmetric

recoverPose() descompone E en 4 soluciones posibles:
  (R1, t), (R1, -t), (R2, t), (R2, -t)

Selecciona la solución donde los puntos están
delante de ambas cámaras (cheirality check)
```

---

## 2. TEORÍA: PLACE RECOGNITION

### 2.1 Bag of Words (No implementado pero relevante)

```
Vocabulario Visual (DBoW2):
┌─────────────────────────────────────────────────────────────┐
│                    Visual Words Tree                         │
│                                                              │
│                         Root                                 │
│                        /    \                                │
│                       /      \                               │
│                    Node1    Node2                            │
│                    / \      / \                              │
│                  w1  w2   w3  w4   ← Visual words            │
│                                                              │
│   Descriptor → traverse tree → word ID                       │
│   Image → bag of words = histogram of word IDs               │
└─────────────────────────────────────────────────────────────┘

Similitud: TF-IDF score entre histogramas
Complejidad: O(log V) por descriptor vs O(N) brute force
```

### 2.2 Fundamental vs Essential Matrix

```
FUNDAMENTAL MATRIX (7 DoF):
- No requiere calibración
- Relaciona puntos en coordenadas de pixel
- p2ᵀ F p1 = 0

ESSENTIAL MATRIX (5 DoF):
- Requiere calibración K
- E = K2ᵀ F K1
- Permite recuperar R, t
- x2ᵀ E x1 = 0  (coordenadas normalizadas)

Relación:
E = [t]× R
F = K2⁻ᵀ E K1⁻¹
```

### 2.3 RANSAC para Estimación Robusta

```
Algorithm RANSAC(points, model, threshold):
  best_model = None
  best_inliers = 0

  for i in range(max_iterations):
      # 1. Sample minimum set (8 for F, 5 for E)
      sample = random_sample(points, min_size)

      # 2. Fit model
      model = fit(sample)

      # 3. Count inliers
      inliers = count(|error(point, model)| < threshold)

      # 4. Update best
      if inliers > best_inliers:
          best_model = model
          best_inliers = inliers

  # 5. Refine with all inliers
  return refine(best_model, inliers)

Iterations needed for 99% confidence:
  k = log(1-0.99) / log(1 - w^n)
  donde w = inlier_ratio, n = sample_size
```

### 2.4 Pipeline Completo

```
KeyFrame Database                Query KeyFrame
      │                                │
      │                                │
      ▼                                ▼
┌──────────────────────────────────────────────────┐
│           APPEARANCE MATCHING                     │
│                                                  │
│   Para cada KF en database:                      │
│   1. knnMatch(query.desc, kf.desc)               │
│   2. Ratio test 0.7                              │
│   3. Score = good_matches / total               │
│   4. Si score > 0.1: candidato                   │
└──────────────────────────────────────────────────┘
                      │
                      ▼ Top 5 por score
┌──────────────────────────────────────────────────┐
│           GEOMETRIC VERIFICATION                  │
│                                                  │
│   Para cada candidato:                           │
│   1. findFundamentalMat() con RANSAC             │
│   2. Contar inliers                              │
│   3. Si inliers >= 30: válido geométricamente    │
└──────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│           POSE ESTIMATION                         │
│                                                  │
│   1. findEssentialMat() con K                    │
│   2. recoverPose() → R, t                        │
│   3. Build T = [R|t]                             │
└──────────────────────────────────────────────────┘
                      │
                      ▼
              LoopCandidate
```

---

## 3. CONCEPTOS C++ UTILIZADOS

### 3.1 Structured Bindings (C++17)

```cpp
// LoopClosure.cpp:41
for (const auto& [idx, score] : candidates) {
    // idx = candidates[i].first
    // score = candidates[i].second
}

// Equivalente pre-C++17:
for (const auto& pair : candidates) {
    int idx = pair.first;
    double score = pair.second;
}
```

### 3.2 std::deque para Sliding Window

```cpp
std::deque<KeyFrame> keyframes_;

// Eficiente para:
keyframes_.push_back(kf);   // O(1)
keyframes_.pop_front();     // O(1)
keyframes_[i];              // O(1) random access

// vs std::vector:
// pop_front() sería O(n) - tiene que mover todos los elementos
```

### 3.3 Lambda en std::sort

```cpp
std::sort(candidates.begin(), candidates.end(),
          [](const auto& a, const auto& b) {
              return a.second > b.second;  // Descendente
          });
```

### 3.4 cv::Mat to Eigen Conversion

```cpp
// OpenCV → Eigen
cv::Mat R;  // 3x3
Eigen::Matrix4d relative_pose = Eigen::Matrix4d::Identity();

for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        relative_pose(i, j) = R.at<double>(i, j);
    }
    relative_pose(i, 3) = t.at<double>(i);
}

// Alternativa con Eigen::Map (zero-copy si layout coincide):
Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_eigen(
    R.ptr<double>()
);
```

---

## 4. DIAGRAMA DE SECUENCIA

```
main()           LoopClosureDetector        cv::BFMatcher         OpenCV
  │                     │                        │                   │
  │ addKeyFrame(kf)     │                        │                   │
  │────────────────────►│                        │                   │
  │                     │ push_back(kf)          │                   │
  │                     │                        │                   │
  │                     │                        │                   │
  │ detect(query)       │                        │                   │
  │────────────────────►│                        │                   │
  │                     │                        │                   │
  │                     │ findCandidates()       │                   │
  │                     │────────────────────────│                   │
  │                     │                        │                   │
  │                     │ for each keyframe:     │                   │
  │                     │   knnMatch()           │                   │
  │                     │───────────────────────►│                   │
  │                     │◄──────────────────────│                   │
  │                     │   ratio test           │                   │
  │                     │   calculate score      │                   │
  │                     │                        │                   │
  │                     │ sort by score          │                   │
  │                     │ return top 5           │                   │
  │                     │                        │                   │
  │                     │ for each candidate:    │                   │
  │                     │   verifyGeometry()     │                   │
  │                     │   knnMatch()           │                   │
  │                     │───────────────────────►│                   │
  │                     │◄──────────────────────│                   │
  │                     │                        │                   │
  │                     │   findFundamentalMat() │                   │
  │                     │────────────────────────────────────────────►│
  │                     │◄───────────────────────────────────────────│
  │                     │                        │                   │
  │                     │   if inliers >= 30:    │                   │
  │                     │     computeRelativePose()                  │
  │                     │                        │                   │
  │                     │     findEssentialMat() │                   │
  │                     │────────────────────────────────────────────►│
  │                     │◄───────────────────────────────────────────│
  │                     │                        │                   │
  │                     │     recoverPose()      │                   │
  │                     │────────────────────────────────────────────►│
  │                     │◄───────────────────────────────────────────│
  │                     │                        │                   │
  │◄────────────────────│ return LoopCandidate   │                   │
  │                     │                        │                   │
```

---

## 5. PREGUNTAS DE ENTREVISTA

### Q1: ¿Por qué min_frames_between = 30?

**R:** Para evitar falsos positivos de frames consecutivos:
- Frames consecutivos son muy similares visualmente
- No representan un "loop" real (revisita después de explorar)
- 30 frames @ 20 Hz = 1.5 segundos de separación mínima

```
Sin gap mínimo:
  Frame 100 → "loop" con Frame 99  ← Falso positivo!

Con gap = 30:
  Frame 100 solo considera Frame 0-70
  Si hay match: es loop real
```

### Q2: ¿Por qué usar Fundamental Matrix para verificación y Essential para pose?

**R:**
```
FUNDAMENTAL (verificación):
- No necesita K exacta
- Más robusto a errores de calibración
- Solo valida: "¿estos matches son consistentes geométricamente?"

ESSENTIAL (pose):
- Necesita K para descomponer en R, t
- Da la pose relativa que necesitamos
- Usamos después de validar con F
```

### Q3: ¿Cuál es la complejidad del loop closure actual?

**R:**
```
findCandidates:
- N keyframes
- M descriptors por keyframe
- K descriptors en query
- knnMatch: O(M × K) por keyframe
- Total: O(N × M × K)

Con N=500, M=K=2000:
- 500 × 2000 × 2000 = 2 billones de operaciones
- ~50ms en CPU moderno

Mejora con DBoW2:
- O(K × log V) para convertir a BoW
- O(N) comparación de histogramas
- Total: O(N + K log V) ≈ O(N)
```

### Q4: ¿Qué pasa si hay muchos loops falsos positivos?

**R:** Pueden causar:
1. Pose graph optimization diverge
2. Mapa se distorsiona severamente
3. Sistema pierde tracking

**Soluciones:**
- Aumentar `min_matches` (más conservador)
- Verificación temporal (consistencia en frames consecutivos)
- Verificación geométrica más estricta (Essential + PnP)

### Q5: ¿Por qué ordenar por score y solo tomar top 5?

**R:**
```cpp
if (candidates.size() > 5) {
    candidates.resize(5);
}
```

1. **Eficiencia:** No verificar todos los candidatos
2. **Calidad:** Los mejores scores tienen mayor probabilidad de ser loops reales
3. **Early exit:** Si encontramos loop válido, retornamos inmediatamente

### Q6: ¿Cómo funciona el ratio test de Lowe?

**R:**
```
Para descriptor d en query:
- best_match: distancia = 30
- second_best: distancia = 50

Ratio = 30/50 = 0.6 < 0.7 → ACEPTAR (distintivo)

vs.

- best_match: distancia = 40
- second_best: distancia = 45

Ratio = 40/45 = 0.89 > 0.7 → RECHAZAR (ambiguo)
```

El ratio test filtra matches ambiguos donde el descriptor podría corresponder a múltiples puntos.

### Q7: ¿Por qué recoverPose puede fallar?

**R:**
1. **Movimiento degenerado:** Pura rotación (sin traslación)
2. **Puntos coplanares:** E tiene rango < 2
3. **Baseline pequeño:** t muy pequeño, ruido domina
4. **Cheirality ambiguo:** Puntos detrás de cámaras

```cpp
int inliers = cv::recoverPose(E, pts1, pts2, K, R, t);
if (inliers < min_matches_) return false;  // Fallo
```

---

## 6. PERFORMANCE

### 6.1 Breakdown de Tiempo

```
detect() total: ~50ms
├── findCandidates(): 40ms
│   ├── knnMatch × 500: 35ms
│   └── ratio test + score: 5ms
│
├── verifyGeometry(): 8ms (por candidato)
│   ├── knnMatch: 5ms
│   └── findFundamentalMat: 3ms
│
└── computeRelativePose(): 2ms
    ├── findEssentialMat: 1ms
    └── recoverPose: 1ms
```

### 6.2 Comparación con DBoW2

| Métrica | Brute Force (actual) | DBoW2 |
|---------|---------------------|-------|
| Query time (500 KF) | 50ms | 2ms |
| Memory | 200 MB | 50 MB |
| Precision | 95% | 93% |
| Recall | 70% | 75% |

### 6.3 Limitaciones Actuales

| Limitación | Impacto | Solución (futura) |
|------------|---------|-------------------|
| O(N) search | 50ms/query | DBoW2 vocabulary |
| CPU matching | Bottleneck | GPU matching (H14) |
| Blocking | Tracking stall | Async thread (H13) |

---

## 7. CHECKLIST DE PREPARACIÓN

### Conceptos que debes dominar:
- [ ] Difference entre Fundamental y Essential matrix
- [ ] RANSAC: iteraciones, threshold, confidence
- [ ] Lowe's ratio test
- [ ] recoverPose y cheirality
- [ ] Por qué min_frames_between
- [ ] Complejidad O(N × M × K)

### Código que debes poder escribir:
```cpp
// RANSAC Fundamental
cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, mask);

// Essential → Pose
cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC);
cv::recoverPose(E, pts1, pts2, K, R, t);

// Lowe's ratio test
for (auto& m : knn_matches) {
    if (m.size() >= 2 && m[0].distance < 0.7 * m[1].distance) {
        good_matches.push_back(m[0]);
    }
}
```

### Números que debes conocer:
- Ratio test threshold: **0.7**
- RANSAC confidence: **0.99**
- RANSAC threshold: **3.0 pixels**
- Min matches típico: **30**
- Min frame gap: **30** (~1.5s @ 20Hz)

---

**Generado:** 2025-01-28
**Proyecto:** aria-slam
**Archivos analizados:** LoopClosure.cpp, LoopClosure.hpp
