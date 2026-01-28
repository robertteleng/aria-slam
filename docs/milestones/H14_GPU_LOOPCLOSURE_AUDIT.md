# Auditoría Técnica: H14 - GPU-Accelerated Loop Closure

**Proyecto:** aria-slam (C++)
**Milestone:** H14 - Loop Closure acelerado por GPU
**Fecha:** 2025-01-28
**Autor:** Roberto (preparación entrevista técnica)

---

## 0. RESUMEN EJECUTIVO

### Objetivo
Mover la detección de loop closure a GPU para aliviar el cuello de botella de CPU, especialmente crítico para Jetson Orin Nano.

### Motivación
```
CPU Loop Closure (actual):
- Búsqueda lineal: O(n) keyframes
- Matching por candidato: ~50ms
- Bloquea tracking thread en CPUs débiles

GPU Loop Closure (propuesto):
- Matching paralelo de todos los keyframes
- Batch descriptor matching: ~5ms total
- Async, no bloquea tracking
```

### Resultados Esperados
| Operación | CPU | GPU (RTX 2060) | Speedup |
|-----------|-----|----------------|---------|
| Match 1 KF | 5ms | 0.8ms | 6x |
| Match 500 KF | 250ms | 8ms | **31x** |
| Full loop detect | 300ms | 15ms | **20x** |

---

## 1. DISEÑO: GPU KEYFRAME DATABASE

### 1.1 Estructura de Datos

```cpp
class GpuKeyframeDatabase {
public:
    GpuKeyframeDatabase(int max_keyframes = 500, int descriptor_size = 32) {
        // Allocate GPU memory for all descriptors
        // Shape: [max_keyframes * max_features, descriptor_size]
        d_descriptors_.create(max_keyframes * MAX_FEATURES, descriptor_size, CV_8UC1);
        d_counts_.create(1, max_keyframes, CV_32SC1);
    }

    void addKeyframe(int id, const cv::Mat& descriptors) {
        int offset = id * MAX_FEATURES;

        // Upload descriptors to GPU
        cv::cuda::GpuMat region = d_descriptors_.rowRange(offset, offset + descriptors.rows);
        region.upload(descriptors);

        // Update count
        counts_[id] = descriptors.rows;
        d_counts_.upload(cv::Mat(counts_));

        keyframe_ids_.push_back(id);
    }

    cv::cuda::GpuMat getDescriptors(int id) const {
        int offset = id * MAX_FEATURES;
        int count = counts_[id];
        return d_descriptors_.rowRange(offset, offset + count);
    }

private:
    static const int MAX_FEATURES = 2000;
    cv::cuda::GpuMat d_descriptors_;  // [N*2000, 32] uint8
    cv::cuda::GpuMat d_counts_;       // [1, N] int32
    std::vector<int> counts_;
    std::vector<int> keyframe_ids_;
};
```

**Layout de memoria GPU:**
```
d_descriptors_ layout:
┌────────────────────────────────────────────────────┐
│ KF0: desc[0..n0-1]  │ padding to MAX_FEATURES      │  offset: 0
├────────────────────────────────────────────────────┤
│ KF1: desc[0..n1-1]  │ padding                      │  offset: MAX_FEATURES
├────────────────────────────────────────────────────┤
│ KF2: desc[0..n2-1]  │ padding                      │  offset: 2*MAX_FEATURES
├────────────────────────────────────────────────────┤
│ ...                                                │
└────────────────────────────────────────────────────┘

Cada descriptor: 32 bytes (ORB)
Total para 500 KF: 500 * 2000 * 32 = 32 MB
```

### 1.2 Memory Estimation

```
GPU Memory Budget para Loop Closure:

Descriptors:     500 KF * 2000 features * 32 bytes = 32 MB
Counts:          500 * 4 bytes = 2 KB
Match buffers:   2000 * 500 * 4 bytes = 4 MB (distance matrix)
Working memory:  ~10 MB

Total: ~50 MB dedicados a loop closure

RTX 2060: 6 GB disponibles
Jetson Orin: 8 GB compartidos (reservar 1 GB para LC)
```

---

## 2. DISEÑO: BATCH GPU MATCHER

### 2.1 Matching Paralelo

```cpp
class GpuBatchMatcher {
public:
    GpuBatchMatcher() {
        matcher_ = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    }

    // Match query against all keyframes in parallel
    std::vector<MatchResult> matchAll(const cv::cuda::GpuMat& query_desc,
                                       const GpuKeyframeDatabase& db,
                                       cv::cuda::Stream& stream) {
        std::vector<MatchResult> results;

        for (int id : db.getKeyframeIds()) {
            cv::cuda::GpuMat kf_desc = db.getDescriptors(id);

            // Async matching
            matcher_->knnMatchAsync(query_desc, kf_desc, d_matches_, 2,
                                    cv::noArray(), stream);
        }

        stream.waitForCompletion();

        // Download and apply ratio test
        for (size_t i = 0; i < results.size(); i++) {
            results[i].score = applyRatioTest(results[i].matches);
        }

        return results;
    }

private:
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher_;
    cv::cuda::GpuMat d_matches_;
};
```

**Pipeline de ejecución:**
```
CPU                          GPU
 │                            │
 │ for each KF:               │
 │   knnMatchAsync() ────────►│ [Queue work]
 │   (returns immediately)    │
 │                            │
 │ waitForCompletion() ──────►│ [Execute all]
 │◄───────────────────────────│
 │                            │
 │ Download + ratio test      │
 │                            │
```

### 2.2 Optimización: Batch en Single Kernel

```cpp
// Versión más eficiente: una sola operación de matching
void matchAllBatched(const cv::cuda::GpuMat& query_desc,
                     const cv::cuda::GpuMat& all_descriptors,
                     const std::vector<int>& boundaries) {
    // Match query contra TODOS los descriptores concatenados
    cv::cuda::GpuMat all_matches;
    matcher_->knnMatch(query_desc, all_descriptors, all_matches, 2);

    // Post-process: separar matches por keyframe usando boundaries
    for (size_t i = 0; i < boundaries.size() - 1; i++) {
        // Extraer matches para keyframe i
        auto kf_matches = extractMatches(all_matches, boundaries[i], boundaries[i+1]);
        results.push_back({i, computeScore(kf_matches)});
    }
}
```

---

## 3. DISEÑO: GPU RANSAC (OPCIONAL AVANZADO)

### 3.1 CUDA Kernel para RANSAC Paralelo

```cpp
__global__ void ransacFundamentalKernel(
    const float2* pts1,          // Query points
    const float2* pts2,          // Match points
    int num_points,
    const int* sample_indices,   // Pre-generated random samples
    int num_samples,
    float* scores,               // Output: inlier count per sample
    float3x3* models)            // Output: F matrix per sample
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_samples) return;

    // Get 8 random points for this sample
    int* idx = sample_indices + tid * 8;

    // Compute fundamental matrix from 8 points (8-point algorithm)
    float3x3 F = computeFundamental8Point(pts1, pts2, idx);

    // Count inliers in parallel (within warp)
    int inliers = 0;
    for (int i = 0; i < num_points; i++) {
        float error = computeEpipolarError(F, pts1[i], pts2[i]);
        if (error < 3.0f) inliers++;
    }

    scores[tid] = inliers;
    models[tid] = F;
}
```

**Paralelismo:**
```
CPU RANSAC:                    GPU RANSAC:
──────────────                 ──────────────
for i in 1000:                 Launch 1000 threads:
  sample 8 pts                   Thread 0: sample 0 → F0 → count inliers
  compute F                      Thread 1: sample 1 → F1 → count inliers
  count inliers                  ...
  update best                    Thread 999: sample 999 → F999 → count inliers
                                Reduce: find max inliers

Tiempo: 1000 * 0.1ms = 100ms   Tiempo: ~2ms (paralelo)
```

---

## 4. DISEÑO: INTEGRATED GPU LOOP DETECTOR

### 4.1 Clase Principal

```cpp
class GpuLoopClosureDetector {
public:
    GpuLoopClosureDetector(int min_gap = 30, double min_score = 0.3)
        : min_gap_(min_gap), min_score_(min_score) {}

    void addKeyframe(const KeyFrame& kf) {
        // Upload descriptors to GPU database
        d_database_.addKeyframe(kf.id, kf.descriptors);
        keyframes_.push_back(kf);
    }

    bool detect(const KeyFrame& query, LoopCandidate& candidate) {
        // Upload query descriptors
        cv::cuda::GpuMat d_query;
        d_query.upload(query.descriptors);

        // Batch match against all keyframes on GPU
        auto results = matcher_.matchAll(d_query, d_database_, stream_);

        // Find best candidate (CPU, results already downloaded)
        for (const auto& result : results) {
            if (result.score < min_score_) continue;
            if (query.id - result.kf_id < min_gap_) continue;

            // Geometric verification (can also be GPU)
            if (verifyGeometry(query, keyframes_[result.kf_id],
                               result.matches, candidate)) {
                return true;
            }
        }

        return false;
    }

private:
    GpuKeyframeDatabase d_database_;
    GpuBatchMatcher matcher_;
    cv::cuda::Stream stream_;

    std::vector<KeyFrame> keyframes_;
    int min_gap_;
    double min_score_;
};
```

### 4.2 Integración con Async Pipeline

```cpp
class AsyncGpuLoopDetector {
public:
    void run() {
        while (!stop_) {
            auto kf = keyframe_queue_.pop();
            if (!kf) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // GPU detection (non-blocking CPU during GPU work)
            LoopCandidate candidate;
            if (detector_.detect(*kf, candidate)) {
                loop_queue_.push(candidate);
            }

            // CPU is free during GPU work for:
            // - Preparing next keyframe
            // - Other housekeeping
        }
    }

private:
    GpuLoopClosureDetector detector_;
    LockFreeQueue<KeyFrame> keyframe_queue_;
    LockFreeQueue<LoopCandidate> loop_queue_;
    std::atomic<bool> stop_{false};
};
```

---

## 5. TEORÍA: GPU DESCRIPTOR MATCHING

### 5.1 Brute Force Matcher en GPU

```
Query descriptors: Q[M, 32]   (M features)
Train descriptors: T[N, 32]   (N features)

Distance matrix: D[M, N]

Para cada par (i, j):
  D[i,j] = hamming_distance(Q[i], T[j])
         = popcount(Q[i] XOR T[j])

GPU paralleliza:
- Un thread por elemento de D
- Blocks: (M/16, N/16)
- Threads per block: (16, 16)

Con M=N=2000:
- 4M comparaciones
- ~1000 threads activos en RTX 2060
- Cada thread: ~10 elementos
```

### 5.2 Optimización con Shared Memory

```cpp
__global__ void hammingMatchKernel(
    const uint8_t* query,   // [M, 32]
    const uint8_t* train,   // [N, 32]
    int* distances,         // [M, N]
    int M, int N)
{
    __shared__ uint8_t s_query[16][32];  // Cache query descriptors
    __shared__ uint8_t s_train[16][32];  // Cache train descriptors

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int qidx = blockIdx.y * 16 + ty;
    int tidx = blockIdx.x * 16 + tx;

    // Load to shared memory
    if (qidx < M) {
        for (int i = tx; i < 32; i += 16) {
            s_query[ty][i] = query[qidx * 32 + i];
        }
    }
    if (tidx < N) {
        for (int i = ty; i < 32; i += 16) {
            s_train[tx][i] = train[tidx * 32 + i];
        }
    }
    __syncthreads();

    // Compute distance
    if (qidx < M && tidx < N) {
        int dist = 0;
        for (int i = 0; i < 32; i++) {
            dist += __popc(s_query[ty][i] ^ s_train[tx][i]);
        }
        distances[qidx * N + tidx] = dist;
    }
}
```

### 5.3 Memory Bandwidth Analysis

```
Sin shared memory:
  Query reads: M * N * 32 = 2000 * 2000 * 32 = 128 MB
  Train reads: M * N * 32 = 128 MB
  Total: 256 MB → 256 MB / 400 GB/s = 0.64 ms (memory bound)

Con shared memory:
  Query reads: M * 32 = 64 KB (una vez por bloque)
  Train reads: N * 32 = 64 KB (una vez por bloque)
  Total: ~128 KB → casi todo desde shared memory
  Speedup: ~10x en memoria

Resultado: de memory-bound a compute-bound
```

---

## 6. PREGUNTAS DE ENTREVISTA

### Q1: ¿Por qué 31x speedup para 500 keyframes pero solo 6x para 1?

**R:**
```
GPU overhead:
- Kernel launch: ~5 μs
- Memory transfer: ~0.1 ms

Para 1 keyframe:
  CPU: 5 ms
  GPU: 0.1 ms (transfer) + 0.5 ms (compute) + 0.2 ms (download) = 0.8 ms
  Speedup: 5 / 0.8 = 6x

Para 500 keyframes:
  CPU: 500 * 5 ms = 2500 ms (secuencial)
  GPU: 0.1 ms + 8 ms (compute, paralelo!) + 0.2 ms = 8.3 ms
  Speedup: 2500 / 8.3 = 301x teórico, ~31x real

El GPU amortiza overhead cuando hay mucho trabajo paralelo.
```

### Q2: ¿Cómo manejar más keyframes que caben en GPU memory?

**R:**
```cpp
// Estrategia 1: Sliding window
void addKeyframe(const KeyFrame& kf) {
    if (d_database_.size() >= MAX_GPU_KEYFRAMES) {
        d_database_.removeOldest();  // FIFO eviction
    }
    d_database_.add(kf);
}

// Estrategia 2: Hierarchical search
// 1. Coarse search con vocabulario en CPU
// 2. Fine search solo contra top candidates en GPU

// Estrategia 3: Streaming
// Dividir database en chunks
// Procesar chunk por chunk en GPU
for (chunk : database.chunks()) {
    uploadChunk(chunk);
    matchChunk(query, chunk);
    results.merge(chunk_results);
}
```

### Q3: ¿Cuándo mantener datos en GPU vs transferir?

**R:**
```
MANTENER EN GPU cuando:
- Datos se reutilizan múltiples veces (descriptors database)
- Próximo paso también es GPU (matching → RANSAC)
- Tamaño < GPU memory disponible

TRANSFERIR cuando:
- Datos se usan solo una vez (frame actual)
- Siguiente paso es CPU-only (pose graph optimization)
- GPU memory está escasa

En loop closure:
- Database: SIEMPRE en GPU (se reutiliza para cada query)
- Query descriptors: temporal (upload → match → descartar)
- Match results: download para ratio test en CPU
```

### Q4: ¿Por qué RANSAC en GPU es más complejo?

**R:**
```
Challenges:
1. Random sampling: GPU no tiene buen RNG
   Solución: pre-generar samples en CPU

2. Data-dependent branching: diferentes paths
   Solución: procesar todos los samples, filtrar después

3. Reducción: encontrar mejor modelo
   Solución: parallel reduction (tree-based)

4. Memoria: cada thread necesita su modelo
   Solución: shared memory o registers

Alternativa: híbrido
- GPU: compute distances matrix
- CPU: RANSAC sobre matches descargados
- Más simple, casi tan rápido para loop closure
```

### Q5: ¿Cómo sincronizar GPU loop closure con tracking thread?

**R:**
```cpp
// Producer (tracking thread)
void onNewKeyframe(const KeyFrame& kf) {
    // Non-blocking push
    keyframe_queue_.push(kf);
}

// Consumer (loop closure thread)
void loopClosureLoop() {
    while (!stop_) {
        auto kf = keyframe_queue_.pop();
        if (!kf) continue;

        // GPU work - CPU free durante esto
        auto candidate = detector_.detect(*kf);

        if (candidate) {
            // Notify main thread
            loop_callback_(candidate);
        }
    }
}

// No locks compartidos entre threads
// Comunicación solo via lock-free queues
// GPU work no bloquea CPU tracking
```

### Q6: ¿Cuál es el bottleneck actual y cómo lo resuelve GPU?

**R:**
```
CPU bottleneck actual:
┌───────────────────────────────────────────────┐
│ Frame N:  [Track 15ms][Loop 50ms][Graph 25ms] │
│ Frame N+1: ────────────[Track]──────────────  │
│                        ↑                      │
│               Esperando loop closure          │
└───────────────────────────────────────────────┘

Con GPU + async:
┌───────────────────────────────────────────────┐
│ Frame N:  [Track 15ms]                        │
│ Frame N+1: [Track 15ms]                       │
│ GPU:       ───────────[Loop 5ms]────          │
│ Graph:                          [Opt 25ms]    │
└───────────────────────────────────────────────┘

Tracking ya no espera loop closure
GPU hace matching en paralelo
FPS limitado solo por tracking (15ms = 66 FPS)
```

---

## 7. PERFORMANCE

### 7.1 Comparación Detallada

| Operación | CPU (i7) | GPU (RTX 2060) | GPU (Jetson Orin) |
|-----------|----------|----------------|-------------------|
| Upload query | - | 0.1 ms | 0.05 ms |
| Match 1 KF | 5 ms | 0.8 ms | 2.0 ms |
| Match 500 KF | 2500 ms | 8 ms | 25 ms |
| Download results | - | 0.2 ms | 0.1 ms |
| Ratio test (CPU) | 5 ms | 5 ms | 8 ms |
| **Total 500 KF** | **2505 ms** | **13.3 ms** | **33.2 ms** |

### 7.2 Memory Footprint

```
Component               | Size
------------------------|--------
Descriptor database     | 32 MB (500 KF * 2000 * 32)
Count array             | 2 KB
Match distance buffer   | 4 MB (temp)
Query descriptors       | 64 KB (temp)
------------------------|--------
Total permanent         | ~32 MB
Total peak              | ~40 MB
```

### 7.3 Scalability

```
Keyframes vs Time (GPU):

Time │
     │
 20ms│                              ●
     │                        ●
 15ms│                  ●
     │            ●
 10ms│      ●
     │  ●
  5ms│●
     └─────────────────────────────────
        100  200  300  400  500  600  Keyframes

Casi lineal con número de keyframes
Pendiente mucho menor que CPU (que es O(n) secuencial)
```

---

## 8. CHECKLIST DE PREPARACIÓN

### Conceptos que debes dominar:
- [ ] GPU memory hierarchy (global, shared, registers)
- [ ] Kernel launch overhead y amortización
- [ ] Brute force matching en GPU
- [ ] Memory bandwidth vs compute bound
- [ ] Sincronización GPU-CPU con streams
- [ ] cv::cuda::GpuMat y cv::cuda::Stream

### Código que debes poder escribir:
```cpp
// GPU matcher setup
auto matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

// Async matching
cv::cuda::Stream stream;
matcher->knnMatchAsync(query_gpu, train_gpu, matches_gpu, 2,
                       cv::noArray(), stream);
// ... do other work ...
stream.waitForCompletion();

// Download results
std::vector<std::vector<cv::DMatch>> matches;
matcher->knnMatchConvert(matches_gpu, matches);
```

### Números que debes conocer:
- Descriptor size (ORB): **32 bytes**
- Max keyframes típico: **500**
- GPU memory for LC: **~50 MB**
- Speedup 500 KF: **31x** (RTX), **75x** (Jetson vs CPU)
- Kernel launch overhead: **~5 μs**

---

**Generado:** 2025-01-28
**Proyecto:** aria-slam
**Status:** Diseño (no implementado aún)
