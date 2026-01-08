# Auditoría Técnica: CUDA Streams - C++ vs Python

**Proyectos comparados:**
- **aria-slam** (C++) - Visual-Inertial SLAM
- **aria-nav** (Python) - Navigation System for Visually Impaired

**Milestone:** H11 - Paralelización con CUDA Streams
**Fecha:** 2025-01-05
**Autor:** Roberto (preparación entrevista técnica)

---

## 0. COMPARACIÓN EJECUTIVA C++ vs PYTHON

### Tabla Comparativa Principal

| Aspecto | C++ (aria-slam) | Python (aria-nav) |
|---------|-----------------|-------------------|
| **Framework GPU** | CUDA Runtime API + TensorRT | PyTorch CUDA |
| **Operaciones paralelas** | ORB + YOLO | Depth + YOLO |
| **Tipo de stream** | `cudaStream_t` (nativo) | `torch.cuda.Stream` (wrapper) |
| **Creación de streams** | `cudaStreamCreate(&stream)` | `torch.cuda.Stream()` |
| **Sincronización** | `cudaStreamSynchronize(s)` | `torch.cuda.synchronize()` |
| **Scope de ejecución** | Manual (launch → sync) | Context manager (`with`) |
| **Cleanup** | `cudaStreamDestroy()` (manual) | Garbage collector (automático) |
| **Interop OpenCV** | `cv::cuda::StreamAccessor` | N/A (usa PyTorch nativo) |
| **Nivel de abstracción** | Bajo (control total) | Alto (más seguro) |
| **Error handling** | Códigos de error CUDA | Excepciones Python |
| **Memory management** | RAII + manual | GC automático |

### Código Side-by-Side

#### Creación de Streams
```cpp
// C++ (aria-slam/src/main.cpp:74-77)
cudaStream_t stream_orb, stream_yolo;
cudaStreamCreate(&stream_orb);
cudaStreamCreate(&stream_yolo);
```

```python
# Python (aria-nav/src/core/navigation/navigation_pipeline.py:138-139)
self.yolo_stream = torch.cuda.Stream()
self.depth_stream = torch.cuda.Stream()
```

#### Lanzamiento Paralelo
```cpp
// C++ (aria-slam/src/main.cpp:103-110)
// Launch async - NO blocking
Frame current_frame(frame, orb, stream_orb);
yolo->detectAsync(frame, stream_yolo);
```

```python
# Python (aria-nav - lines 215-240)
# Context managers - scope-based
with torch.cuda.stream(self.depth_stream):
    depth_prediction = self.depth_estimator.estimate_depth(frame)

with torch.cuda.stream(self.yolo_stream):
    detections = self.yolo_processor.process_frame(frame)
```

#### Sincronización
```cpp
// C++ - Explícito por stream
cudaStreamSynchronize(stream_orb);
cudaStreamSynchronize(stream_yolo);
```

```python
# Python - Global (todos los streams)
torch.cuda.synchronize()
```

#### Cleanup
```cpp
// C++ - Manual RAII
cudaStreamDestroy(stream_orb);
cudaStreamDestroy(stream_yolo);
```

```python
# Python - Automático (GC)
# No cleanup necesario, el GC lo maneja
del self.yolo_stream  # Opcional
```

---

### Diagrama de Flujo Comparativo

```
                    C++ (aria-slam)                     Python (aria-nav)
                    ===============                     =================

Frame input         Frame input
     │                   │
     ├────────┬──────────┤
     │        │          │
     ▼        ▼          ▼
[stream_orb] [stream_yolo]   with torch.cuda.stream(depth_stream):
     │        │               │
 ORB GPU   YOLO GPU          Depth GPU
     │        │               │
     │        │          with torch.cuda.stream(yolo_stream):
     │        │               │
     │        │           YOLO GPU
     │        │               │
     ▼        ▼               ▼
cudaStreamSync(orb)      torch.cuda.synchronize()
cudaStreamSync(yolo)          │
     │        │               │
     ▼        ▼               ▼
downloadResults()        Results ready
getDetections()               │
     │                        │
     ▼                        ▼
  Results                  Results
```

---

### Ventajas y Desventajas

#### C++ (aria-slam)
**Ventajas:**
- Control granular de sincronización (por stream)
- Menor overhead (no GIL, no interpretador)
- Interoperabilidad directa con TensorRT, OpenCV CUDA
- Debugging con `cuda-memcheck`, `nvprof`

**Desventajas:**
- Más código boilerplate
- Memory leaks si olvidas cleanup
- Más difícil de mantener
- Compilación lenta

#### Python (aria-nav)
**Ventajas:**
- Context managers (`with`) previenen resource leaks
- Código más limpio y legible
- Iteración rápida (no compilación)
- Ecosystem rico (PyTorch, transformers)

**Desventajas:**
- GIL limita paralelismo CPU
- Overhead del interpretador
- `torch.cuda.synchronize()` es global (menos control)
- Debugging GPU más difícil

---

## 1. Resumen Ejecutivo

### Objetivo
Ejecutar **ORB feature detection** y **YOLO object detection** en paralelo usando CUDA streams separados para mejorar el throughput del pipeline SLAM.

### Resultados
| Métrica | Antes (Secuencial) | Después (Paralelo) | Mejora |
|---------|-------------------|-------------------|--------|
| Latencia | 13.7 ms/frame | 12.5 ms/frame | -9% |
| FPS | ~73 | ~80 | +10% |
| Speedup | 1.0x | 1.1x | - |

### Arquitectura de Paralelismo
```
Frame N (entrada)
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
[Stream ORB]                         [Stream YOLO]
    │                                     │
    │ detectAndComputeAsync()             │ enqueueV3()
    │ (cv::cuda::ORB)                     │ (TensorRT)
    │                                     │
    ▼                                     ▼
GPU Keypoints                        GPU Output Tensor
GPU Descriptors                      (84 × 8400 floats)
    │                                     │
    └─────────────────────────────────────┘
                      │
                      ▼
            cudaStreamSynchronize() × 2
                      │
                      ▼
            downloadResults() + getDetections()
                      │
                      ▼
              Feature Matching + Pose Estimation
```

---

## 2. Análisis de Independencia de Datos

### ¿Por qué ORB y YOLO pueden ejecutarse en paralelo?

| Aspecto | ORB | YOLO | Independencia |
|---------|-----|------|---------------|
| **Input** | Grayscale de frame | BGR frame original | ✅ Copias separadas |
| **GPU Buffers** | `gpu_keypoints_`, `gpu_descriptors` | `buffers_[0]`, `buffers_[1]` | ✅ Memoria disjunta |
| **Output** | `keypoints`, `descriptors` | `detections` | ✅ Sin overlap |
| **Dependencias** | Ninguna con YOLO | Ninguna con ORB | ✅ Independientes |

**Conclusión:** No hay data hazards (RAW, WAR, WAW) entre las operaciones.

---

## 3. Análisis Detallado del Código

### 3.1 Creación de CUDA Streams (`main.cpp:74-77`)

```cpp
cudaStream_t stream_orb, stream_yolo;
cudaStreamCreate(&stream_orb);
cudaStreamCreate(&stream_yolo);
```

**Concepto:** Un CUDA stream es una cola de operaciones GPU que se ejecutan en orden FIFO. Operaciones en **streams diferentes** pueden ejecutarse en paralelo (concurrent kernels).

**API CUDA:**
- `cudaStreamCreate()` - Crea un stream no-blocking
- El default stream (0 o NULL) serializa todas las operaciones
- Streams custom permiten overlap de kernels y transfers

### 3.2 Lanzamiento Paralelo (`main.cpp:103-110`)

```cpp
// Stream 1: ORB feature extraction (async)
Frame current_frame(frame, orb, stream_orb);

// Stream 2: YOLO object detection (async)
if (yolo) {
    yolo->detectAsync(frame, stream_yolo);
}
```

**Timeline de ejecución:**
```
Tiempo →
Stream ORB:  [preprocess]──[detectAndComputeAsync]──────────────────────►
Stream YOLO: [preprocess]──[cudaMemcpyAsync]──[enqueueV3]──[cudaMemcpyAsync]►
                                              ↑
                                     TensorRT inference
```

### 3.3 Frame.cpp - Async ORB Detection

```cpp
Frame::Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu, cudaStream_t stream) {
    // ...
    if (stream) {
        // Async mode: use provided CUDA stream
        cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

        orb_gpu->detectAndComputeAsync(gpu_img, cv::cuda::GpuMat(),
                                        gpu_keypoints_, gpu_descriptors,
                                        false, cv_stream);
        downloaded_ = false;  // Lazy download
    } else {
        // Sync mode (backward compatible)
        // ...
        downloaded_ = true;
    }
}
```

**Puntos clave:**
1. **`cv::cuda::StreamAccessor::wrapStream()`** - Convierte `cudaStream_t` a `cv::cuda::Stream`
2. **`detectAndComputeAsync()`** - Versión no-blocking de detección ORB
3. **`downloaded_ = false`** - Patrón lazy evaluation para diferir GPU→CPU transfer

### 3.4 TRTInference.cpp - Async YOLO Detection

```cpp
void TRTInference::detectAsync(const cv::Mat& image, cudaStream_t stream) {
    // Store scale factors for later postprocessing
    last_scale_x_ = (float)image.cols / input_w_;
    last_scale_y_ = (float)image.rows / input_h_;

    // Preprocess (CPU work + async copy to GPU)
    preprocess(image, (float*)buffers_[input_idx_], stream);

    // Set tensor addresses
    context_->setTensorAddress(engine_->getIOTensorName(input_idx_), buffers_[input_idx_]);
    context_->setTensorAddress(engine_->getIOTensorName(output_idx_), buffers_[output_idx_]);

    // Run inference asynchronously on provided stream
    context_->enqueueV3(stream);

    // Prepare output buffer and async copy results
    output_buffer_.resize(output_size_);
    cudaMemcpyAsync(output_buffer_.data(), buffers_[output_idx_],
                    output_size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    // No synchronization here - caller must sync
}
```

**Desglose de operaciones:**

| Operación | Tipo | Blocking? | Stream |
|-----------|------|-----------|--------|
| `preprocess()` | CPU + H2D copy | No (async) | `stream` |
| `setTensorAddress()` | CPU | Sí (trivial) | N/A |
| `enqueueV3()` | GPU kernel | No | `stream` |
| `cudaMemcpyAsync()` | D2H copy | No | `stream` |

### 3.5 Sincronización (`main.cpp:112-114`)

```cpp
cudaStreamSynchronize(stream_orb);
cudaStreamSynchronize(stream_yolo);
```

**¿Por qué dos llamadas separadas?**
- Cada stream tiene su propia cola de trabajo
- `cudaStreamSynchronize(s)` bloquea hasta que **todas** las operaciones en stream `s` completen
- Alternativa: `cudaDeviceSynchronize()` espera **todos** los streams (menos eficiente si hay otros trabajos GPU)

### 3.6 Descarga de Resultados (`main.cpp:116-123`)

```cpp
current_frame.downloadResults();

std::vector<Detection> detections;
if (yolo) {
    detections = yolo->getDetections(0.5f, 0.45f);
}
```

**Frame::downloadResults():**
```cpp
void Frame::downloadResults() {
    if (downloaded_) return;  // Idempotente

    if (orb_gpu_ && !gpu_keypoints_.empty()) {
        orb_gpu_->convert(gpu_keypoints_, keypoints);  // GPU format → std::vector
    }
    if (!gpu_descriptors.empty()) {
        gpu_descriptors.download(descriptors);  // GpuMat → Mat
    }
    downloaded_ = true;
}
```

**TRTInference::getDetections():**
```cpp
std::vector<Detection> TRTInference::getDetections(float conf_thresh, float nms_thresh) {
    return postprocess(output_buffer_.data(), output_size_,
                       conf_thresh, nms_thresh,
                       last_scale_x_, last_scale_y_);
}
```

**Nota:** `postprocess()` es **CPU-only** (NMS, threshold filtering). Los datos ya están en RAM después del sync.

### 3.7 Cleanup RAII (`main.cpp:210-212`)

```cpp
cudaStreamDestroy(stream_orb);
cudaStreamDestroy(stream_yolo);
```

**Buenas prácticas:**
- Siempre destruir streams creados manualmente
- TRTInference destructor también limpia su stream interno:
  ```cpp
  TRTInference::~TRTInference() {
      cudaStreamDestroy(stream_);
      cudaFree(buffers_[0]);
      cudaFree(buffers_[1]);
      delete context_;
      delete engine_;
      delete runtime_;
  }
  ```

---

## 4. Conceptos C++ Utilizados

### 4.1 Smart Pointers
```cpp
std::unique_ptr<TRTInference> yolo;
std::unique_ptr<Frame> prev_frame;
```
- **Ownership semántico:** Un solo dueño, destrucción automática
- **Move semantics:** `prev_frame = std::make_unique<Frame>(current_frame);`

### 4.2 RAII (Resource Acquisition Is Initialization)
```cpp
// TRTInference constructor: adquiere recursos
TRTInference::TRTInference(...) {
    cudaMalloc(&buffers_[...], ...);
    cudaStreamCreate(&stream_);
}

// Destructor: libera recursos
TRTInference::~TRTInference() {
    cudaStreamDestroy(stream_);
    cudaFree(buffers_[...]);
}
```

### 4.3 Constructor Overloading
```cpp
// GPU async mode
Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu, cudaStream_t stream = nullptr);

// CPU fallback
Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb);

// Copy constructor
Frame(const Frame& other);
```

### 4.4 Default Arguments para Backward Compatibility
```cpp
Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu,
      cudaStream_t stream = nullptr);  // nullptr = sync mode
```

---

## 5. Patrones de Diseño

### 5.1 Lazy Evaluation
```cpp
// En constructor: solo lanza trabajo
downloaded_ = false;

// En downloadResults(): materializa cuando se necesita
void Frame::downloadResults() {
    if (downloaded_) return;
    // ... transfer data
    downloaded_ = true;
}
```

### 5.2 Two-Phase Initialization (Async Pattern)
```cpp
// Phase 1: Launch async work
yolo->detectAsync(frame, stream_yolo);

// Phase 2: Synchronize + retrieve
cudaStreamSynchronize(stream_yolo);
auto detections = yolo->getDetections();
```

### 5.3 Backward Compatibility via Overloading
```cpp
// Old API (sync, still works)
auto dets = yolo->detect(frame);

// New API (async)
yolo->detectAsync(frame, stream);
cudaStreamSynchronize(stream);
auto dets = yolo->getDetections();
```

---

## 6. Posibles Preguntas de Entrevista

### Q1: ¿Por qué usar CUDA streams en lugar de threads CPU?
**R:** Los CUDA streams permiten **overlap de operaciones GPU** (kernels, memory transfers) sin overhead de context switching. Los threads CPU no pueden paralelizar trabajo GPU de esta manera.

### Q2: ¿Qué pasa si no sincronizo antes de leer los resultados?
**R:** **Race condition.** Los datos en `output_buffer_` podrían estar incompletos o corruptos. El `cudaMemcpyAsync` podría no haber terminado.

### Q3: ¿Por qué guardar `last_scale_x_` como miembro en lugar de recalcular?
**R:** Porque en `detectAsync()` tenemos acceso a `image.cols/rows`, pero en `getDetections()` ya no tenemos la imagen. Es un **patrón de state capture** para operaciones async.

### Q4: ¿Cuál es la limitación del speedup observado (1.1x)?
**R:**
1. **Preprocessing es CPU-bound:** `cv::resize`, `cv::cvtColor`, `memcpy` son secuenciales
2. **La GPU tiene un solo motor de copia:** H2D y D2H comparten el DMA engine
3. **El trabajo GPU es relativamente corto:** ORB ~10ms, YOLO ~5ms
4. **Ley de Amdahl:** Si 70% del tiempo es paralelo, speedup máximo ≈ 1.4x

### Q5: ¿Cómo mejorarías el speedup?
**R:**
1. **Pinned memory:** `cudaMallocHost()` para evitar staging buffer en transfers
2. **Async preprocessing:** Usar CUDA kernels para resize/colorspace en GPU
3. **Triple buffering:** Procesar frame N mientras N-1 está en postprocess y N+1 se carga
4. **Multi-GPU:** Distribuir ORB y YOLO en GPUs separadas

### Q6: ¿Por qué `cv::cuda::StreamAccessor::wrapStream()` en lugar de crear un `cv::cuda::Stream` directamente?
**R:** OpenCV CUDA usa su propia abstracción `cv::cuda::Stream`. Para interoperar con APIs nativas CUDA (como TensorRT que usa `cudaStream_t`), necesitamos el wrapper. Esto permite usar **el mismo stream** entre OpenCV y CUDA puro.

### Q7: Explica el flujo de memoria en YOLO inference.
**R:**
```
1. CPU: preprocess() crea input_data (vector<float>)
2. H2D: cudaMemcpyAsync() → buffers_[input_idx_] (GPU)
3. GPU: enqueueV3() ejecuta inferencia → buffers_[output_idx_]
4. D2H: cudaMemcpyAsync() → output_buffer_ (CPU)
5. CPU: postprocess() filtra y aplica NMS
```

### Q8: ¿Qué significa "enqueueV3" en TensorRT?
**R:** Es la API de ejecución de TensorRT 8.x+. El "V3" indica la versión del API que usa **tensor addresses** en lugar de bindings por índice. Es más flexible para modelos con shapes dinámicos.

---

## 7. Diagrama de Secuencia Completo

```
main()                  Frame                   TRTInference              GPU
  │                       │                          │                      │
  │ Frame(img,orb,stream) │                          │                      │
  │──────────────────────►│                          │                      │
  │                       │ detectAndComputeAsync()  │                      │
  │                       │─────────────────────────────────────────────────►│
  │                       │                          │                      │
  │ detectAsync(img,s)    │                          │                      │
  │──────────────────────────────────────────────────►│                      │
  │                       │                          │ preprocess()+enqueue │
  │                       │                          │─────────────────────►│
  │                       │                          │                      │
  │ cudaStreamSync(s_orb) │                          │                      │
  │────────────────────────────────────────────────────────────────────────►│
  │◄───────────────────────────────────────────────────────────────────────│
  │                       │                          │                      │
  │ cudaStreamSync(s_yolo)│                          │                      │
  │────────────────────────────────────────────────────────────────────────►│
  │◄───────────────────────────────────────────────────────────────────────│
  │                       │                          │                      │
  │ downloadResults()     │                          │                      │
  │──────────────────────►│                          │                      │
  │                       │ convert() + download()   │                      │
  │                       │─────────────────────────────────────────────────►│
  │◄──────────────────────│                          │                      │
  │                       │                          │                      │
  │ getDetections()       │                          │                      │
  │──────────────────────────────────────────────────►│                      │
  │                       │                          │ postprocess() [CPU]  │
  │◄─────────────────────────────────────────────────│                      │
  │                       │                          │                      │
```

---

## 8. Checklist de Preparación

- [ ] Entender qué es un CUDA stream y cómo permite paralelismo
- [ ] Saber explicar el patrón async launch + sync + retrieve
- [ ] Conocer la diferencia entre `cudaStreamSynchronize()` y `cudaDeviceSynchronize()`
- [ ] Entender por qué ORB y YOLO son independientes (no hay data hazards)
- [ ] Poder explicar el flujo de memoria H2D → GPU → D2H
- [ ] Saber qué es RAII y cómo se aplica en el destructor
- [ ] Entender backward compatibility con default arguments
- [ ] Conocer las limitaciones del speedup (Ley de Amdahl, CPU bottlenecks)

---

## 9. PREGUNTAS DE ENTREVISTA: C++ vs Python

### Q1: ¿Por qué elegiste C++ para aria-slam y Python para aria-nav?

**R:**
- **aria-slam (C++)**: Requiere latencia mínima para SLAM en tiempo real. Cada milisegundo cuenta para tracking preciso. Además, OpenCV CUDA y TensorRT tienen APIs nativas en C++.
- **aria-nav (Python)**: Sistema de navegación donde la latencia de ~50ms es aceptable. Priorizo velocidad de desarrollo y el ecosistema de PyTorch/transformers para modelos de depth estimation.

### Q2: ¿Cuál es la diferencia fundamental en el modelo de ejecución de streams?

**R:**
```cpp
// C++: Lanzamiento explícito, ejecución concurrente real
Frame f(img, orb, stream_orb);      // Returns immediately
yolo->detectAsync(img, stream_yolo); // Returns immediately
// Ambos ejecutan EN PARALELO desde aquí
```

```python
# Python: Context managers, ejecución secuencial de SUBMIT
with torch.cuda.stream(self.depth_stream):
    depth = estimator.estimate(frame)  # Submit to stream
# El with sale, pero el trabajo GPU sigue

with torch.cuda.stream(self.yolo_stream):
    dets = yolo.process(frame)  # Submit to stream
# Ahora AMBOS trabajan en paralelo en GPU
```

**Diferencia clave:** En C++ el paralelismo es inmediato. En Python, el paralelismo ocurre después de que el código Python envía los comandos (porque Python es single-threaded por GIL).

### Q3: ¿Por qué C++ usa `cudaStreamSynchronize()` por stream y Python usa `torch.cuda.synchronize()` global?

**R:**
- **C++**: Tengo control granular. Puedo sincronizar solo ORB si necesito sus resultados antes que YOLO.
- **Python**: `torch.cuda.synchronize()` es más simple pero sincroniza TODOS los streams. PyTorch también tiene `stream.synchronize()` pero es menos común.

```cpp
// C++: Puedo hacer esto
cudaStreamSynchronize(stream_orb);
// Usar resultados de ORB mientras YOLO sigue trabajando
process_orb_results();
cudaStreamSynchronize(stream_yolo);
```

```python
# Python: Esto es posible pero no idiomático
self.depth_stream.synchronize()  # Solo depth
# Usar depth mientras YOLO sigue
torch.cuda.synchronize()  # Esperar todo
```

### Q4: ¿Cómo manejas errores en cada lenguaje?

**R:**
```cpp
// C++: Códigos de error CUDA (fácil olvidar checkear)
cudaError_t err = cudaStreamSynchronize(stream);
if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
}

// Mejor práctica: macro CHECK_CUDA
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); \
}
```

```python
# Python: Excepciones automáticas
try:
    with torch.cuda.stream(self.yolo_stream):
        dets = yolo.process(frame)
    torch.cuda.synchronize()
except RuntimeError as e:
    print(f"CUDA error: {e}")
```

### Q5: ¿Qué pasa con la memoria GPU en cada caso?

**R:**
```cpp
// C++: Manual allocation + RAII cleanup
cudaMalloc(&buffer, size);  // Constructor
// ... uso ...
cudaFree(buffer);  // Destructor (RAII)

// Riesgo: Si olvidas cudaFree = memory leak permanente
```

```python
# Python: PyTorch maneja automáticamente
tensor = torch.zeros(1000, device='cuda')  # Alloc
del tensor  # El GC lo liberará eventualmente
torch.cuda.empty_cache()  # Forzar liberación (opcional)

# Riesgo: GC puede no liberar inmediatamente = OOM
```

### Q6: ¿Cómo harías profiling en cada lenguaje?

**R:**
```cpp
// C++: NVIDIA tools nativos
// nsys profile ./aria_slam
// ncu --set full ./aria_slam

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, stream);
// ... trabajo GPU ...
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

```python
# Python: PyTorch profiler o NVIDIA tools
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA]
) as prof:
    with torch.cuda.stream(stream):
        result = model(input)
print(prof.key_averages().table())

# También: nsys profile python script.py
```

### Q7: Si tuvieras que portar aria-nav de Python a C++, ¿qué sería lo más difícil?

**R:**
1. **Depth Anything V2**: El modelo está en PyTorch. Tendría que exportar a ONNX → TensorRT.
2. **Context managers**: Reemplazar `with torch.cuda.stream()` por pares manuales de launch/sync.
3. **Dynamic typing**: Python dicts → C++ structs/classes con tipos explícitos.
4. **Config system**: El Config YAML de Python → implementación C++ (yaml-cpp o similar).
5. **Testing**: pytest → Google Test, más boilerplate.

### Q8: ¿Cuándo usarías multiprocessing (Python) vs threads (C++)?

**R:**
```python
# Python: multiprocessing para bypass GIL
# aria-nav usa mp.Process para workers GPU separados
from torch import multiprocessing as mp
p = mp.Process(target=gpu_worker, args=(queue,))
p.start()
```

```cpp
// C++: threads directos, sin GIL
std::thread t1([&]{ orb_detection(frame); });
std::thread t2([&]{ yolo_detection(frame); });
t1.join(); t2.join();

// O mejor: CUDA streams sin threads adicionales
```

**Punto clave:** En Python el multiprocessing es necesario para trabajo CPU paralelo pesado. En C++, threads simples funcionan. Pero para trabajo **GPU**, CUDA streams son más eficientes en ambos lenguajes.

### Q9: ¿Cuál es el overhead de PyTorch vs CUDA nativo?

**R:** PyTorch añade ~microsegundos de overhead por operación (dispatch, autograd tracking). Para operaciones largas (inferencia de modelo), es negligible. Para muchas operaciones pequeñas, puede sumar.

```
Operación        | CUDA C++ | PyTorch Python
-----------------|----------|---------------
Kernel launch    | ~5μs     | ~10-20μs
Memory alloc     | ~1ms     | ~1ms + GC overhead
Stream sync      | ~1μs     | ~5μs
Total inference  | ~5ms     | ~5.5ms (10% overhead)
```

### Q10: ¿Cómo garantizas thread-safety en cada caso?

**R:**
```cpp
// C++: CUDA streams son thread-safe por diseño
// Pero cuidado con datos compartidos
std::mutex mtx;
{
    std::lock_guard<std::mutex> lock(mtx);
    shared_data = result;
}
```

```python
# Python: GIL protege estructuras Python automáticamente
# Pero operaciones CUDA no están protegidas por GIL
import threading
lock = threading.Lock()
with lock:
    shared_data = result

# Para multiprocessing: mp.Queue, mp.Value, shared memory
```

---

## 10. Checklist Final para Entrevista

### Conceptos que debes dominar:
- [ ] Qué es un CUDA stream y cómo permite paralelismo
- [ ] Diferencia entre `cudaStreamSynchronize()` y `torch.cuda.synchronize()`
- [ ] Context managers de Python vs launch manual de C++
- [ ] RAII en C++ vs GC en Python para recursos GPU
- [ ] Cuándo usar streams vs multiprocessing vs threads
- [ ] Trade-offs de cada lenguaje para GPU computing

### Código que debes poder escribir de memoria:
```cpp
// C++: Patrón básico de CUDA streams
cudaStream_t s;
cudaStreamCreate(&s);
kernel<<<blocks, threads, 0, s>>>(...);
cudaStreamSynchronize(s);
cudaStreamDestroy(s);
```

```python
# Python: Patrón básico de PyTorch streams
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    result = model(input)
torch.cuda.synchronize()
```

### Números que debes conocer:
- Overhead PyTorch vs CUDA nativo: ~10%
- Latencia típica de kernel launch: 5-20μs
- Latencia de stream sync: 1-5μs
- Speedup típico con streams: 1.1x - 1.5x (depende del workload)

---

**Generado:** 2025-01-05
**Proyectos:** aria-slam (C++), aria-nav (Python)
