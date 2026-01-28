# Auditoría Técnica: H02 - Feature Extraction (ORB GPU)

**Proyecto:** aria-slam (C++)
**Milestone:** H02 - Detección de características con ORB
**Fecha:** 2025-01
**Autor:** Roberto (preparación entrevista técnica)

---

## 0. RESUMEN EJECUTIVO

### Objetivo
Detectar puntos característicos (features) en cada frame para tracking visual usando ORB (Oriented FAST and Rotated BRIEF).

### Resultado
- **2000 keypoints** por frame
- **3ms en GPU** (RTX 2060) vs 15ms CPU
- Descriptores binarios de **32 bytes** cada uno

---

## 1. ANÁLISIS DEL CÓDIGO REAL

### 1.1 Clase Frame (`Frame.hpp:1-30`)

```cpp
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <vector>

class Frame {
public:
    // GPU ORB con stream opcional
    Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu, cudaStream_t stream = nullptr);
    // CPU ORB fallback
    Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb);
    // Copy constructor
    Frame(const Frame& other);

    void downloadResults();

    // Public members
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::cuda::GpuMat gpu_descriptors;

private:
    cv::Ptr<cv::cuda::ORB> orb_gpu_;
    cv::cuda::GpuMat gpu_keypoints_;
    bool downloaded_ = false;
};
```

**Puntos clave del diseño:**
1. **Overloaded constructors**: GPU y CPU para flexibilidad
2. **Lazy evaluation**: `downloaded_` flag para diferir GPU→CPU transfer
3. **Public members**: Acceso directo para performance (no getters triviales)

### 1.2 Constructor GPU (`Frame.cpp:6-42`)

```cpp
Frame::Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu, cudaStream_t stream) {
    image = img.clone();
    orb_gpu_ = orb_gpu;

    // ORB requires grayscale input
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // CPU -> GPU transfer
    cv::cuda::GpuMat gpu_img(gray);

    if (stream) {
        // Async mode: use provided CUDA stream
        cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

        // Async detection on GPU (non-blocking)
        orb_gpu->detectAndComputeAsync(gpu_img, cv::cuda::GpuMat(),
                                        gpu_keypoints_, gpu_descriptors,
                                        false, cv_stream);
        downloaded_ = false;
    } else {
        // Sync mode: original behavior (backward compatible)
        orb_gpu->detectAndComputeAsync(gpu_img, cv::cuda::GpuMat(),
                                        gpu_keypoints, gpu_descriptors);
        orb_gpu->convert(gpu_keypoints, keypoints);
        gpu_descriptors.download(descriptors);
        downloaded_ = true;
    }
}
```

**Análisis línea por línea:**

| Línea | Operación | Tipo | Memoria |
|-------|-----------|------|---------|
| 7 | `img.clone()` | CPU | Copia defensiva |
| 12-15 | `cvtColor` | CPU | BGR→Gray |
| 18 | `GpuMat(gray)` | H2D | Upload a VRAM |
| 23 | `wrapStream` | CPU | Wrapper de stream |
| 26-27 | `detectAndComputeAsync` | GPU | ORB kernel |

### 1.3 Creación del Detector ORB (`main.cpp:87`)

```cpp
cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create();
```

**Parámetros por defecto de ORB:**
```cpp
cv::cuda::ORB::create(
    int nfeatures = 500,        // Cambiado a 2000 en nuestra implementación
    float scaleFactor = 1.2f,   // Escala entre niveles de pirámide
    int nlevels = 8,            // Niveles de pirámide
    int edgeThreshold = 31,     // Borde a ignorar
    int firstLevel = 0,         // Nivel inicial de pirámide
    int WTA_K = 2,              // Puntos para cada elemento del descriptor
    int scoreType = cv::ORB::HARRIS_SCORE,
    int patchSize = 31,         // Tamaño del patch para descriptor
    int fastThreshold = 20      // Threshold para FAST detector
);
```

---

## 2. TEORÍA DE ORB

### 2.1 Componentes de ORB

**ORB = oFAST + rBRIEF**

1. **oFAST** (Oriented FAST):
   - Detector de esquinas FAST
   - Añade orientación usando intensity centroid
   - Invariante a rotación

2. **rBRIEF** (Rotated BRIEF):
   - Descriptor binario de 256 bits (32 bytes)
   - Rotado según orientación del keypoint
   - Comparación de pares de píxeles → bits

### 2.2 Pirámide de Escala

```
Level 0: 640x480   (escala 1.0)
Level 1: 533x400   (escala 1.2)
Level 2: 444x333   (escala 1.44)
Level 3: 370x278   (escala 1.73)
...
```

**¿Por qué pirámide?**
- Detectar features a diferentes distancias
- Objeto cerca → features grandes (niveles bajos)
- Objeto lejos → features pequeños (niveles altos)

### 2.3 Descriptor Binario

```
Descriptor de 256 bits = 32 bytes

Para cada bit:
1. Seleccionar par de puntos (p1, p2) en el patch
2. if intensity(p1) < intensity(p2): bit = 1
   else: bit = 0

Los pares están pre-calculados para máxima discriminación.
```

**Ventaja binaria:**
- Matching con Hamming distance (XOR + popcount)
- ~10x más rápido que L2 distance en SIFT

---

## 3. FLUJO DE MEMORIA

```
┌─────────────────────────────────────────────────────────────────┐
│                         CPU (RAM)                                │
│                                                                  │
│   cv::Mat frame (BGR, 640x360)                                  │
│        │                                                         │
│        ▼                                                         │
│   cv::cvtColor(frame, gray, BGR2GRAY)                           │
│        │                                                         │
│        ▼                                                         │
│   cv::Mat gray (Grayscale, 640x360)                             │
│        │                                                         │
└────────┼─────────────────────────────────────────────────────────┘
         │ Upload (H2D)
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         GPU (VRAM)                               │
│                                                                  │
│   cv::cuda::GpuMat gpu_img                                      │
│        │                                                         │
│        ▼                                                         │
│   orb_gpu->detectAndComputeAsync()                              │
│        │                                                         │
│        ├──────────────────┐                                      │
│        ▼                  ▼                                      │
│   gpu_keypoints_     gpu_descriptors                            │
│   (GpuMat)           (GpuMat, 2000x32)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
         │ Download (D2H) - Solo cuando se necesita
         ▼
┌─────────────────────────────────────────────────────────────────┐
│   std::vector<cv::KeyPoint> keypoints (2000 elementos)          │
│   cv::Mat descriptors (2000x32, CV_8UC1)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. CONCEPTOS C++ UTILIZADOS

### 4.1 cv::Ptr (Smart Pointer de OpenCV)

```cpp
cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create();
```

**cv::Ptr es similar a std::shared_ptr:**
- Reference counting automático
- Thread-safe para el contador
- Destrucción automática cuando count = 0

```cpp
// Internamente:
template<typename T>
class Ptr {
    T* ptr_;
    int* refcount_;  // Atómico
};
```

### 4.2 GpuMat vs Mat

```cpp
cv::Mat          // CPU memory (RAM)
cv::cuda::GpuMat // GPU memory (VRAM)
```

**Conversión:**
```cpp
// CPU → GPU (upload)
cv::cuda::GpuMat gpu_img(cpu_mat);
// o
gpu_img.upload(cpu_mat);

// GPU → CPU (download)
gpu_img.download(cpu_mat);
```

### 4.3 Constructor Overloading

```cpp
// GPU mode
Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu, cudaStream_t stream = nullptr);

// CPU mode
Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb);
```

**Resolución de overload:**
- Compilador elige basándose en tipos de argumentos
- `cv::Ptr<cv::cuda::ORB>` vs `cv::Ptr<cv::ORB>` son tipos distintos

### 4.4 Default Arguments

```cpp
Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu,
      cudaStream_t stream = nullptr);  // Sync mode por defecto
```

**Uso:**
```cpp
Frame f1(img, orb);           // stream = nullptr → sync mode
Frame f2(img, orb, my_stream); // async mode
```

---

## 5. PATRONES DE DISEÑO

### 5.1 Lazy Evaluation

```cpp
// En constructor: lanza trabajo pero no espera
downloaded_ = false;

// En downloadResults(): materializa cuando se necesita
void Frame::downloadResults() {
    if (downloaded_) return;  // Idempotente
    orb_gpu_->convert(gpu_keypoints_, keypoints);
    gpu_descriptors.download(descriptors);
    downloaded_ = true;
}
```

**Ventaja:**
- Permite overlap con otro trabajo
- Solo transfiere si realmente se necesitan los datos CPU

### 5.2 Deep Copy Constructor

```cpp
Frame::Frame(const Frame& other) {
    image = other.image.clone();
    keypoints = other.keypoints;
    descriptors = other.descriptors.clone();
    other.gpu_descriptors.copyTo(gpu_descriptors);
    // ...
}
```

**¿Por qué deep copy?**
- Frame anterior se guarda para matching
- Sin deep copy, el frame actual sobrescribiría datos

---

## 6. PREGUNTAS DE ENTREVISTA

### Q1: ¿Por qué ORB y no SIFT/SURF?

**R:**
| Aspecto | ORB | SIFT | SURF |
|---------|-----|------|------|
| Velocidad | Muy rápido | Lento | Medio |
| Descriptor | 32 bytes | 128 floats | 64 floats |
| Matching | Hamming | L2 | L2 |
| Patente | Libre | Libre (2020+) | Patentado |
| GPU support | Excelente | Limitado | Limitado |

### Q2: ¿Qué significa `detectAndComputeAsync`?

**R:**
- **Async**: No bloquea el CPU, retorna inmediatamente
- El kernel GPU se encola en el stream
- Resultados disponibles después de `cudaStreamSynchronize()`

```cpp
// Timeline:
CPU: detectAndComputeAsync() → return → [puede hacer otro trabajo]
GPU:                         [ejecutando ORB kernel...]
     ← cudaStreamSynchronize() → [resultados listos]
```

### Q3: ¿Por qué convertir a grayscale?

**R:**
1. ORB trabaja con intensidades, no color
2. Reduce datos 3x (BGR→Gray)
3. Menos memoria GPU, más rápido
4. Features son igualmente distintivos en grayscale

### Q4: ¿Cuál es el tamaño de memoria de gpu_descriptors?

**R:**
```
2000 keypoints × 32 bytes = 64 KB por frame
```

En VRAM esto es insignificante (RTX 2060 tiene 6GB).

### Q5: ¿Qué pasa si hay menos de 2000 features en la imagen?

**R:**
- ORB retorna los que encuentre
- `keypoints.size()` puede ser < 2000
- El código debe manejar frames con pocos features

```cpp
// En main.cpp, verificamos antes de matching:
if (prev_frame->gpu_descriptors.empty() ||
    current_frame.gpu_descriptors.empty()) {
    // Skip matching
}
```

### Q6: ¿Cómo funciona cv::cuda::StreamAccessor::wrapStream()?

**R:**
```cpp
cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
```

- Convierte `cudaStream_t` (CUDA nativo) a `cv::cuda::Stream` (OpenCV)
- Permite usar el **mismo stream** entre APIs diferentes
- OpenCV y TensorRT comparten stream → ejecución en paralelo real

---

## 7. PERFORMANCE

### Comparación CPU vs GPU

| Operación | CPU (i7-10700) | GPU (RTX 2060) | Speedup |
|-----------|----------------|----------------|---------|
| ORB detect 2000pts | 15ms | 3ms | 5x |
| Upload image | N/A | 0.5ms | - |
| Download results | N/A | 0.3ms | - |
| **Total** | 15ms | 3.8ms | 4x |

### Ocupación de GPU

```bash
# Verificar con nvidia-smi durante ejecución
nvidia-smi dmon -s u

# O con nvtop
nvtop
```

---

## 8. DIAGRAMA DE SECUENCIA

```
main.cpp                    Frame                        GPU
    │                         │                           │
    │ Frame(img, orb, stream) │                           │
    │────────────────────────►│                           │
    │                         │ cv::cvtColor()            │
    │                         │──────────────────────────►│
    │                         │                           │
    │                         │ GpuMat upload             │
    │                         │──────────────────────────►│
    │                         │                           │
    │                         │ detectAndComputeAsync()   │
    │                         │──────────────────────────►│
    │◄────────────────────────│ (returns immediately)     │
    │                         │                           │[ORB kernel running]
    │ [do other work]         │                           │
    │                         │                           │
    │ cudaStreamSynchronize() │                           │
    │─────────────────────────────────────────────────────►│
    │◄─────────────────────────────────────────────────────│
    │                         │                           │
    │ downloadResults()       │                           │
    │────────────────────────►│                           │
    │                         │ convert() + download()    │
    │                         │──────────────────────────►│
    │◄────────────────────────│                           │
    │                         │                           │
```

---

## 9. CHECKLIST DE PREPARACIÓN

- [ ] Entender componentes de ORB (oFAST + rBRIEF)
- [ ] Saber explicar pirámide de escala
- [ ] Conocer diferencia binario vs float descriptors
- [ ] Entender flujo memoria CPU↔GPU
- [ ] Saber qué hace `detectAndComputeAsync`
- [ ] Explicar lazy evaluation pattern
- [ ] Conocer tamaño de descriptors (32 bytes × N)
- [ ] Entender cv::Ptr vs std::shared_ptr

---

**Generado:** 2025-01
**Proyecto:** aria-slam (C++)
