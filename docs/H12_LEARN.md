# H12: Aprendiendo Clean Architecture con Ejemplos Reales

## El Problema: Código Acoplado

Mira tu `main.cpp` actual:

```cpp
// main.cpp SABE demasiado:
#include <opencv2/cudafeatures2d.hpp>  // ← Sabe que usas CUDA
#include "Frame.hpp"                    // ← Sabe que Frame usa cv::KeyPoint

int main() {
    // main.cpp CREA el detector específico
    cv::Ptr<cv::cuda::ORB> orb_gpu = cv::cuda::ORB::create(1000);

    // main.cpp SABE cómo construir un Frame con CUDA
    Frame frame(img, orb_gpu, stream);

    // main.cpp USA tipos de OpenCV
    cv::KeyPoint kp = frame.keypoints[0];
    cv::Point2f pt = kp.pt;  // ← Acoplado a OpenCV
}
```

**¿Por qué es malo?**

1. Si cambias ORB por SuperPoint, tienes que modificar `main.cpp`
2. No puedes testear `main.cpp` sin GPU
3. `main.cpp` tiene 500 líneas porque hace todo

---

## La Solución: Separar Responsabilidades

### Paso 1: Define QUÉ necesitas (Interface)

```cpp
// interfaces/IFeatureExtractor.hpp
// NO dice CÓMO extraer features, solo QUÉ espera

class IFeatureExtractor {
public:
    virtual void extract(
        const uint8_t* image,  // ← Datos crudos, no cv::Mat
        int width, int height,
        core::Frame& frame     // ← Tipo del dominio, no OpenCV
    ) = 0;
};
```

**Pregunta clave:** ¿Qué necesita saber `main.cpp` para extraer features?
- Necesita pasar una imagen
- Necesita recibir keypoints y descriptores
- **NO necesita saber** si usas ORB, SIFT, SuperPoint, CPU o GPU

### Paso 2: Define los tipos del dominio (Core)

```cpp
// core/Types.hpp
// Tipos PUROS sin dependencias externas

namespace aria::core {

struct KeyPoint {
    float x, y;      // ← NO es cv::Point2f
    float size;
    float angle;
    float response;
    int octave;
};

struct Frame {
    std::vector<KeyPoint> keypoints;      // ← NO es std::vector<cv::KeyPoint>
    std::vector<uint8_t> descriptors;     // ← NO es cv::Mat
};

}
```

**¿Por qué no usar cv::KeyPoint directamente?**

Porque si mañana quieres usar una librería diferente (ej: SuperPoint en PyTorch), tendrías que cambiar TODO el código que usa `cv::KeyPoint`.

Con tipos propios:
- `OrbCudaExtractor` traduce `cv::KeyPoint` → `core::KeyPoint`
- `SuperPointExtractor` traduce `torch::Tensor` → `core::KeyPoint`
- **El resto del código no cambia**

### Paso 3: Implementa el Adapter

```cpp
// adapters/gpu/OrbCudaExtractor.cpp
// AQUÍ vive OpenCV CUDA, OCULTO del resto del sistema

void OrbCudaExtractor::extract(
    const uint8_t* image_data,
    int width, int height,
    core::Frame& frame
) {
    // === TRADUCCIÓN: Entrada ===
    // El mundo externo nos da uint8_t*
    // OpenCV necesita cv::Mat
    cv::Mat image(height, width, CV_8UC1, (void*)image_data);

    // === TRABAJO INTERNO (OpenCV CUDA) ===
    cv::cuda::GpuMat gpu_img(image);
    orb_->detectAndComputeAsync(gpu_img, ..., gpu_keypoints_, gpu_descriptors_);

    // === TRADUCCIÓN: Salida ===
    // OpenCV produce cv::KeyPoint
    // El dominio espera core::KeyPoint
    std::vector<cv::KeyPoint> cv_kps;
    orb_->convert(gpu_keypoints_, cv_kps);

    for (const auto& kp : cv_kps) {
        frame.keypoints.push_back({
            .x = kp.pt.x,        // cv::Point2f.x → float
            .y = kp.pt.y,
            .size = kp.size,
            .angle = kp.angle,
            .response = kp.response,
            .octave = kp.octave
        });
    }
}
```

**El Adapter hace DOS traducciones:**
1. **Entrada:** `uint8_t*` → `cv::Mat` → `cv::cuda::GpuMat`
2. **Salida:** `cv::KeyPoint` → `core::KeyPoint`

---

## El Resultado: main.cpp Limpio

```cpp
// main.cpp NUEVO
#include "interfaces/IFeatureExtractor.hpp"
#include "factory/PipelineFactory.hpp"

int main() {
    // Factory decide qué implementación usar
    auto extractor = PipelineFactory::createExtractor("gpu");

    // main.cpp NO sabe qué hay dentro de extractor
    core::Frame frame;
    extractor->extract(image_data, 640, 480, frame);

    // main.cpp usa tipos del dominio
    core::KeyPoint kp = frame.keypoints[0];
    float x = kp.x;  // ← NO es cv::Point2f, es float directo
}
```

**¿Qué ganamos?**

| Antes | Después |
|-------|---------|
| main.cpp tiene `#include <opencv2/cuda...>` | main.cpp solo incluye interfaces |
| Cambiar ORB→SuperPoint requiere modificar main.cpp | Solo creas nuevo adapter |
| No puedes testear sin GPU | Creas MockExtractor para tests |
| main.cpp tiene 500 líneas | main.cpp orquesta, adapters trabajan |

---

## Ejercicio Mental: ¿Dónde va cada cosa?

Pregúntate: **"¿Esta línea de código depende de una librería externa?"**

| Código | ¿Dónde va? | ¿Por qué? |
|--------|------------|-----------|
| `cv::cuda::ORB::create()` | `adapters/gpu/` | Depende de OpenCV CUDA |
| `struct KeyPoint { float x, y; }` | `core/` | Solo usa tipos básicos |
| `virtual void extract() = 0` | `interfaces/` | Es un contrato abstracto |
| `extractor->extract(...)` | `main.cpp` / `pipeline/` | Orquesta componentes |
| `kp.pt.x` (cv::Point2f) | `adapters/` | Traduce a `kp.x` del dominio |

---

## La Analogía del Enchufe

```
Tu laptop (dominio)     Adaptador de viaje     Enchufe de pared (tecnología)
      │                       │                        │
      │                       │                        │
   [USB-C] ──────────────► [????] ◄─────────────── [220V Europa]
                              │
                    Traduce entre ambos
                    sin modificar ninguno
```

- **Tu laptop** = código del dominio (core::Frame)
- **Enchufe de pared** = tecnología externa (OpenCV CUDA)
- **Adaptador** = OrbCudaExtractor

Si viajas a USA (enchufe diferente), **solo cambias el adaptador**, no tu laptop.

---

## Checklist: ¿Entendí Clean Architecture?

- [ ] ¿Puedo explicar qué hace cada capa? (core, interfaces, adapters)
- [ ] ¿Puedo identificar qué código va en cada capa?
- [ ] ¿Entiendo por qué los tipos del dominio NO usan cv::Mat?
- [ ] ¿Puedo explicar qué es un "adapter" con mis palabras?
- [ ] ¿Entiendo que main.cpp NO debe saber si usamos CPU o GPU?

---

## Siguiente Paso

Ahora implementa el mismo patrón para:
1. `CudaMatcher` (matching de descriptores)
2. `YoloTrtDetector` (detección de objetos)

La estructura es SIEMPRE la misma:
1. Interface define el contrato
2. Adapter implementa usando tecnología específica
3. Adapter traduce tipos externos → tipos del dominio
