# Migración Legacy → Clean Architecture

## Vista General: El Camino Completo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVOLUCIÓN DEL PROYECTO                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ETAPA 0          ETAPA 1          ETAPA 2          ETAPA 3               │
│   ════════         ════════         ════════         ════════               │
│   Legacy           + Tipos          + Interfaces     + Adapters             │
│   (todo junto)     (core/)          (contracts)      (implementar)          │
│                                                                             │
│   ┌─────┐          ┌─────┐          ┌─────┐          ┌─────┐               │
│   │Frame│          │Frame│          │Frame│          │Frame│               │
│   │.cpp │          │.cpp │          │.cpp │          │.cpp │  ← legacy     │
│   │     │          │     │          │     │          │     │    (intacto)  │
│   │cv:: │          │cv:: │          │cv:: │          │cv:: │               │
│   │todo │          │todo │          │todo │          │todo │               │
│   └─────┘          └─────┘          └─────┘          └─────┘               │
│                       +                +                +                   │
│                    ┌─────┐          ┌─────┐          ┌─────┐               │
│                    │core/│          │core/│          │core/│               │
│                    │Types│          │Types│          │Types│               │
│                    └─────┘          └─────┘          └─────┘               │
│                                        +                +                   │
│                                     ┌─────┐          ┌─────┐               │
│                                     │ I   │          │ I   │               │
│                                     │Match│          │Match│               │
│                                     │er   │          │er   │               │
│                                     └─────┘          └─────┘               │
│                                                         +                   │
│                                                      ┌─────┐               │
│                                                      │Cuda │  ← NUEVO      │
│                                                      │Match│    adapter    │
│                                                      │er   │               │
│                                                      └─────┘               │
│                                                                             │
│   ESTAMOS AQUÍ ─────────────────────────────────────────►                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Etapa 0: Legacy (Punto de Partida)

**Estado:** Todo el código usa OpenCV directamente.

```
src/
├── Frame.cpp          # usa cv::cuda::ORB, cv::KeyPoint, cv::Mat
├── TRTInference.cpp   # usa TensorRT directamente
├── LoopClosure.cpp    # usa cv::BFMatcher
└── main.cpp           # todo mezclado
```

**Problema:**
```cpp
// Frame.cpp - acoplado a OpenCV
class Frame {
    cv::cuda::GpuMat gpu_descriptors;  // tipo de OpenCV
    std::vector<cv::KeyPoint> keypoints;  // tipo de OpenCV
};

// Si quieres cambiar a otro detector (no OpenCV):
// - Tienes que cambiar Frame.cpp
// - Tienes que cambiar main.cpp
// - Tienes que cambiar todos los archivos que usan Frame
```

---

## Etapa 1: Crear Tipos Propios (core/)

**Acción:** Crear tipos que NO dependen de ninguna librería.

```
include/
└── core/
    └── Types.hpp   ← NUEVO
```

**Código:**
```cpp
// include/core/Types.hpp
namespace aria::core {

// Tu propio KeyPoint (NO cv::KeyPoint)
struct KeyPoint {
    float x, y;
    float size;
    float angle;
};

// Tu propio Frame (NO depende de OpenCV)
struct Frame {
    std::vector<KeyPoint> keypoints;
    std::vector<uint8_t> descriptors;
};

// Tu propio Match (NO cv::DMatch)
struct Match {
    int query_idx;
    int train_idx;
    float distance;
};

}
```

**Por qué:**
- Estos tipos son TUYOS
- No importa si usas OpenCV, otro detector, o nada
- El resto del código puede usar estos tipos

---

## Etapa 2: Crear Interfaces (Contratos)

**Acción:** Definir QUÉ hace cada componente, sin decir CÓMO.

```
include/
├── core/
│   └── Types.hpp
└── interfaces/        ← NUEVO
    ├── IMatcher.hpp
    ├── IFeatureExtractor.hpp
    └── IObjectDetector.hpp
```

**Código:**
```cpp
// include/interfaces/IMatcher.hpp
namespace aria::interfaces {

class IMatcher {
public:
    // Solo dice: "un matcher recibe 2 frames y devuelve matches"
    // NO dice cómo lo hace (GPU? CPU? qué librería?)
    virtual void match(
        const core::Frame& query,
        const core::Frame& train,
        std::vector<core::Match>& matches
    ) = 0;  // = 0 significa "sin implementación"
};

}
```

**Por qué:**
- La interfaz es un CONTRATO
- Dice "cualquier matcher debe tener esta función"
- No importa si es CUDA, CPU, o un mock para tests

---

## Etapa 3: Crear Adapters (Implementaciones)

**Acción:** Implementar las interfaces usando librerías concretas.

```
include/
├── core/
│   └── Types.hpp
├── interfaces/
│   └── IMatcher.hpp
└── adapters/          ← NUEVO
    └── gpu/
        └── CudaMatcher.hpp

src/
└── adapters/          ← NUEVO
    └── gpu/
        └── CudaMatcher.cpp
```

**Código:**
```cpp
// CudaMatcher.hpp
class CudaMatcher : public IMatcher {  // "implementa IMatcher"
    void match(...) override;  // implementa la función del contrato
private:
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher_;  // OpenCV aquí dentro
};

// CudaMatcher.cpp
void CudaMatcher::match(
    const core::Frame& query,   // recibe TU tipo
    const core::Frame& train,
    std::vector<core::Match>& matches  // devuelve TU tipo
) {
    // DENTRO traduce a OpenCV
    cv::Mat query_mat = ...;
    cv::cuda::GpuMat query_gpu = ...;

    // Usa OpenCV
    matcher_->knnMatch(...);

    // Traduce resultado a TU tipo
    for (auto& knn : knn_matches) {
        core::Match m;
        m.query_idx = knn[0].queryIdx;
        matches.push_back(m);
    }
}
```

**Por qué:**
- OpenCV está AISLADO dentro del adapter
- El resto del código solo ve `core::Frame` y `core::Match`
- Si cambias de OpenCV a otra cosa, solo cambias el adapter

---

## Etapa 4: Usar en el Pipeline (Futuro)

**Acción:** El pipeline usa interfaces, no implementaciones concretas.

```cpp
// SlamPipeline.cpp (futuro)
class SlamPipeline {
public:
    SlamPipeline(
        std::unique_ptr<IMatcher> matcher,        // interfaz
        std::unique_ptr<IFeatureExtractor> extractor  // interfaz
    ) : matcher_(std::move(matcher)),
        extractor_(std::move(extractor)) {}

    void process(const cv::Mat& image) {
        core::Frame frame;
        extractor_->extract(image, frame);  // usa interfaz

        std::vector<core::Match> matches;
        matcher_->match(frame, prev_frame_, matches);  // usa interfaz
    }

private:
    std::unique_ptr<IMatcher> matcher_;
    std::unique_ptr<IFeatureExtractor> extractor_;
};
```

**Uso:**
```cpp
// main.cpp
auto matcher = std::make_unique<CudaMatcher>();      // implementación GPU
auto extractor = std::make_unique<OrbCudaExtractor>();

SlamPipeline pipeline(std::move(matcher), std::move(extractor));

// O para tests:
auto mock_matcher = std::make_unique<MockMatcher>();  // implementación fake
```

---

## Resumen Visual

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DE LEGACY A CLEAN ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ANTES (Legacy)                    DESPUÉS (Clean)                         │
│   ══════════════                    ═══════════════                         │
│                                                                             │
│   ┌──────────────┐                  ┌──────────────┐                        │
│   │   main.cpp   │                  │   main.cpp   │                        │
│   │              │                  │              │                        │
│   │ cv::KeyPoint │                  │ core::Frame  │ ← tipos propios        │
│   │ cv::Mat      │                  │ core::Match  │                        │
│   │ cv::DMatch   │                  │              │                        │
│   └──────┬───────┘                  └──────┬───────┘                        │
│          │                                 │                                │
│          │ usa directamente                │ usa interfaz                   │
│          ▼                                 ▼                                │
│   ┌──────────────┐                  ┌──────────────┐                        │
│   │  Frame.cpp   │                  │   IMatcher   │ ← contrato             │
│   │              │                  │   (= 0)      │                        │
│   │ OpenCV CUDA  │                  └──────┬───────┘                        │
│   │ directamente │                         │                                │
│   └──────────────┘                         │ implementa                     │
│                                            ▼                                │
│                                     ┌──────────────┐                        │
│                                     │ CudaMatcher  │ ← adapter              │
│                                     │              │                        │
│                                     │ OpenCV CUDA  │ (OpenCV aislado)       │
│                                     │ (interno)    │                        │
│                                     └──────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Checklist de Progreso

### Etapa 1: Tipos (core/) ✅ COMPLETADO
- [x] `core::KeyPoint`
- [x] `core::Frame`
- [x] `core::Match`
- [x] `core::Detection`
- [x] `core::Pose`
- [x] `core::ImuMeasurement`
- [x] `core::KeyFrame`
- [x] `core::MapPoint`
- [x] `core::LoopCandidate`

### Etapa 2: Interfaces ✅ COMPLETADO
- [x] `IMatcher`
- [x] `IFeatureExtractor`
- [x] `IObjectDetector`
- [x] `ILoopDetector`
- [x] `IMapper`
- [x] `ISensorFusion`

### Etapa 3: Adapters 🔄 EN PROGRESO
- [x] `CudaMatcher` ← COMPLETADO
- [ ] `OrbCudaExtractor` (header existe, falta .cpp)
- [ ] `YoloTrtDetector` (header existe, falta .cpp)
- [ ] `G2oMapper`
- [ ] `EkfSensorFusion`

### Etapa 4: Pipeline ⏳ PENDIENTE
- [ ] `SlamPipeline`
- [ ] `PipelineFactory`
- [ ] Tests con mocks

---

## Orden Recomendado para Continuar

```
1. OrbCudaExtractor.cpp   ← Siguiente (extrae features)
   └── Traduce: cv::Mat → core::Frame

2. YoloTrtDetector.cpp    ← Después (detecta objetos)
   └── Traduce: cv::Mat → vector<core::Detection>

3. SlamPipeline.cpp       ← Conecta todo
   └── Usa IMatcher, IFeatureExtractor, IObjectDetector

4. Reemplazar main.cpp    ← Final
   └── Usa SlamPipeline en vez de código legacy
```

---

## Archivos Relacionados

| Archivo | Descripción |
|---------|-------------|
| [CLEAN_ARCHITECTURE_DIAGRAM.md](CLEAN_ARCHITECTURE_DIAGRAM.md) | Diagrama de la arquitectura final |
| [PIPELINE_DIAGRAM_LEGACY.md](PIPELINE_DIAGRAM_LEGACY.md) | Cómo funciona el código legacy |
| [../learn/cpp_basics/](../learn/cpp_basics/) | Ejercicios de C++ básico |
| [../learn/cpp_basics/04_adapter/](../learn/cpp_basics/04_adapter/) | Ejemplo del patrón adapter |
