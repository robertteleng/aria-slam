# Clean Architecture - aria-slam

## Vista General

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLEAN ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    CORE (Dominio)           INTERFACES               ADAPTERS              │
│    ═══════════════          ══════════════           ══════════════        │
│    Tipos puros              Contratos                Implementaciones      │
│    Sin dependencias         (= 0)                    (OpenCV, TensorRT)    │
│                                                                             │
│    ┌───────────┐            ┌───────────┐            ┌───────────┐         │
│    │  Frame    │◄───────────│ IMatcher  │◄───────────│CudaMatcher│         │
│    │  Match    │            │           │            │           │         │
│    │  KeyPoint │            │  match()  │            │ OpenCV    │         │
│    └───────────┘            │   = 0     │            │ CUDA      │         │
│                             └───────────┘            └───────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Estructura de Carpetas

```
include/
├── core/                      # CAPA 1: Dominio (sin dependencias externas)
│   └── Types.hpp              # Frame, Match, KeyPoint, Pose, Detection...
│
├── interfaces/                # CAPA 2: Contratos (solo dependen de core/)
│   ├── IMatcher.hpp           # match(Frame, Frame) → vector<Match>
│   ├── IFeatureExtractor.hpp  # extract(image) → Frame
│   ├── IObjectDetector.hpp    # detect(image) → vector<Detection>
│   ├── ILoopDetector.hpp      # detectLoop(Frame) → LoopCandidate
│   ├── IMapper.hpp            # addKeyframe(), optimize()
│   ├── ISensorFusion.hpp      # fuse(Pose, ImuMeasurement)
│   ├── IAriaDevice.hpp        # Meta Aria glasses interface
│   └── IAudioFeedback.hpp     # Audio feedback interface
│
├── adapters/                  # CAPA 3: Implementaciones concretas
│   └── gpu/
│       ├── CudaMatcher.hpp    # Implementa IMatcher usando OpenCV CUDA
│       ├── OrbCudaExtractor.hpp
│       └── YoloTrtDetector.hpp
│
├── pipeline/                  # CAPA 4: Orquestación
│   └── SlamPipeline.hpp       # Conecta todas las interfaces
│
├── factory/                   # Creación de objetos
│   └── PipelineFactory.hpp    # Crea pipeline con implementaciones concretas
│
└── legacy/                    # Código viejo (H01-H14)
    ├── Frame.hpp
    ├── TRTInference.hpp
    ├── IMU.hpp
    ├── LoopClosure.hpp
    ├── Mapper.hpp
    └── EuRoCReader.hpp

src/
├── adapters/gpu/
│   └── CudaMatcher.cpp        # Implementación
│
└── legacy/                    # Código viejo
    ├── Frame.cpp
    ├── TRTInference.cpp
    └── ...
```

---

## Flujo de Dependencias

```
                         REGLA DE DEPENDENCIA
                    (las flechas apuntan hacia adentro)

                    ┌─────────────────────────────┐
                    │         ADAPTERS            │
                    │  (OpenCV, TensorRT, g2o)    │
                    └──────────────┬──────────────┘
                                   │ depende de
                                   ▼
                    ┌─────────────────────────────┐
                    │        INTERFACES           │
                    │   (IMatcher, IExtractor)    │
                    └──────────────┬──────────────┘
                                   │ depende de
                                   ▼
                    ┌─────────────────────────────┐
                    │           CORE              │
                    │  (Frame, Match, KeyPoint)   │
                    └─────────────────────────────┘


    IMPORTANTE:
    - core/ NO incluye nada de OpenCV, TensorRT, etc.
    - interfaces/ solo incluye core/
    - adapters/ incluye interfaces/ y librerias externas
```

---

## Ejemplo Concreto: CudaMatcher

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PATRON ADAPTER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TU PROGRAMA                    LIBRERIA EXTERNA                           │
│   ════════════                   ════════════════                           │
│   usa tipos propios              usa tipos de OpenCV                        │
│                                                                             │
│   ┌───────────────┐              ┌───────────────┐                          │
│   │ core::Frame   │              │ cv::Mat       │                          │
│   │ core::Match   │              │ cv::DMatch    │                          │
│   │ core::KeyPoint│              │ cv::KeyPoint  │                          │
│   └───────────────┘              └───────────────┘                          │
│           │                              │                                  │
│           │         ┌───────────────┐    │                                  │
│           └────────►│  CudaMatcher  │◄───┘                                  │
│                     │   (ADAPTER)   │                                       │
│                     │               │                                       │
│                     │  Traduce:     │                                       │
│                     │  Frame → Mat  │                                       │
│                     │  DMatch→Match │                                       │
│                     └───────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Código en CudaMatcher.cpp

```cpp
void CudaMatcher::match(
    const core::Frame& query,    // <-- Tipo de TU programa
    const core::Frame& train,
    std::vector<core::Match>& matches,
    float ratio_threshold
) {
    // PASO 1: Traducir core::Frame → cv::Mat
    cv::Mat query_mat(query.numKeypoints(), 32, CV_8UC1,
                      (void*)query.descriptors.data());

    // PASO 2: Subir a GPU
    cv::cuda::GpuMat query_gpu;
    query_gpu.upload(query_mat);

    // PASO 3: Usar libreria externa (OpenCV CUDA)
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(query_gpu, train_gpu, knn_matches, 2);

    // PASO 4: Traducir cv::DMatch → core::Match
    for (auto& knn : knn_matches) {
        if (knn[0].distance < ratio_threshold * knn[1].distance) {
            core::Match m;                    // <-- Tipo de TU programa
            m.query_idx = knn[0].queryIdx;    // <-- Copia datos de OpenCV
            m.train_idx = knn[0].trainIdx;
            m.distance = knn[0].distance;
            matches.push_back(m);
        }
    }
}
```

---

## Interfaces Definidas

### IMatcher (ya implementado: CudaMatcher)

```cpp
class IMatcher {
public:
    virtual void match(
        const core::Frame& query,
        const core::Frame& train,
        std::vector<core::Match>& matches,
        float ratio_threshold = 0.75f
    ) = 0;
};
```

### IFeatureExtractor (pendiente)

```cpp
class IFeatureExtractor {
public:
    virtual void extract(
        const cv::Mat& image,        // entrada: imagen
        core::Frame& frame           // salida: keypoints + descriptors
    ) = 0;
};
```

### IObjectDetector (pendiente)

```cpp
class IObjectDetector {
public:
    virtual void detect(
        const cv::Mat& image,
        std::vector<core::Detection>& detections
    ) = 0;
};
```

---

## Tipos en core/Types.hpp

```cpp
namespace aria::core {

struct KeyPoint {
    float x, y;
    float size;
    float angle;
    float response;
    int octave;
};

struct Frame {
    uint64_t id;
    double timestamp;
    std::vector<KeyPoint> keypoints;
    std::vector<uint8_t> descriptors;  // N x 32 bytes (ORB)
    Eigen::Matrix4d pose;
};

struct Match {
    int query_idx;      // Indice en frame query
    int train_idx;      // Indice en frame train
    float distance;     // Distancia del descriptor
};

struct Detection {
    float x1, y1, x2, y2;   // Bounding box
    float confidence;
    int class_id;
};

} // namespace aria::core
```

---

## Como Usar (Polimorfismo)

```cpp
// main.cpp o SlamPipeline

#include "interfaces/IMatcher.hpp"
#include "adapters/gpu/CudaMatcher.hpp"

int main() {
    // Crear implementacion concreta
    std::unique_ptr<aria::interfaces::IMatcher> matcher =
        std::make_unique<aria::adapters::gpu::CudaMatcher>();

    // Usar a traves de la interfaz
    core::Frame frame1, frame2;
    std::vector<core::Match> matches;

    matcher->match(frame1, frame2, matches, 0.75f);

    // Podrias cambiar a otra implementacion:
    // matcher = std::make_unique<CpuMatcher>();  // Sin cambiar el resto
    // matcher = std::make_unique<MockMatcher>(); // Para tests
}
```

---

## Legacy vs Clean Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPARACION                                          │
├──────────────────────────────┬──────────────────────────────────────────────┤
│         LEGACY               │            CLEAN ARCHITECTURE                │
├──────────────────────────────┼──────────────────────────────────────────────┤
│                              │                                              │
│  Frame.cpp usa:              │  CudaMatcher.cpp usa:                        │
│  - cv::cuda::ORB             │  - core::Frame (nuestro tipo)                │
│  - cv::cuda::GpuMat          │  - IMatcher (interfaz abstracta)             │
│  - cv::KeyPoint              │  - Traduce a cv:: solo internamente          │
│                              │                                              │
│  Acoplado a OpenCV           │  Desacoplado de OpenCV                       │
│                              │                                              │
├──────────────────────────────┼──────────────────────────────────────────────┤
│                              │                                              │
│  Si cambias de CUDA a CPU:   │  Si cambias de CUDA a CPU:                   │
│  - Reescribir Frame.cpp      │  - Crear CpuMatcher.cpp                      │
│  - Cambiar todos los #include│  - Cambiar 1 linea en factory                │
│  - Cambiar tipos en todo el  │  - El resto del codigo igual                 │
│    codigo                    │                                              │
│                              │                                              │
└──────────────────────────────┴──────────────────────────────────────────────┘
```

---

## Progreso H12

| Componente | Interfaz | Adapter | Estado |
|------------|----------|---------|--------|
| Matching | IMatcher | CudaMatcher | COMPLETO |
| Feature Extraction | IFeatureExtractor | OrbCudaExtractor | Header only |
| Object Detection | IObjectDetector | YoloTrtDetector | Header only |
| Loop Detection | ILoopDetector | - | Pendiente |
| Mapping | IMapper | - | Pendiente |
| Sensor Fusion | ISensorFusion | - | Pendiente |

---

## Documentos Relacionados

- [PIPELINE_DIAGRAM_LEGACY.md](PIPELINE_DIAGRAM_LEGACY.md) - Diagrama del sistema legacy (H01-H14)
- [learn/cpp_basics/README.md](../learn/cpp_basics/README.md) - Conceptos C++ basicos
- [learn/cpp_basics/04_adapter/](../learn/cpp_basics/04_adapter/) - Ejercicio del patron adapter
