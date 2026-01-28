# H12: Clean Architecture - Theory & Design

**Status:** 📐 Design Document (Architecture Blueprint)

---

## Teoría: ¿Por Qué Arquitectura de Software?

### El Problema del Código Acoplado

Sin arquitectura clara, el código SLAM típico termina así:

```cpp
// ❌ Código acoplado (anti-patrón)
void processFrame(cv::Mat& image) {
    cv::cuda::GpuMat d_image;
    d_image.upload(image);                    // Acoplado a CUDA

    auto orb = cv::cuda::ORB::create(1000);   // Acoplado a OpenCV CUDA
    orb->detectAndCompute(...);

    // 500 líneas después...
    auto engine = loadTensorRT("yolo.engine"); // Acoplado a TensorRT

    // ¿Cómo testear esto sin GPU?
    // ¿Cómo cambiar ORB por SuperPoint?
    // ¿Cómo saber qué hace cada parte?
}
```

**Problemas:**
1. **Imposible testear** - Requiere GPU física
2. **Imposible cambiar** - ORB está hardcodeado
3. **Imposible entender** - 500 líneas mezcladas
4. **Imposible mantener** - Un cambio rompe todo

### La Solución: Separación en Capas

```
┌─────────────────────────────────────────────────────────────────────┐
│  "El código bien arquitectado es como un edificio bien diseñado:   │
│   cada piso tiene su propósito, las escaleras conectan todo,       │
│   y puedes renovar un piso sin demoler los demás."                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Teoría: Hexagonal Architecture (Ports & Adapters)

### Origen

Propuesta por Alistair Cockburn (2005). También conocida como:
- **Ports & Adapters**
- **Onion Architecture** (variante de Jeffrey Palermo)
- **Clean Architecture** (variante de Robert C. Martin)

### Concepto Central

```
                    ┌─────────────────────────────────────┐
                    │         MUNDO EXTERIOR              │
                    │  (GPU, archivos, red, sensores)     │
                    └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────┐
                    │           ADAPTERS                  │
                    │  (Traducen tecnología → dominio)    │
                    │  OrbCuda, TensorRT, EuRoCReader     │
                    └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────┐
                    │            PORTS                    │
                    │     (Contratos/Interfaces)          │
                    │  IFeatureExtractor, IMatcher, ...   │
                    └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────┐
                    │         DOMAIN (CORE)               │
                    │    (Lógica de negocio pura)         │
                    │   Frame, KeyFrame, MapPoint, Pose   │
                    └─────────────────────────────────────┘

        Regla de Dependencia: Las flechas SIEMPRE apuntan hacia adentro.
        El dominio NO conoce a los adapters. Los adapters conocen al dominio.
```

### Analogía: El Enchufe Universal

```
    Adaptador de Viaje
    ┌─────────────────┐
    │  ┌───┐   ┌───┐  │
    │  │ A │   │ B │  │     A = Enchufe europeo (tu dispositivo)
    │  └─┬─┘   └─┬─┘  │     B = Enchufe americano (la pared)
    │    │       │    │
    │    └───┬───┘    │     El adaptador traduce entre ambos
    │        │        │     sin modificar ninguno de los dos.
    └────────┼────────┘
             │
    ═════════╧═════════

    En código:
    - Tu dispositivo = Lógica de dominio (Frame, Pose)
    - La pared = Tecnología externa (CUDA, TensorRT)
    - Adaptador = OrbCudaExtractor, YoloTrtDetector
```

### Port (Puerto)

**Definición:** Interfaz abstracta que define un contrato.

```cpp
// Port = Contrato que el dominio necesita
class IFeatureExtractor {
public:
    virtual void extract(const uint8_t* image, int w, int h, Frame& out) = 0;
    virtual ~IFeatureExtractor() = default;
};
```

**Características:**
- No tiene implementación
- No conoce tecnologías específicas
- Define QUÉ se necesita, no CÓMO se hace
- Usa tipos del dominio (Frame, no cv::Mat)

### Adapter (Adaptador)

**Definición:** Implementación concreta que traduce tecnología externa al contrato del puerto.

```cpp
// Adapter = Implementación específica
class OrbCudaExtractor : public IFeatureExtractor {
public:
    void extract(const uint8_t* image, int w, int h, Frame& out) override {
        // Traducción: uint8_t* → cv::cuda::GpuMat
        cv::cuda::GpuMat d_image;
        d_image.upload(cv::Mat(h, w, CV_8UC1, (void*)image));

        // Usa tecnología específica (CUDA)
        orb_->detectAndCompute(d_image, ...);

        // Traducción: cv::KeyPoint → core::KeyPoint
        for (auto& kp : cv_keypoints) {
            out.keypoints.push_back({kp.pt.x, kp.pt.y, ...});
        }
    }
private:
    cv::Ptr<cv::cuda::ORB> orb_;  // Detalle de implementación oculto
};
```

---

## Teoría: SOLID Principles

### S - Single Responsibility Principle (SRP)

**"Una clase debe tener una, y solo una, razón para cambiar."**

```cpp
// ❌ Viola SRP: Hace extracción Y matching Y detección
class MegaProcessor {
    void process(cv::Mat& img) {
        extractFeatures(img);
        matchFeatures();
        detectObjects();
        computePose();
        saveToFile();
    }
};

// ✅ Cumple SRP: Cada clase tiene UNA responsabilidad
class OrbCudaExtractor { /* Solo extrae features */ };
class CudaMatcher { /* Solo hace matching */ };
class YoloTrtDetector { /* Solo detecta objetos */ };
class PoseEstimator { /* Solo estima pose */ };
```

**En SLAM:**
| Clase | Responsabilidad única |
|-------|----------------------|
| `OrbCudaExtractor` | Extraer keypoints y descriptores |
| `CudaMatcher` | Encontrar correspondencias entre descriptores |
| `YoloTrtDetector` | Detectar objetos en la imagen |
| `EKFSensorFusion` | Fusionar IMU y visual odometry |
| `LoopDetector` | Detectar cierres de bucle |

### O - Open/Closed Principle (OCP)

**"Abierto para extensión, cerrado para modificación."**

```cpp
// ❌ Viola OCP: Hay que modificar SlamPipeline para agregar SuperPoint
class SlamPipeline {
    void processFrame() {
        if (use_orb_) {
            orb_cuda_->detect(...);  // Hardcodeado
        } else if (use_superpoint_) {
            superpoint_->detect(...); // Hay que agregar esto
        }
        // Cada nuevo extractor requiere modificar esta clase
    }
};

// ✅ Cumple OCP: Agregar SuperPoint sin tocar SlamPipeline
class SlamPipeline {
    IFeatureExtractor* extractor_;  // Inyectado

    void processFrame() {
        extractor_->extract(...);   // Funciona con cualquier extractor
    }
};

// Agregar nuevo extractor: solo crear nueva clase
class SuperPointExtractor : public IFeatureExtractor {
    void extract(...) override { /* Implementación */ }
};
```

### L - Liskov Substitution Principle (LSP)

**"Los objetos de una superclase deben poder reemplazarse por objetos de sus subclases sin alterar el programa."**

```cpp
// Ambos deben ser intercambiables sin romper el código
IFeatureExtractor* extractor;

extractor = new OrbCudaExtractor();  // Funciona
extractor = new OrbCpuExtractor();   // También funciona
extractor = new SuperPointExtractor(); // También funciona

// El código que usa extractor NO cambia:
Frame frame;
extractor->extract(image_data, 640, 480, frame);
```

**Violación típica:**

```cpp
// ❌ Viola LSP: La CPU no soporta extractAsync()
class OrbCpuExtractor : public IFeatureExtractor {
    void extractAsync(...) override {
        throw std::runtime_error("CPU no soporta async!");
        // Esto rompe código que espera que async funcione
    }
};

// ✅ Cumple LSP: Proporcionar comportamiento por defecto
class IFeatureExtractor {
    virtual void extractAsync(...) {
        extract(...);  // Default: ejecutar síncronamente
    }
};
```

### I - Interface Segregation Principle (ISP)

**"Muchas interfaces específicas son mejores que una interfaz general."**

```cpp
// ❌ Viola ISP: Interfaz "gorda" que obliga a implementar todo
class IVisionComponent {
    virtual void extract(...) = 0;
    virtual void match(...) = 0;
    virtual void detectObjects(...) = 0;
    virtual void estimatePose(...) = 0;
    virtual void optimizeGraph(...) = 0;
    // Un Matcher tiene que implementar extract() aunque no lo use
};

// ✅ Cumple ISP: Interfaces segregadas por responsabilidad
class IFeatureExtractor { virtual void extract(...) = 0; };
class IMatcher { virtual void match(...) = 0; };
class IObjectDetector { virtual void detect(...) = 0; };
class IPoseEstimator { virtual void estimate(...) = 0; };
```

### D - Dependency Inversion Principle (DIP)

**"Depende de abstracciones, no de implementaciones concretas."**

```cpp
// ❌ Viola DIP: Depende de clase concreta
class SlamPipeline {
    OrbCudaExtractor extractor_;  // Dependencia concreta
    // No puedo usar otro extractor sin modificar esta clase
};

// ✅ Cumple DIP: Depende de abstracción
class SlamPipeline {
    IFeatureExtractor* extractor_;  // Dependencia abstracta

    // Constructor recibe la abstracción (Dependency Injection)
    SlamPipeline(IFeatureExtractor* ext) : extractor_(ext) {}
};

// El "main" o Factory decide qué implementación usar
int main() {
    OrbCudaExtractor cuda_extractor;
    SlamPipeline pipeline(&cuda_extractor);

    // O para testing:
    MockExtractor mock;
    SlamPipeline test_pipeline(&mock);
}
```

---

## Teoría: Dependency Injection (DI)

### ¿Qué es?

**Inyección de Dependencias:** Técnica donde las dependencias se pasan desde afuera en lugar de crearlas internamente.

```cpp
// ❌ Sin DI: La clase crea sus dependencias
class SlamPipeline {
    SlamPipeline() {
        extractor_ = new OrbCudaExtractor();  // Hardcodeado
        matcher_ = new CudaMatcher();          // Hardcodeado
    }
};

// ✅ Con DI: Las dependencias se inyectan
class SlamPipeline {
    SlamPipeline(
        IFeatureExtractor* extractor,
        IMatcher* matcher
    ) : extractor_(extractor), matcher_(matcher) {}
};
```

### Tipos de Inyección

```cpp
// 1. Constructor Injection (Preferido)
class SlamPipeline {
    SlamPipeline(IFeatureExtractor* ext, IMatcher* match);
};

// 2. Setter Injection
class SlamPipeline {
    void setExtractor(IFeatureExtractor* ext);
    void setMatcher(IMatcher* match);
};

// 3. Interface Injection
class IExtractorAware {
    virtual void injectExtractor(IFeatureExtractor* ext) = 0;
};
```

### Factory Pattern

El Factory centraliza la creación de objetos con sus dependencias:

```cpp
class PipelineFactory {
public:
    static std::unique_ptr<SlamPipeline> createGpu() {
        auto extractor = std::make_unique<OrbCudaExtractor>();
        auto matcher = std::make_unique<CudaMatcher>();
        auto detector = std::make_unique<YoloTrtDetector>("yolo.engine");

        return std::make_unique<SlamPipeline>(
            std::move(extractor),
            std::move(matcher),
            std::move(detector)
        );
    }

    static std::unique_ptr<SlamPipeline> createCpu() {
        auto extractor = std::make_unique<OrbCpuExtractor>();
        auto matcher = std::make_unique<BruteForceMatcher>();
        // Sin detector de objetos en CPU

        return std::make_unique<SlamPipeline>(
            std::move(extractor),
            std::move(matcher),
            nullptr
        );
    }

    static std::unique_ptr<SlamPipeline> createMock() {
        return std::make_unique<SlamPipeline>(
            std::make_unique<MockExtractor>(),
            std::make_unique<MockMatcher>(),
            std::make_unique<MockDetector>()
        );
    }
};
```

---

## Teoría: Domain-Driven Design (DDD) Concepts

### Entidades de Dominio

**Entidad:** Objeto con identidad única que persiste en el tiempo.

```cpp
struct Frame {
    uint64_t id;          // ← Identidad única
    double timestamp;
    std::vector<KeyPoint> keypoints;
    std::vector<uint8_t> descriptors;
};

// Dos frames con mismo contenido pero distinto ID son diferentes
Frame f1{.id = 1, .keypoints = {...}};
Frame f2{.id = 2, .keypoints = {...}};  // f1 != f2
```

### Value Objects

**Value Object:** Objeto sin identidad, definido por sus atributos.

```cpp
struct KeyPoint {
    float x, y;
    float size;
    float angle;
    // Sin ID - dos KeyPoints con mismos valores son iguales
};

KeyPoint kp1{100.0f, 200.0f, 31.0f, 45.0f};
KeyPoint kp2{100.0f, 200.0f, 31.0f, 45.0f};
// kp1 == kp2 (mismo valor = mismo objeto conceptual)
```

### Agregados

**Agregado:** Grupo de entidades tratadas como unidad.

```cpp
// KeyFrame es un agregado que contiene Frame + observaciones
struct KeyFrame {
    Frame frame;                              // Entidad contenida
    std::vector<uint64_t> observed_mappoints; // Referencias
    std::vector<uint64_t> covisible_keyframes;

    // El KeyFrame es la "raíz del agregado"
    // Acceso a mappoints/covisibility solo a través de KeyFrame
};
```

---

## Teoría: Capas de la Arquitectura

### Por Qué Capas

```
┌────────────────────────────────────────────────────────────────────┐
│                    SIN CAPAS vs CON CAPAS                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   SIN CAPAS (Espagueti)         CON CAPAS (Lasaña)                │
│                                                                    │
│   ┌──────────────────┐          ┌──────────────────┐              │
│   │ main.cpp         │          │  Application     │ ← Orquesta   │
│   │ ┌──────────────┐ │          ├──────────────────┤              │
│   │ │ CUDA code    │ │          │  Ports           │ ← Contratos  │
│   │ │ TRT code     │ │          ├──────────────────┤              │
│   │ │ OpenCV code  │ │          │  Adapters        │ ← Implementa │
│   │ │ g2o code     │ │          ├──────────────────┤              │
│   │ │ Domain logic │ │          │  Domain          │ ← Lógica     │
│   │ │ ALL MIXED!   │ │          └──────────────────┘              │
│   │ └──────────────┘ │                                            │
│   └──────────────────┘          Cada capa tiene su rol            │
│                                                                    │
│   Problema: Todo depende        Solución: Dependencias claras     │
│   de todo. Un cambio en CUDA    Un cambio en CUDA solo afecta     │
│   puede romper g2o.             al adapter de CUDA.               │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Capa 1: Domain (Core)

**Propósito:** Contener la lógica de negocio pura, sin dependencias externas.

```cpp
namespace aria::core {

// Solo tipos básicos de C++ y Eigen (matemáticas puras)
struct Frame {
    uint64_t id;
    double timestamp;
    std::vector<KeyPoint> keypoints;
    std::vector<uint8_t> descriptors;
    Eigen::Matrix4d pose;
};

// NO hay: cv::Mat, cuda::GpuMat, nvinfer1::*, g2o::*
// Solo: std::*, Eigen::*, tipos propios

}
```

**Reglas:**
- Cero dependencias a librerías externas (excepto Eigen para matemáticas)
- No conoce cómo se extraen features ni cómo se hace matching
- Puede compilar y testearse sin GPU, sin OpenCV, sin TensorRT

### Capa 2: Ports (Interfaces)

**Propósito:** Definir contratos que el dominio necesita.

```cpp
namespace aria::interfaces {

class IFeatureExtractor {
public:
    // Contrato: dado una imagen, extraer features a un Frame
    // No dice CÓMO hacerlo (CPU? GPU? ORB? SIFT?)
    virtual void extract(
        const uint8_t* image_data,  // Tipo básico, no cv::Mat
        int width, int height,
        core::Frame& frame          // Tipo del dominio
    ) = 0;

    virtual ~IFeatureExtractor() = default;
};

}
```

**Reglas:**
- Solo definiciones abstractas (= 0)
- Usa tipos del dominio, no de librerías externas
- Un port por responsabilidad (ISP)

### Capa 3: Adapters (Implementaciones)

**Propósito:** Implementar los ports usando tecnologías específicas.

```cpp
namespace aria::adapters::gpu {

class OrbCudaExtractor : public interfaces::IFeatureExtractor {
public:
    void extract(const uint8_t* image, int w, int h, core::Frame& frame) override {
        // Aquí SÍ usamos OpenCV CUDA
        cv::cuda::GpuMat d_image;
        d_image.upload(cv::Mat(h, w, CV_8UC1, (void*)image));

        cv::cuda::GpuMat d_keypoints, d_descriptors;
        orb_->detectAndComputeAsync(d_image, cv::cuda::GpuMat(),
                                     d_keypoints, d_descriptors);

        // Traducir cv::KeyPoint → core::KeyPoint
        std::vector<cv::KeyPoint> cv_kps;
        orb_->convert(d_keypoints, cv_kps);

        for (const auto& kp : cv_kps) {
            frame.keypoints.push_back({
                kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response, kp.octave
            });
        }
    }

private:
    cv::Ptr<cv::cuda::ORB> orb_;
};

}
```

**Reglas:**
- Implementa exactamente un port
- Traduce entre tipos externos (cv::KeyPoint) y dominio (core::KeyPoint)
- Encapsula todos los detalles de la tecnología

### Capa 4: Application (Pipeline)

**Propósito:** Orquestar los componentes para ejecutar casos de uso.

```cpp
namespace aria::pipeline {

class SlamPipeline {
public:
    // Recibe interfaces, no implementaciones concretas
    SlamPipeline(
        interfaces::FeatureExtractorPtr extractor,
        interfaces::MatcherPtr matcher,
        interfaces::LoopDetectorPtr loop_detector
    );

    core::Pose processFrame(const uint8_t* image, int w, int h, double ts) {
        // Orquesta el flujo, pero no sabe los detalles
        core::Frame frame;
        extractor_->extract(image, w, h, frame);

        std::vector<interfaces::Match> matches;
        matcher_->match(frame, *prev_frame_, matches);

        // ... resto del pipeline
    }

private:
    interfaces::FeatureExtractorPtr extractor_;
    interfaces::MatcherPtr matcher_;
    interfaces::LoopDetectorPtr loop_detector_;
};

}
```

---

## Teoría: Beneficios Prácticos

### 1. Testabilidad

```cpp
// Sin arquitectura: Necesitas GPU para testear
TEST(SlamTest, ProcessFrame) {
    SlamPipeline pipeline;  // Crea OrbCuda internamente
    // FALLA si no hay GPU
}

// Con arquitectura: Mock sin dependencias
class MockExtractor : public IFeatureExtractor {
    void extract(..., Frame& frame) override {
        // Retorna datos predefinidos para testing
        frame.keypoints = {{100, 100}, {200, 200}};
        frame.descriptors = {/* datos de prueba */};
    }
};

TEST(SlamTest, ProcessFrame) {
    auto mock_extractor = std::make_unique<MockExtractor>();
    auto mock_matcher = std::make_unique<MockMatcher>();

    SlamPipeline pipeline(std::move(mock_extractor), std::move(mock_matcher));

    auto pose = pipeline.processFrame(test_image, 640, 480, 0.0);
    EXPECT_FALSE(pose.position.hasNaN());
    // Funciona sin GPU!
}
```

### 2. Flexibilidad

```cpp
// Cambiar ORB por SuperPoint: solo crear nuevo adapter
class SuperPointExtractor : public IFeatureExtractor {
    void extract(...) override {
        // Usa PyTorch/ONNX/TensorRT para SuperPoint
    }
};

// El pipeline NO cambia
auto pipeline = PipelineFactory::create(config);
// Internamente usa SuperPointExtractor si config lo indica
```

### 3. Mantenibilidad

```
Estructura clara de directorios:

include/
├── core/           ← Dominio (sin dependencias)
│   ├── Frame.hpp
│   └── Pose.hpp
├── interfaces/     ← Contratos (solo abstracciones)
│   ├── IFeatureExtractor.hpp
│   └── IMatcher.hpp
├── adapters/       ← Implementaciones (tecnología específica)
│   ├── gpu/
│   │   └── OrbCudaExtractor.hpp
│   └── cpu/
│       └── OrbCpuExtractor.hpp
└── pipeline/       ← Aplicación (orquestación)
    └── SlamPipeline.hpp

"¿Dónde está el código de CUDA?" → adapters/gpu/
"¿Dónde están las interfaces?" → interfaces/
"¿Dónde está la lógica de negocio?" → core/ y pipeline/
```

### 4. Paralelismo con H13

```cpp
// Las interfaces permiten ejecución async sin cambiar el dominio
class IFeatureExtractor {
    virtual void extractAsync(...) { extract(...); }  // Default: sync
    virtual void sync() {}
};

// El adapter GPU implementa async real
class OrbCudaExtractor : public IFeatureExtractor {
    void extractAsync(...) override {
        // Ejecuta en stream CUDA
        orb_->detectAndComputeAsync(..., stream_);
    }

    void sync() override {
        stream_.waitForCompletion();
    }
};

// El pipeline puede usar async sin conocer los detalles
extractor_->extractAsync(image, w, h, frame);
object_detector_->detectAsync(image, w, h);  // En paralelo
extractor_->sync();
object_detector_->sync();
```

---

## Interview Questions

### Q1: ¿Cuál es la diferencia entre Hexagonal Architecture y Clean Architecture?

**Respuesta:**

Son variantes del mismo concepto con diferente énfasis:

| Aspecto | Hexagonal (Cockburn) | Clean (Martin) |
|---------|---------------------|----------------|
| Énfasis | Ports & Adapters | Capas concéntricas |
| Metáfora | Hexágono con puertos | Círculos concéntricos |
| Regla central | Adapters traducen | Dependency Rule |

```
Hexagonal:                    Clean Architecture:
    ┌──────┐                       ┌───────────────┐
   ╱        ╲                      │   Entities    │
  │  DOMAIN  │                     ├───────────────┤
  │          │                     │   Use Cases   │
   ╲        ╱                      ├───────────────┤
    └──────┘                       │   Interface   │
   /│╲    /│╲                      │   Adapters    │
  P  P   P  P                      ├───────────────┤
  O  O   O  O                      │  Frameworks   │
  R  R   R  R                      │   & Drivers   │
  T  T   T  T                      └───────────────┘
  S  S   S  S
```

En la práctica, ambas logran lo mismo: **aislar el dominio de los detalles de implementación**.

### Q2: ¿Por qué usar `std::unique_ptr` en lugar de punteros raw para DI?

**Respuesta:**

```cpp
// ❌ Raw pointer: ¿Quién hace delete? ¿Es ownership o referencia?
class SlamPipeline {
    IFeatureExtractor* extractor_;  // ¿Debo hacer delete en destructor?
};

// ✅ unique_ptr: Ownership claro, destrucción automática
class SlamPipeline {
    std::unique_ptr<IFeatureExtractor> extractor_;
    // Se destruye automáticamente cuando SlamPipeline se destruye
};

// ✅ shared_ptr: Cuando múltiples objetos comparten ownership
class SlamPipeline {
    std::shared_ptr<ILoopDetector> loop_detector_;
    // Puede ser compartido con LoopClosureThread
};
```

**Regla práctica:**
- `unique_ptr`: El pipeline es dueño exclusivo del componente
- `shared_ptr`: Múltiples objetos comparten el componente
- `raw pointer` / `reference`: Solo si el lifetime está garantizado externamente

### Q3: ¿Cómo evitar que el dominio dependa de Eigen?

**Respuesta:**

Técnicamente Eigen es una dependencia, pero se considera aceptable porque:

1. **Eigen es header-only** - No requiere linking dinámico
2. **Eigen es matemáticas puras** - No es "infraestructura"
3. **Alternativa más pura:**

```cpp
// Sin Eigen (más puro pero más verbose)
namespace aria::core {

struct Vector3 {
    double x, y, z;
    Vector3 operator+(const Vector3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    double dot(const Vector3& o) const { return x*o.x + y*o.y + z*o.z; }
};

struct Matrix4 {
    double data[16];
    Vector3 transform(const Vector3& v) const;
    static Matrix4 identity();
};

}
```

**Trade-off:** Eigen proporciona operaciones matriciales optimizadas (SIMD) que serían costosas de reimplementar. En SLAM, la matemática ES el dominio.

### Q4: ¿Cómo manejar configuración sin violar DIP?

**Respuesta:**

```cpp
// ❌ Viola DIP: El adapter lee su propia config
class OrbCudaExtractor : public IFeatureExtractor {
    OrbCudaExtractor() {
        max_features_ = readConfigFile("orb.yaml");  // Dependencia a filesystem
    }
};

// ✅ Inyectar configuración
struct OrbConfig {
    int max_features = 1000;
    int num_levels = 8;
    float scale_factor = 1.2f;
};

class OrbCudaExtractor : public IFeatureExtractor {
    explicit OrbCudaExtractor(const OrbConfig& config) {
        orb_ = cv::cuda::ORB::create(config.max_features, ...);
    }
};

// El Factory lee config y la inyecta
class PipelineFactory {
    static auto create(const std::string& config_path) {
        auto config = YAML::LoadFile(config_path);
        OrbConfig orb_cfg;
        orb_cfg.max_features = config["orb"]["max_features"].as<int>();

        return std::make_unique<OrbCudaExtractor>(orb_cfg);
    }
};
```

### Q5: ¿Cuándo es aceptable violar SOLID?

**Respuesta:**

SOLID son guías, no leyes absolutas. Violaciones aceptables:

1. **Prototipos rápidos** - Valida la idea antes de arquitectar
2. **Código que nunca cambiará** - Algoritmo matemático estándar
3. **Performance crítica** - Si la abstracción cuesta ciclos inaceptables

```cpp
// Violación aceptable por performance
class UltraFastMatcher {
    // Implementación inline específica para CUDA
    // No usa interface porque el virtual call overhead importa
    __device__ void matchKernel(...) { /* CUDA kernel directo */ }
};

// Pero envuélvelo para el resto del sistema
class CudaMatcherAdapter : public IMatcher {
    UltraFastMatcher fast_impl_;  // Implementación optimizada interna

    void match(...) override {
        fast_impl_.run(...);  // Adapter traduce
    }
};
```

---

## C++ Concepts Used

### 1. Pure Virtual Functions

```cpp
class IFeatureExtractor {
    virtual void extract(...) = 0;  // = 0 hace la clase abstracta
};

// No se puede instanciar:
// IFeatureExtractor ext;  // ERROR: cannot instantiate abstract class
```

### 2. Override Specifier

```cpp
class OrbCudaExtractor : public IFeatureExtractor {
    void extract(...) override;  // Garantiza que override existe en base
    // void extrac(...) override;  // ERROR de compilación: typo detectado
};
```

### 3. Smart Pointers

```cpp
using FeatureExtractorPtr = std::unique_ptr<IFeatureExtractor>;
using SharedExtractorPtr = std::shared_ptr<IFeatureExtractor>;

// Transfer ownership
void SlamPipeline::setExtractor(FeatureExtractorPtr ext) {
    extractor_ = std::move(ext);  // Transfer, no copy
}
```

### 4. RAII (Resource Acquisition Is Initialization)

```cpp
class OrbCudaExtractor {
    cv::cuda::Stream stream_;  // RAII: se destruye automáticamente

    // No necesita destructor explícito si todos los miembros son RAII
};
```

### 5. Namespaces for Organization

```cpp
namespace aria {
    namespace core { /* Domain */ }
    namespace interfaces { /* Ports */ }
    namespace adapters {
        namespace gpu { /* GPU adapters */ }
        namespace cpu { /* CPU adapters */ }
    }
    namespace pipeline { /* Application */ }
    namespace factory { /* DI Factory */ }
}
```

---

## Preparation Checklist

### Conceptos Teóricos

- [ ] Explicar Hexagonal Architecture con diagrama
- [ ] Enumerar y explicar los 5 principios SOLID
- [ ] Diferenciar Port vs Adapter con ejemplo
- [ ] Explicar Dependency Injection y sus tipos
- [ ] Diferenciar Entity vs Value Object
- [ ] Explicar la "Dependency Rule" (flechas hacia adentro)

### Diseño Práctico

- [ ] Diseñar una interface para un componente dado
- [ ] Identificar violaciones de SOLID en código existente
- [ ] Crear Factory para inyección de dependencias
- [ ] Explicar cómo testear con mocks
- [ ] Diseñar estructura de directorios por capas

### C++ Específico

- [ ] Cuándo usar `unique_ptr` vs `shared_ptr` vs raw pointer
- [ ] Propósito de `virtual`, `override`, `= 0`, `final`
- [ ] RAII y gestión de recursos
- [ ] Move semantics con `std::move` para transfer de ownership

---

## Overview

This document defines the architectural refactoring of aria-slam following **Hexagonal Architecture** (Ports & Adapters) and **SOLID principles**. The goal is to enable:

- **Testability**: Mock any component for unit testing
- **Flexibility**: Swap CPU/GPU implementations without changing business logic
- **Maintainability**: Clear boundaries between layers
- **Multithreading**: Thread-safe interfaces ready for H13

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ SlamPipeline│  │ EurocRunner │  │  AriaRunner │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Ports (Interfaces)                          │
│  ┌────────────────┐  ┌────────────┐  ┌─────────────────┐        │
│  │IFeatureExtractor│  │  IMatcher  │  │ ILoopDetector   │        │
│  └────────────────┘  └────────────┘  └─────────────────┘        │
│  ┌────────────────┐  ┌────────────┐  ┌─────────────────┐        │
│  │IObjectDetector │  │ISensorFusion│  │    IMapper      │        │
│  └────────────────┘  └────────────┘  └─────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
          ▲                ▲                ▲
          │                │                │
┌─────────┼────────────────┼────────────────┼─────────────────────┐
│         │    Adapters    │                │                      │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐              │
│  │  GPU Impl   │  │  CPU Impl   │  │    Mocks    │              │
│  │ OrbCuda     │  │  OrbCpu     │  │ MockExtract │              │
│  │ CudaMatcher │  │  BFMatcher  │  │ MockMatcher │              │
│  │ YoloTrt     │  │             │  │             │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
          ▲                ▲                ▲
          │                │                │
┌─────────────────────────────────────────────────────────────────┐
│                     Domain Layer (Core)                          │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐          │
│  │  Frame  │  │ KeyFrame │  │ MapPoint │  │  Pose   │          │
│  └─────────┘  └──────────┘  └──────────┘  └─────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Domain Layer (Core Entities)

Pure data structures with no dependencies on OpenCV, CUDA, or external libraries.

### Frame.hpp

```cpp
#pragma once
#include <vector>
#include <cstdint>
#include <Eigen/Dense>

namespace aria::core {

struct KeyPoint {
    float x, y;           // Position in image
    float size;           // Diameter of meaningful keypoint neighborhood
    float angle;          // Orientation in degrees [0, 360)
    float response;       // Response by which the keypoints are sorted
    int octave;           // Octave (pyramid layer) from which the keypoint was extracted
};

struct Frame {
    uint64_t id;
    double timestamp;
    int width, height;

    std::vector<KeyPoint> keypoints;
    std::vector<uint8_t> descriptors;  // Flattened: N x 32 bytes for ORB

    // Computed pose (optional, filled after tracking)
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

    size_t descriptorSize() const { return 32; }  // ORB descriptor size
    size_t numKeypoints() const { return keypoints.size(); }
};

} // namespace aria::core
```

### KeyFrame.hpp

```cpp
#pragma once
#include "Frame.hpp"
#include <memory>

namespace aria::core {

struct KeyFrame {
    uint64_t id;
    double timestamp;

    Frame frame;
    Eigen::Matrix4d pose;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;

    // Covisibility (frames that see same map points)
    std::vector<uint64_t> covisible_keyframes;

    // Map point observations
    std::vector<uint64_t> observed_mappoints;
};

} // namespace aria::core
```

### MapPoint.hpp

```cpp
#pragma once
#include <Eigen/Dense>
#include <vector>

namespace aria::core {

struct MapPoint {
    uint64_t id;
    Eigen::Vector3d position;
    Eigen::Vector3d normal;       // Mean viewing direction

    std::vector<uint8_t> descriptor;  // Representative descriptor

    // Observations: keyframe_id -> keypoint_index
    std::vector<std::pair<uint64_t, int>> observations;

    // Quality metrics
    int num_observations = 0;
    float min_distance = 0.0f;    // Scale invariance bounds
    float max_distance = 0.0f;

    bool is_bad = false;
};

} // namespace aria::core
```

### Pose.hpp

```cpp
#pragma once
#include <Eigen/Dense>

namespace aria::core {

struct Pose {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    double timestamp;

    // Covariance (6x6: position + orientation)
    Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity();

    Eigen::Matrix4d toMatrix() const {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = orientation.toRotationMatrix();
        T.block<3,1>(0,3) = position;
        return T;
    }

    static Pose fromMatrix(const Eigen::Matrix4d& T, double ts = 0.0) {
        Pose p;
        p.position = T.block<3,1>(0,3);
        p.orientation = Eigen::Quaterniond(T.block<3,3>(0,0));
        p.timestamp = ts;
        return p;
    }
};

} // namespace aria::core
```

## Ports (Interfaces)

Abstract interfaces that define contracts. No implementation details.

### IFeatureExtractor.hpp

```cpp
#pragma once
#include "core/Frame.hpp"
#include <memory>
#include <vector>

namespace aria::interfaces {

class IFeatureExtractor {
public:
    virtual ~IFeatureExtractor() = default;

    // Extract keypoints and descriptors from raw image data
    // @param image_data Raw pixel data (grayscale, row-major)
    // @param width Image width
    // @param height Image height
    // @param frame Output frame with keypoints and descriptors
    virtual void extract(
        const uint8_t* image_data,
        int width,
        int height,
        core::Frame& frame
    ) = 0;

    // Async extraction (for GPU implementations)
    // Returns immediately, results available after sync()
    virtual void extractAsync(
        const uint8_t* image_data,
        int width,
        int height,
        core::Frame& frame
    ) { extract(image_data, width, height, frame); }  // Default: sync

    // Wait for async operation to complete
    virtual void sync() {}

    // Configuration
    virtual void setMaxFeatures(int n) = 0;
    virtual int getMaxFeatures() const = 0;
};

using FeatureExtractorPtr = std::unique_ptr<IFeatureExtractor>;

} // namespace aria::interfaces
```

### IMatcher.hpp

```cpp
#pragma once
#include "core/Frame.hpp"
#include <vector>

namespace aria::interfaces {

struct Match {
    int query_idx;      // Index in query frame
    int train_idx;      // Index in train frame
    float distance;     // Descriptor distance
};

class IMatcher {
public:
    virtual ~IMatcher() = default;

    // Match descriptors between two frames
    // @param query Query frame (current)
    // @param train Train frame (previous/reference)
    // @param matches Output matches
    // @param ratio_threshold Lowe's ratio test threshold (0.0 = disabled)
    virtual void match(
        const core::Frame& query,
        const core::Frame& train,
        std::vector<Match>& matches,
        float ratio_threshold = 0.75f
    ) = 0;

    // Match one frame against multiple (for loop closure)
    virtual void matchMultiple(
        const core::Frame& query,
        const std::vector<core::Frame>& candidates,
        std::vector<std::vector<Match>>& all_matches,
        float ratio_threshold = 0.75f
    ) {
        all_matches.resize(candidates.size());
        for (size_t i = 0; i < candidates.size(); i++) {
            match(query, candidates[i], all_matches[i], ratio_threshold);
        }
    }
};

using MatcherPtr = std::unique_ptr<IMatcher>;

} // namespace aria::interfaces
```

### ILoopDetector.hpp

```cpp
#pragma once
#include "core/KeyFrame.hpp"
#include "IMatcher.hpp"
#include <optional>

namespace aria::interfaces {

struct LoopCandidate {
    uint64_t query_id;
    uint64_t match_id;
    double score;
    std::vector<Match> matches;
    Eigen::Matrix4d relative_pose;
};

class ILoopDetector {
public:
    virtual ~ILoopDetector() = default;

    // Add keyframe to database
    virtual void addKeyFrame(const core::KeyFrame& kf) = 0;

    // Detect loop closure
    // @param query Current keyframe
    // @return Loop candidate if found, nullopt otherwise
    virtual std::optional<LoopCandidate> detect(const core::KeyFrame& query) = 0;

    // Get number of detected loops
    virtual int getLoopCount() const = 0;

    // Configuration
    virtual void setMinFramesBetween(int n) = 0;
    virtual void setMinScore(double s) = 0;
    virtual void setMinMatches(int n) = 0;
};

using LoopDetectorPtr = std::unique_ptr<ILoopDetector>;

} // namespace aria::interfaces
```

### IObjectDetector.hpp

```cpp
#pragma once
#include <vector>
#include <string>

namespace aria::interfaces {

struct Detection {
    float x1, y1, x2, y2;   // Bounding box
    float confidence;
    int class_id;
    std::string class_name;
};

class IObjectDetector {
public:
    virtual ~IObjectDetector() = default;

    // Detect objects in image
    // @param image_data RGB image data (row-major, 3 channels)
    // @param width Image width
    // @param height Image height
    // @param detections Output detections
    // @param conf_threshold Confidence threshold
    // @param nms_threshold NMS IoU threshold
    virtual void detect(
        const uint8_t* image_data,
        int width,
        int height,
        std::vector<Detection>& detections,
        float conf_threshold = 0.5f,
        float nms_threshold = 0.45f
    ) = 0;

    // Async detection
    virtual void detectAsync(
        const uint8_t* image_data,
        int width,
        int height
    ) = 0;

    // Get results after async detection
    virtual void getDetections(
        std::vector<Detection>& detections,
        float conf_threshold = 0.5f,
        float nms_threshold = 0.45f
    ) = 0;

    virtual void sync() = 0;
};

using ObjectDetectorPtr = std::unique_ptr<IObjectDetector>;

} // namespace aria::interfaces
```

### ISensorFusion.hpp

```cpp
#pragma once
#include "core/Pose.hpp"
#include <Eigen/Dense>

namespace aria::interfaces {

struct ImuMeasurement {
    double timestamp;
    Eigen::Vector3d accel;      // m/s^2
    Eigen::Vector3d gyro;       // rad/s
};

class ISensorFusion {
public:
    virtual ~ISensorFusion() = default;

    // IMU prediction step (high frequency: 200Hz)
    virtual void predictIMU(const ImuMeasurement& imu) = 0;

    // Visual odometry update step (low frequency: 30Hz)
    virtual void updateVO(const core::Pose& vo_pose) = 0;

    // Get current fused state
    virtual core::Pose getFusedPose() const = 0;

    // Get velocity estimate
    virtual Eigen::Vector3d getVelocity() const = 0;

    // Reset filter
    virtual void reset() = 0;
    virtual void reset(const core::Pose& initial_pose) = 0;
};

using SensorFusionPtr = std::unique_ptr<ISensorFusion>;

} // namespace aria::interfaces
```

### IMapper.hpp

```cpp
#pragma once
#include "core/Frame.hpp"
#include "core/MapPoint.hpp"
#include "core/Pose.hpp"
#include "IMatcher.hpp"
#include <vector>
#include <string>

namespace aria::interfaces {

class IMapper {
public:
    virtual ~IMapper() = default;

    // Triangulate new map points from matched frames
    // @param frame1 First frame with pose
    // @param frame2 Second frame with pose
    // @param matches Matches between frames
    // @param K Camera intrinsic matrix (3x3)
    // @param new_points Output: newly created map points
    virtual void triangulate(
        const core::Frame& frame1,
        const core::Frame& frame2,
        const core::Pose& pose1,
        const core::Pose& pose2,
        const std::vector<Match>& matches,
        const Eigen::Matrix3d& K,
        std::vector<core::MapPoint>& new_points
    ) = 0;

    // Get all map points
    virtual const std::vector<core::MapPoint>& getMapPoints() const = 0;

    // Export to file
    virtual void exportPLY(const std::string& filename) const = 0;
    virtual void exportPCD(const std::string& filename) const = 0;

    // Clear map
    virtual void clear() = 0;

    // Statistics
    virtual size_t size() const = 0;
};

using MapperPtr = std::unique_ptr<IMapper>;

} // namespace aria::interfaces
```

## GPU Adapters

Implementations using CUDA and TensorRT.

### OrbCudaExtractor.hpp

```cpp
#pragma once
#include "interfaces/IFeatureExtractor.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include <cuda_runtime.h>

namespace aria::adapters::gpu {

class OrbCudaExtractor : public interfaces::IFeatureExtractor {
public:
    explicit OrbCudaExtractor(int max_features = 1000, cudaStream_t stream = nullptr);
    ~OrbCudaExtractor() override;

    void extract(
        const uint8_t* image_data,
        int width,
        int height,
        core::Frame& frame
    ) override;

    void extractAsync(
        const uint8_t* image_data,
        int width,
        int height,
        core::Frame& frame
    ) override;

    void sync() override;

    void setMaxFeatures(int n) override;
    int getMaxFeatures() const override { return max_features_; }

    // GPU-specific: get descriptors without download (for GPU matching)
    const cv::cuda::GpuMat& getGpuDescriptors() const { return gpu_descriptors_; }

private:
    cv::Ptr<cv::cuda::ORB> orb_;
    cv::cuda::GpuMat gpu_image_;
    cv::cuda::GpuMat gpu_keypoints_;
    cv::cuda::GpuMat gpu_descriptors_;
    cv::cuda::Stream cv_stream_;
    cudaStream_t cuda_stream_;
    int max_features_;
    bool owns_stream_;
};

} // namespace aria::adapters::gpu
```

### CudaMatcher.hpp

```cpp
#pragma once
#include "interfaces/IMatcher.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include <cuda_runtime.h>

namespace aria::adapters::gpu {

class CudaMatcher : public interfaces::IMatcher {
public:
    explicit CudaMatcher(cudaStream_t stream = nullptr);
    ~CudaMatcher() override;

    void match(
        const core::Frame& query,
        const core::Frame& train,
        std::vector<interfaces::Match>& matches,
        float ratio_threshold = 0.75f
    ) override;

    // GPU-to-GPU matching (zero-copy when used with OrbCudaExtractor)
    void matchGpu(
        const cv::cuda::GpuMat& query_desc,
        const cv::cuda::GpuMat& train_desc,
        std::vector<interfaces::Match>& matches,
        float ratio_threshold = 0.75f
    );

private:
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher_;
    cv::cuda::Stream cv_stream_;
    cudaStream_t cuda_stream_;
    bool owns_stream_;
};

} // namespace aria::adapters::gpu
```

### YoloTrtDetector.hpp

```cpp
#pragma once
#include "interfaces/IObjectDetector.hpp"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>

namespace aria::adapters::gpu {

class YoloTrtDetector : public interfaces::IObjectDetector {
public:
    explicit YoloTrtDetector(const std::string& engine_path, cudaStream_t stream = nullptr);
    ~YoloTrtDetector() override;

    void detect(
        const uint8_t* image_data,
        int width,
        int height,
        std::vector<interfaces::Detection>& detections,
        float conf_threshold = 0.5f,
        float nms_threshold = 0.45f
    ) override;

    void detectAsync(
        const uint8_t* image_data,
        int width,
        int height
    ) override;

    void getDetections(
        std::vector<interfaces::Detection>& detections,
        float conf_threshold = 0.5f,
        float nms_threshold = 0.45f
    ) override;

    void sync() override;

private:
    void preprocess(const uint8_t* image_data, int width, int height);
    void postprocess(std::vector<interfaces::Detection>& detections,
                     float conf_threshold, float nms_threshold);

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_;
    bool owns_stream_;

    // Buffers
    void* buffers_[2];
    float* output_host_;
    int input_h_, input_w_;
    int output_size_;
};

} // namespace aria::adapters::gpu
```

## Application Layer (Pipeline)

Orchestrates components using dependency injection.

### SlamPipeline.hpp

```cpp
#pragma once
#include "interfaces/IFeatureExtractor.hpp"
#include "interfaces/IMatcher.hpp"
#include "interfaces/ILoopDetector.hpp"
#include "interfaces/IObjectDetector.hpp"
#include "interfaces/ISensorFusion.hpp"
#include "interfaces/IMapper.hpp"
#include "core/Pose.hpp"
#include <memory>
#include <functional>

namespace aria::pipeline {

struct PipelineConfig {
    bool enable_loop_closure = true;
    bool enable_object_detection = true;
    bool enable_mapping = true;
    bool filter_dynamic_objects = true;

    // Camera intrinsics
    double fx = 700, fy = 700;
    double cx = 320, cy = 180;
};

class SlamPipeline {
public:
    // Dependency injection via constructor
    SlamPipeline(
        interfaces::FeatureExtractorPtr extractor,
        interfaces::MatcherPtr matcher,
        interfaces::LoopDetectorPtr loop_detector,
        interfaces::ObjectDetectorPtr object_detector,
        interfaces::SensorFusionPtr sensor_fusion,
        interfaces::MapperPtr mapper,
        const PipelineConfig& config = {}
    );

    ~SlamPipeline();

    // Process single frame
    // @param image_data RGB image data
    // @param width Image width
    // @param height Image height
    // @param timestamp Frame timestamp
    // @return Current pose estimate
    core::Pose processFrame(
        const uint8_t* image_data,
        int width,
        int height,
        double timestamp
    );

    // Process IMU measurement
    void processIMU(const interfaces::ImuMeasurement& imu);

    // Get current state
    core::Pose getCurrentPose() const;
    const std::vector<core::Pose>& getTrajectory() const;
    const interfaces::IMapper& getMapper() const;

    // Callbacks for external consumers
    using PoseCallback = std::function<void(const core::Pose&)>;
    using LoopCallback = std::function<void(const interfaces::LoopCandidate&)>;

    void setPoseCallback(PoseCallback cb) { pose_callback_ = std::move(cb); }
    void setLoopCallback(LoopCallback cb) { loop_callback_ = std::move(cb); }

private:
    // Components (injected)
    interfaces::FeatureExtractorPtr extractor_;
    interfaces::MatcherPtr matcher_;
    interfaces::LoopDetectorPtr loop_detector_;
    interfaces::ObjectDetectorPtr object_detector_;
    interfaces::SensorFusionPtr sensor_fusion_;
    interfaces::MapperPtr mapper_;

    // Configuration
    PipelineConfig config_;
    Eigen::Matrix3d K_;  // Camera intrinsics

    // State
    std::unique_ptr<core::Frame> prev_frame_;
    core::Pose current_pose_;
    std::vector<core::Pose> trajectory_;
    uint64_t frame_id_ = 0;

    // Callbacks
    PoseCallback pose_callback_;
    LoopCallback loop_callback_;

    // Internal methods
    void filterDynamicKeypoints(
        core::Frame& frame,
        const std::vector<interfaces::Detection>& detections
    );

    core::Pose estimatePose(
        const core::Frame& prev,
        const core::Frame& curr,
        const std::vector<interfaces::Match>& matches
    );
};

} // namespace aria::pipeline
```

## Factory (Dependency Injection)

Create pipeline with different configurations.

### PipelineFactory.hpp

```cpp
#pragma once
#include "pipeline/SlamPipeline.hpp"
#include <string>

namespace aria::factory {

enum class ExecutionMode {
    GPU,        // Full GPU acceleration (production)
    CPU,        // CPU-only (debugging, profiling)
    MOCK        // Mock components (unit testing)
};

struct FactoryConfig {
    ExecutionMode mode = ExecutionMode::GPU;

    // GPU settings
    std::string yolo_engine_path = "../models/yolo26s.engine";
    int cuda_device = 0;

    // Feature extraction
    int max_features = 1000;

    // Pipeline config
    pipeline::PipelineConfig pipeline_config;
};

class PipelineFactory {
public:
    static std::unique_ptr<pipeline::SlamPipeline> create(const FactoryConfig& config);

    // Convenience methods
    static std::unique_ptr<pipeline::SlamPipeline> createGpu(
        const std::string& yolo_engine = "../models/yolo26s.engine"
    );

    static std::unique_ptr<pipeline::SlamPipeline> createCpu();

    static std::unique_ptr<pipeline::SlamPipeline> createMock();
};

} // namespace aria::factory
```

### Usage Example

```cpp
#include "factory/PipelineFactory.hpp"

int main() {
    // Production: full GPU
    auto pipeline = aria::factory::PipelineFactory::createGpu();

    // Or with custom config
    aria::factory::FactoryConfig config;
    config.mode = aria::factory::ExecutionMode::GPU;
    config.max_features = 2000;
    config.pipeline_config.filter_dynamic_objects = true;

    auto custom_pipeline = aria::factory::PipelineFactory::create(config);

    // Process frames
    while (auto frame = capture.getFrame()) {
        auto pose = pipeline->processFrame(
            frame.data, frame.width, frame.height, frame.timestamp
        );
        std::cout << "Position: " << pose.position.transpose() << std::endl;
    }

    // Export map
    pipeline->getMapper().exportPLY("map.ply");
}
```

## Testing with Mocks

```cpp
#include "factory/PipelineFactory.hpp"
#include <gtest/gtest.h>

TEST(SlamPipeline, ProcessFrameReturnsPose) {
    // Create pipeline with mock components
    auto pipeline = aria::factory::PipelineFactory::createMock();

    // Create test image
    std::vector<uint8_t> test_image(640 * 480 * 3, 128);

    // Process frame
    auto pose = pipeline->processFrame(test_image.data(), 640, 480, 0.0);

    // Verify pose is valid
    EXPECT_FALSE(pose.position.hasNaN());
    EXPECT_NEAR(pose.orientation.norm(), 1.0, 1e-6);
}
```

## SOLID Principles Summary

| Principle | How Applied |
|-----------|-------------|
| **S**ingle Responsibility | `OrbCudaExtractor` only extracts, `CudaMatcher` only matches |
| **O**pen/Closed | Add `SuperPointExtractor` without modifying `SlamPipeline` |
| **L**iskov Substitution | `CudaMatcher` and `BFMatcher` are interchangeable |
| **I**nterface Segregation | `IFeatureExtractor` != `IMatcher` != `IObjectDetector` |
| **D**ependency Inversion | `SlamPipeline` depends on `IFeatureExtractor`, not `OrbCudaExtractor` |

## Migration Plan

1. **Create interfaces** in `include/interfaces/` (no code changes)
2. **Create domain entities** in `include/core/` (copy existing structs)
3. **Wrap existing code** in adapters (minimal changes)
4. **Create SlamPipeline** that uses interfaces
5. **Update main.cpp** to use factory
6. **Add tests** with mocks

Each step is a separate commit, maintaining a working build throughout.

## File Structure After H12

```
include/
├── core/
│   ├── Frame.hpp
│   ├── KeyFrame.hpp
│   ├── MapPoint.hpp
│   └── Pose.hpp
├── interfaces/
│   ├── IFeatureExtractor.hpp
│   ├── IMatcher.hpp
│   ├── ILoopDetector.hpp
│   ├── IObjectDetector.hpp
│   ├── ISensorFusion.hpp
│   └── IMapper.hpp
├── adapters/
│   ├── gpu/
│   │   ├── OrbCudaExtractor.hpp
│   │   ├── CudaMatcher.hpp
│   │   └── YoloTrtDetector.hpp
│   ├── cpu/
│   │   ├── OrbCpuExtractor.hpp
│   │   └── BruteForceMatcher.hpp
│   └── sensors/
│       └── EuRoCReader.hpp
├── pipeline/
│   └── SlamPipeline.hpp
└── factory/
    └── PipelineFactory.hpp
```

## Next Steps

After H12 is complete:
- **H13**: Add `LoopClosureThread` for async loop detection
- **H14**: Migrate loop closure matching to GPU
- **H16**: Add GoogleTest with mock-based unit tests
