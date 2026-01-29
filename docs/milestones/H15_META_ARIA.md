# H15: Meta Aria Integration

**Status:** 📋 Planned
**Dependencies:** H12 (Clean Architecture)

---

## Objetivo

Conectar las gafas Meta Aria al sistema SLAM para streaming en tiempo real de:
- 3 cámaras (RGB + 2 SLAM)
- IMU a 1000 Hz
- Calibración de cámaras

## Arquitectura

### El Problema: SDK solo en Python

El Meta Aria SDK (`aria.sdk`) **solo tiene API en Python** para streaming en vivo.
Las herramientas C++ (`projectaria_tools`) solo procesan archivos VRS grabados.

### La Solución: pybind11 Embedding

```
┌────────────────────────────────────────────────────────────────┐
│                       aria-slam (C++)                          │
│                                                                │
│  ┌──────────────────┐      ┌────────────────────────────────┐ │
│  │  SlamPipeline    │◄─────│  AriaDeviceAdapter             │ │
│  │  (C++ puro)      │      │  (pybind11 embedding)          │ │
│  └──────────────────┘      │                                │ │
│                            │  ┌──────────────────────────┐  │ │
│                            │  │  Python Interpreter      │  │ │
│                            │  │  ┌────────────────────┐  │  │ │
│                            │  │  │  import aria.sdk   │  │  │ │
│                            │  │  │  device.connect()  │  │  │ │
│                            │  │  │  start_streaming() │  │  │ │
│                            │  │  └────────────────────┘  │  │ │
│                            │  └──────────────────────────┘  │ │
│                            └────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

**Ventajas:**
- Un solo proceso
- Latencia mínima (llamadas directas)
- Código SLAM 100% C++
- Python solo para SDK de hardware

## Interface Definida

```cpp
// include/interfaces/IAriaDevice.hpp

enum class AriaCamera {
    RGB,        // 1408x1408 fisheye, centro
    SLAM_LEFT,  // 640x480 fisheye, izquierda
    SLAM_RIGHT  // 640x480 fisheye, derecha
};

struct AriaImage {
    const uint8_t* data;
    int width, height, channels;
    uint64_t timestamp_ns;
    AriaCamera camera;
};

class IAriaDevice {
public:
    virtual bool connect(const std::string& ip = "") = 0;  // USB o WiFi
    virtual void disconnect() = 0;
    virtual bool startStreaming(int cameras = 0x7) = 0;
    virtual void stopStreaming() = 0;

    virtual void setImageCallback(ImageCallback cb) = 0;
    virtual void setImuCallback(ImuCallback cb) = 0;

    virtual bool getCalibration(AriaCamera cam, float& fx, float& fy, float& cx, float& cy) = 0;
    virtual void spinOnce() = 0;
};
```

## Implementación: AriaDeviceAdapter

### Estructura

```
src/adapters/hardware/
├── AriaDeviceAdapter.hpp
├── AriaDeviceAdapter.cpp     # pybind11 embedding
└── MockAriaDevice.cpp        # Para testing sin hardware
```

### Código Principal

```cpp
// src/adapters/hardware/AriaDeviceAdapter.cpp

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

class AriaDeviceAdapter : public IAriaDevice {
public:
    AriaDeviceAdapter() {
        // Inicializar Python una vez
        if (!Py_IsInitialized()) {
            py::initialize_interpreter();
        }
    }

    bool connect(const std::string& ip) override {
        try {
            // Importar SDK
            py::module aria = py::module::import("aria.sdk");

            // Crear cliente
            device_ = aria.attr("DeviceClient")();

            // Conectar (USB o WiFi)
            if (ip.empty()) {
                device_.attr("connect")();
            } else {
                device_.attr("connect")(ip);
            }

            connected_ = true;
            return true;
        } catch (const py::error_already_set& e) {
            std::cerr << "Aria connection error: " << e.what() << std::endl;
            return false;
        }
    }

    bool startStreaming(int cameras) override {
        // Configurar streaming profile
        py::module aria = py::module::import("aria.sdk");
        auto config = aria.attr("StreamingConfig")();
        config.attr("profile_name") = "profile28";  // RGB + SLAM @ 30fps

        // Registrar callbacks Python que llaman a C++
        streaming_manager_ = device_.attr("streaming_manager")();

        // Observer pattern para recibir frames
        observer_ = py::module::import("aria_observer").attr("Observer")(
            py::cpp_function([this](py::object img, int cam_id, uint64_t ts) {
                onImageReceived(img, cam_id, ts);
            }),
            py::cpp_function([this](py::array_t<double> accel, uint64_t ts) {
                onImuReceived(accel, ts);
            })
        );

        streaming_manager_.attr("set_observer")(observer_);
        streaming_manager_.attr("start_streaming")(config);

        return true;
    }

    void spinOnce() override {
        // Procesar eventos pendientes de Python
        py::gil_scoped_acquire acquire;
        // Los callbacks se disparan automáticamente
    }

private:
    void onImageReceived(py::object img, int camera_id, uint64_t timestamp) {
        if (!image_callback_) return;

        // Convertir numpy array a puntero
        py::array_t<uint8_t> arr = img.cast<py::array_t<uint8_t>>();
        auto buf = arr.request();

        AriaImage aria_img;
        aria_img.data = static_cast<uint8_t*>(buf.ptr);
        aria_img.width = buf.shape[1];
        aria_img.height = buf.shape[0];
        aria_img.channels = (buf.ndim == 3) ? buf.shape[2] : 1;
        aria_img.timestamp_ns = timestamp;
        aria_img.camera = static_cast<AriaCamera>(camera_id);

        image_callback_(aria_img);
    }

    void onImuReceived(py::array_t<double> accel, uint64_t timestamp) {
        if (!imu_callback_) return;

        auto buf = accel.request();
        double* ptr = static_cast<double*>(buf.ptr);

        core::ImuMeasurement imu;
        imu.timestamp = timestamp * 1e-9;  // ns -> s
        imu.accel = Eigen::Vector3d(ptr[0], ptr[1], ptr[2]);
        // gyro viene en otro callback o se interpola

        imu_callback_(imu);
    }

    py::object device_;
    py::object streaming_manager_;
    py::object observer_;
    ImageCallback image_callback_;
    ImuCallback imu_callback_;
    bool connected_ = false;
};
```

## CMake Configuration

```cmake
# CMakeLists.txt

# Encontrar Python y pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# Agregar adapter
add_library(aria_adapter
    src/adapters/hardware/AriaDeviceAdapter.cpp
)

target_link_libraries(aria_adapter
    pybind11::embed
    Python3::Python
)

# Main executable
target_link_libraries(aria_slam
    aria_adapter
    # ... otras libs
)
```

## Dependencias

### Sistema
```bash
# Ubuntu
sudo apt install python3-dev pybind11-dev

# O via pip
pip install pybind11
```

### Python packages (para el SDK)
```bash
pip install projectaria-client-sdk
pip install projectaria-tools
```

## Datos de las Cámaras

| Cámara | Resolución | FOV | FPS | Uso |
|--------|------------|-----|-----|-----|
| RGB | 1408×1408 | 110° fisheye | 30 | Detección objetos, features |
| SLAM_LEFT | 640×480 | 150° fisheye | 30 | Stereo, periférico izq |
| SLAM_RIGHT | 640×480 | 150° fisheye | 30 | Stereo, periférico dcha |

### Calibración

El SDK proporciona calibración intrinseca/extrinseca:
- Modelo de distorsión fisheye (KB4)
- Matriz de rotación entre cámaras
- Baseline para stereo

## Testing

### MockAriaDevice

```cpp
class MockAriaDevice : public IAriaDevice {
public:
    bool connect(const std::string&) override { return true; }

    bool startStreaming(int) override {
        // Leer imágenes de disco para simular streaming
        streaming_ = true;
        return true;
    }

    void spinOnce() override {
        if (!streaming_ || !image_callback_) return;

        // Simular frame cada 33ms
        static auto last = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (now - last > std::chrono::milliseconds(33)) {
            AriaImage fake;
            fake.data = test_image_.data();
            fake.width = 640;
            fake.height = 480;
            // ...
            image_callback_(fake);
            last = now;
        }
    }
};
```

## Integración con SlamPipeline

```cpp
int main() {
    // Crear componentes
    auto aria = std::make_unique<AriaDeviceAdapter>();
    auto pipeline = PipelineFactory::createGpu();

    // Callback: cada frame va al pipeline
    aria->setImageCallback([&](const AriaImage& img) {
        if (img.camera == AriaCamera::RGB) {
            auto pose = pipeline->processFrame(
                img.data, img.width, img.height,
                img.timestamp_ns * 1e-9
            );
            // Usar pose...
        }
    });

    // Conectar y streamear
    aria->connect();  // USB
    aria->startStreaming();

    // Loop principal
    while (running) {
        aria->spinOnce();
    }

    aria->stopStreaming();
    aria->disconnect();
}
```

## Checklist

- [ ] Instalar pybind11 y Python dev headers
- [ ] Crear AriaDeviceAdapter con embedding
- [ ] Crear helper Python para observer pattern
- [ ] Testear conexión USB
- [ ] Testear conexión WiFi
- [ ] Integrar con SlamPipeline
- [ ] Crear MockAriaDevice para testing
- [ ] Documentar calibración fisheye

## Referencias

- [Project Aria Tools](https://github.com/facebookresearch/projectaria_tools)
- [Client SDK Docs](https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk)
- [pybind11 Embedding](https://pybind11.readthedocs.io/en/stable/advanced/embedding.html)
