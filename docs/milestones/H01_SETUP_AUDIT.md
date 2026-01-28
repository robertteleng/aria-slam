# Auditoría Técnica: H01 - Setup + Build System

**Proyecto:** aria-slam (C++)
**Milestone:** H01 - Configuración del entorno y sistema de build
**Fecha:** 2025-01
**Autor:** Roberto (preparación entrevista técnica)

---

## 0. RESUMEN EJECUTIVO

### Objetivo
Configurar el sistema de build con CMake para un proyecto SLAM con GPU acceleration, incluyendo:
- OpenCV con CUDA
- TensorRT para inferencia
- Eigen para álgebra lineal
- g2o para optimización de grafos

### Resultado Final
Pipeline compilado que ejecuta a **77+ FPS** en RTX 2060.

---

## 1. ANÁLISIS DEL CMakeLists.txt

### Archivo: `CMakeLists.txt` (líneas 1-78)

```cmake
cmake_minimum_required(VERSION 3.16)
project(aria_slam)

set(CMAKE_CXX_STANDARD 17)
```

**Conceptos clave:**
- **CMake 3.16**: Versión mínima para `find_package(CUDA)` moderno y policies actuales
- **C++17**: Requerido para `std::optional`, structured bindings, `if constexpr`, etc.

### 1.1 Configuración de OpenCV CUDA

```cmake
# CMakeLists.txt:5-6
set(OpenCV_DIR "/home/roberto/libs/opencv_cuda/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
```

**¿Por qué path explícito?**
- OpenCV del sistema (`apt install libopencv-dev`) **no tiene CUDA**
- Compilamos OpenCV custom con `CUDA_ARCH_BIN=7.5` (RTX 2060)
- El path apunta a nuestra instalación con módulos CUDA

**Verificación en código (`main.cpp:73-78`):**
```cpp
int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
if (cuda_devices == 0) {
    std::cerr << "Error: No CUDA devices found!" << std::endl;
    return -1;
}
```

### 1.2 Configuración de TensorRT

```cmake
# CMakeLists.txt:10-13
set(TensorRT_DIR "/home/roberto/libs/TensorRT-10.7.0.23")
set(TensorRT_INCLUDE_DIRS "${TensorRT_DIR}/include")
set(TensorRT_LIBS "${TensorRT_DIR}/lib")
```

**¿Por qué no usar `find_package(TensorRT)`?**
- TensorRT no incluye archivo `TensorRTConfig.cmake` oficial
- Instalación manual desde tarball de NVIDIA
- Paths hardcodeados son el approach estándar

**Linkado:**
```cmake
# CMakeLists.txt:37
target_link_libraries(aria_slam
    nvinfer        # TensorRT inference engine
    cudart         # CUDA runtime
)
```

### 1.3 Includes y Libraries

```cmake
# CMakeLists.txt:15-24
include_directories(
    include                    # Headers del proyecto
    ${OpenCV_INCLUDE_DIRS}     # OpenCV headers
    ${CUDA_INCLUDE_DIRS}       # cuda_runtime.h, etc.
    ${TensorRT_INCLUDE_DIRS}   # NvInfer.h
    ${EIGEN3_INCLUDE_DIR}      # Eigen/Dense
)
```

**Orden de includes importa:**
1. `include/` primero para preferir headers locales
2. Dependencias externas después

### 1.4 Definición de Ejecutables

```cmake
# CMakeLists.txt:25-32
add_executable(aria_slam
    src/main.cpp
    src/Frame.cpp
    src/TRTInference.cpp
    src/IMU.cpp
    src/LoopClosure.cpp
    src/Mapper.cpp
)
```

**Arquitectura de archivos:**
```
aria-slam/
├── include/           # Headers (.hpp)
│   ├── Frame.hpp
│   ├── TRTInference.hpp
│   ├── IMU.hpp
│   ├── LoopClosure.hpp
│   └── Mapper.hpp
├── src/               # Implementaciones (.cpp)
│   ├── main.cpp
│   ├── Frame.cpp
│   └── ...
└── CMakeLists.txt
```

---

## 2. DEPENDENCIAS EN DETALLE

### Tabla de Dependencias

| Biblioteca | Versión | Propósito | Líneas CMake |
|------------|---------|-----------|--------------|
| CMake | >= 3.16 | Build system | 1 |
| OpenCV | 4.9.0 + CUDA | Visión por computador | 5-6 |
| CUDA | 12.x | GPU computing | 7 |
| Eigen3 | 3.4.0 | Álgebra lineal | 8 |
| TensorRT | 10.7.0.23 | Inferencia YOLO | 10-13 |
| g2o | 20230223 | Pose graph optimization | 40-43 |

### 2.1 Eigen3

```cmake
find_package(Eigen3 REQUIRED)
target_link_libraries(aria_slam Eigen3::Eigen)
```

**¿Por qué Eigen?**
- Header-only (zero runtime overhead)
- SIMD optimizado (SSE, AVX)
- Expresiones lazy para evitar temporales
- Estándar de facto en robotics/SLAM

**Uso típico (`IMU.hpp:1`):**
```cpp
#include <Eigen/Dense>
Eigen::Vector3d position;
Eigen::Quaterniond orientation;
Eigen::Matrix4d pose;
```

### 2.2 g2o (Graph Optimization)

```cmake
# CMakeLists.txt:40-43
target_link_libraries(aria_slam
    g2o_core
    g2o_stuff
    g2o_types_slam3d
    g2o_solver_eigen
)
```

**Componentes g2o:**
- `g2o_core`: Optimizador sparse
- `g2o_stuff`: Utilities
- `g2o_types_slam3d`: Vértices SE3, edges
- `g2o_solver_eigen`: Solver lineal con Eigen

---

## 3. PATRONES DE BUILD

### 3.1 Múltiples Ejecutables

```cmake
# Ejecutable principal
add_executable(aria_slam src/main.cpp ...)

# Evaluación con dataset EuRoC
add_executable(euroc_eval src/euroc_eval.cpp ...)

# Benchmarks experimentales
add_executable(benchmark_imu experiments/benchmark_imu.cpp ...)
```

**¿Por qué separar ejecutables?**
1. `aria_slam`: Uso con video/webcam
2. `euroc_eval`: Benchmarking contra ground truth
3. `benchmark_imu`: Tests aislados de componentes

### 3.2 Reutilización de Código

Los ejecutables comparten las mismas fuentes:
```cmake
add_executable(aria_slam
    src/main.cpp
    src/Frame.cpp        # Compartido
    src/TRTInference.cpp # Compartido
    src/IMU.cpp          # Compartido
)

add_executable(euroc_eval
    src/euroc_eval.cpp
    src/Frame.cpp        # Reutilizado
    src/TRTInference.cpp # Reutilizado
    src/IMU.cpp          # Reutilizado
)
```

**Alternativa (mejor):** Crear una library estática:
```cmake
# Posible mejora futura
add_library(aria_core STATIC
    src/Frame.cpp
    src/TRTInference.cpp
    src/IMU.cpp
)

add_executable(aria_slam src/main.cpp)
target_link_libraries(aria_slam aria_core)
```

---

## 4. CONCEPTOS C++ UTILIZADOS

### 4.1 C++17 Features

```cpp
// main.cpp: structured bindings
for (const auto& [id, score] : candidates) { ... }

// Frame.cpp: std::optional podría usarse para async results
std::optional<cv::Mat> getDescriptorsIfReady();

// if constexpr (compilación condicional)
if constexpr (USE_GPU) {
    orb_gpu->detect(...);
} else {
    orb_cpu->detect(...);
}
```

### 4.2 Include Guards vs #pragma once

```cpp
// Frame.hpp:1
#pragma once
```

**`#pragma once` vs `#ifndef`:**
| Aspecto | `#pragma once` | `#ifndef` guards |
|---------|----------------|------------------|
| Estándar | No (pero universal) | Sí (C++98) |
| Portabilidad | 99.9% compiladores | 100% |
| Errores | Ninguno | Posible typo en macro |
| Performance | Más rápido* | Ligeramente más lento |

*Algunos compiladores optimizan `#pragma once` evitando re-leer el archivo.

---

## 5. SETUP DE MÁQUINA

### 5.1 Script de Setup Automático

`scripts/setup_machine.sh` automatiza:
1. Detecta GPU (CUDA compute capability)
2. Localiza TensorRT y OpenCV CUDA
3. Actualiza CMakeLists.txt con paths correctos
4. Genera engine YOLO para la GPU detectada

```bash
# Detección de GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)

# Mapeo GPU -> SM
case "$GPU_NAME" in
    *"RTX 2060"*) SM="75" ;;
    *"RTX 3080"*) SM="86" ;;
    *"RTX 4090"*) SM="89" ;;
esac
```

### 5.2 Verificación de Instalación

```bash
# Build
mkdir build && cd build
cmake ..
make -j8

# Test
./aria_slam --headless  # Sin GUI
./aria_slam             # Con visualización
```

---

## 6. PREGUNTAS DE ENTREVISTA

### Q1: ¿Por qué usar CMake en lugar de Makefile directo?

**R:**
1. **Portabilidad**: CMake genera Makefiles, Ninja, VS projects, etc.
2. **Find modules**: `find_package(OpenCV)` maneja paths automáticamente
3. **Dependencias transitivas**: `target_link_libraries` propaga includes
4. **Out-of-source builds**: Código fuente limpio, builds aislados

### Q2: ¿Qué significa `find_package(OpenCV REQUIRED)`?

**R:**
- Busca `OpenCVConfig.cmake` en paths estándar
- `REQUIRED` hace que falle si no encuentra
- Define variables: `OpenCV_INCLUDE_DIRS`, `OpenCV_LIBS`
- Puede especificar componentes: `find_package(OpenCV REQUIRED core imgproc cudafeatures2d)`

### Q3: ¿Por qué C++17 y no C++20?

**R:**
1. **Compatibilidad CUDA**: nvcc tiene soporte limitado de C++20
2. **g2o/Eigen**: Builds estables con C++17
3. **C++17 suficiente**: structured bindings, `std::optional`, `if constexpr`

### Q4: ¿Cómo manejarías dependencias en un entorno de CI/CD?

**R:**
```cmake
# Opción 1: Paths condicionales
if(DEFINED ENV{CI})
    set(OpenCV_DIR "/opt/opencv-cuda")
else()
    set(OpenCV_DIR "$ENV{HOME}/libs/opencv_cuda")
endif()

# Opción 2: CMake presets (CMake 3.21+)
# CMakePresets.json con configuraciones por entorno
```

### Q5: ¿Qué es `target_link_libraries` vs `link_directories`?

**R:**
```cmake
# Moderno (preferido): Por target, propaga dependencias
target_link_libraries(aria_slam ${OpenCV_LIBS})

# Legacy: Global, no propaga
link_directories(/path/to/libs)
```

`target_link_libraries` es preferido porque:
- Scope por target (no contamina otros targets)
- Propaga include dirs transitivamente
- Mejor para builds paralelos

### Q6: ¿Cómo depurarías un error de linkado con TensorRT?

**R:**
```bash
# 1. Verificar que el .so existe
ls -la /home/roberto/libs/TensorRT-10.7.0.23/lib/libnvinfer.so

# 2. Verificar símbolos
nm -D libnvinfer.so | grep createInferRuntime

# 3. Verificar rpath
ldd ./aria_slam | grep nvinfer

# 4. Si falta, añadir al CMakeLists.txt:
set(CMAKE_INSTALL_RPATH "${TensorRT_LIBS}")
```

---

## 7. DIAGRAMA DE DEPENDENCIAS

```
                    ┌─────────────────────────────────────────────┐
                    │              aria_slam                      │
                    │         (src/main.cpp)                      │
                    └─────────────────────────────────────────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬─────────────────┐
        │               │               │               │                 │
        ▼               ▼               ▼               ▼                 ▼
   ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌─────────────┐
   │ Frame   │    │TRTInfer  │    │   IMU   │    │LoopClose │    │   Mapper    │
   │ .hpp/.cpp│   │ .hpp/.cpp│    │.hpp/.cpp│    │.hpp/.cpp │    │ .hpp/.cpp   │
   └────┬────┘    └────┬─────┘    └────┬────┘    └────┬─────┘    └──────┬──────┘
        │              │               │               │                 │
        ▼              ▼               ▼               ▼                 ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                           External Libraries                            │
   ├─────────────┬─────────────┬───────────────┬──────────────┬─────────────┤
   │ OpenCV CUDA │  TensorRT   │    Eigen3     │     g2o      │    CUDA     │
   │  (cv::cuda) │  (nvinfer)  │ (Vector,Mat)  │ (VertexSE3)  │  (runtime)  │
   └─────────────┴─────────────┴───────────────┴──────────────┴─────────────┘
```

---

## 8. CHECKLIST DE PREPARACIÓN

- [ ] Entender estructura de CMakeLists.txt
- [ ] Saber explicar `find_package` vs paths manuales
- [ ] Conocer diferencias C++17 vs C++20 para CUDA
- [ ] Entender `target_link_libraries` transitivo
- [ ] Poder depurar errores de linkado
- [ ] Saber crear múltiples ejecutables con código compartido
- [ ] Entender por qué OpenCV necesita compilación custom para CUDA

---

**Generado:** 2025-01
**Proyecto:** aria-slam (C++)
