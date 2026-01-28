#!/bin/bash
# =============================================================================
# ARIA-SLAM Machine Setup Script
# =============================================================================
# Configura automáticamente el entorno para una nueva máquina:
# 1. Detecta GPU y configura rutas
# 2. Instala/compila OpenCV con CUDA si es necesario
# 3. Descarga TensorRT si es necesario
# 4. Descarga modelo YOLO y genera engine
# 5. Actualiza CMakeLists.txt
#
# Uso: ./scripts/setup_machine.sh [--full | --config-only | --engine-only]
#   --full:        Instalación completa (OpenCV, TensorRT, modelo)
#   --config-only: Solo actualiza CMakeLists.txt con rutas detectadas
#   --engine-only: Solo genera el engine YOLO
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LIBS_DIR="$HOME/libs"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# DETECCIÓN DE GPU
# =============================================================================
detect_gpu() {
    log_info "Detectando GPU..."

    if ! command -v nvidia-smi &>/dev/null; then
        log_error "nvidia-smi no encontrado. ¿Drivers NVIDIA instalados?"
        exit 1
    fi

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    SM=$(echo "$COMPUTE_CAP" | tr -d '.')

    echo ""
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║           GPU DETECTADA                  ║"
    echo "  ╠══════════════════════════════════════════╣"
    printf "  ║  GPU:    %-30s ║\n" "$GPU_NAME"
    printf "  ║  VRAM:   %-30s ║\n" "$GPU_MEMORY"
    printf "  ║  SM:     %-30s ║\n" "$COMPUTE_CAP (SM $SM)"
    echo "  ╚══════════════════════════════════════════╝"
    echo ""

    # Determinar arquitectura
    case $SM in
        75) GPU_ARCH="Turing" ;;
        86|87) GPU_ARCH="Ampere" ;;
        89) GPU_ARCH="Ada Lovelace" ;;
        90) GPU_ARCH="Hopper" ;;
        100|120) GPU_ARCH="Blackwell" ;;
        *) GPU_ARCH="Unknown" ;;
    esac

    log_ok "Arquitectura: $GPU_ARCH"

    # Blackwell necesita TensorRT-RTX
    if [ "$SM" -ge 100 ]; then
        NEEDS_TENSORRT_RTX=true
        log_warn "GPU Blackwell detectada - requiere TensorRT-RTX"
    else
        NEEDS_TENSORRT_RTX=false
    fi
}

# =============================================================================
# DETECCIÓN DE DEPENDENCIAS
# =============================================================================
detect_opencv() {
    log_info "Buscando OpenCV con CUDA..."

    OPENCV_DIR=""

    # Buscar en ubicaciones comunes
    for dir in \
        "$LIBS_DIR/opencv_cuda/lib/cmake/opencv4" \
        "/usr/local/lib/cmake/opencv4" \
        "/opt/opencv/lib/cmake/opencv4"; do
        if [ -f "$dir/OpenCVConfig.cmake" ]; then
            # Verificar si tiene CUDA
            if grep -q "CUDA" "$dir/OpenCVConfig.cmake" 2>/dev/null; then
                OPENCV_DIR="$dir"
                break
            fi
        fi
    done

    if [ -n "$OPENCV_DIR" ]; then
        OPENCV_VERSION=$(grep "OpenCV_VERSION " "$OPENCV_DIR/OpenCVConfig-version.cmake" 2>/dev/null | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
        log_ok "OpenCV CUDA encontrado: $OPENCV_DIR (v$OPENCV_VERSION)"
        return 0
    else
        log_warn "OpenCV con CUDA no encontrado"
        return 1
    fi
}

detect_tensorrt() {
    log_info "Buscando TensorRT..."

    TENSORRT_DIR=""
    TENSORRT_LIB_NAME=""

    # Para Blackwell, buscar TensorRT-RTX primero
    if [ "$NEEDS_TENSORRT_RTX" = true ]; then
        for dir in "$LIBS_DIR"/TensorRT-RTX*; do
            if [ -d "$dir" ] && [ -f "$dir/lib/libtensorrt_rtx.so" ]; then
                TENSORRT_DIR="$dir"
                TENSORRT_LIB_NAME="tensorrt_rtx"
                break
            fi
        done
    fi

    # Buscar TensorRT estándar
    if [ -z "$TENSORRT_DIR" ]; then
        for dir in "$LIBS_DIR"/TensorRT-* /usr/local/tensorrt; do
            if [ -d "$dir" ] && [ -f "$dir/lib/libnvinfer.so" ]; then
                TENSORRT_DIR="$dir"
                TENSORRT_LIB_NAME="nvinfer"
                break
            fi
        done
    fi

    if [ -n "$TENSORRT_DIR" ]; then
        TENSORRT_VERSION=$(basename "$TENSORRT_DIR" | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
        log_ok "TensorRT encontrado: $TENSORRT_DIR (v$TENSORRT_VERSION)"
        log_ok "Librería: $TENSORRT_LIB_NAME"
        return 0
    else
        log_warn "TensorRT no encontrado"
        return 1
    fi
}

detect_cuda() {
    log_info "Verificando CUDA..."

    if command -v nvcc &>/dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | grep -oP '\d+\.\d+')
        log_ok "CUDA $CUDA_VERSION encontrado"
        return 0
    else
        log_warn "CUDA toolkit no encontrado"
        return 1
    fi
}

# =============================================================================
# INSTALACIÓN DE DEPENDENCIAS
# =============================================================================
install_opencv_cuda() {
    log_info "Instalando OpenCV con CUDA..."

    mkdir -p "$LIBS_DIR"

    # Crear script de build adaptado a la GPU actual
    cat > "$LIBS_DIR/build_opencv_cuda.sh" << 'OPENCV_SCRIPT'
#!/bin/bash
set -e

LIBS_DIR="$HOME/libs"
TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"
export TMPDIR

# Detectar SM de la GPU
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

cd "$LIBS_DIR"

# Descargar OpenCV
if [ ! -d "opencv" ]; then
    echo "=== Descargando OpenCV 4.9.0 ==="
    git clone --depth 1 --branch 4.9.0 https://github.com/opencv/opencv.git
    git clone --depth 1 --branch 4.9.0 https://github.com/opencv/opencv_contrib.git
fi

rm -rf opencv/build
mkdir -p opencv/build
cd opencv/build

# Detectar compilador compatible con CUDA
GCC_VER=""
for v in 12 11 10; do
    if command -v gcc-$v &>/dev/null; then
        GCC_VER=$v
        break
    fi
done

CMAKE_EXTRA=""
if [ -n "$GCC_VER" ]; then
    CMAKE_EXTRA="-DCMAKE_C_COMPILER=/usr/bin/gcc-$GCC_VER -DCMAKE_CXX_COMPILER=/usr/bin/g++-$GCC_VER"
fi

echo "=== Configurando CMake (SM $SM) ==="
cmake .. \
    $CMAKE_EXTRA \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$LIBS_DIR/opencv_cuda" \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=OFF \
    -DOPENCV_DNN_CUDA=OFF \
    -DENABLE_FAST_MATH=ON \
    -DCUDA_FAST_MATH=ON \
    -DWITH_CUBLAS=ON \
    -DCUDA_ARCH_BIN="${SM:0:1}.${SM:1:1}" \
    -DCUDA_ARCH_PTX="${SM:0:1}.${SM:1:1}" \
    -DOPENCV_EXTRA_MODULES_PATH="$LIBS_DIR/opencv_contrib/modules" \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_opencv_python2=OFF \
    -DWITH_GTK=ON

NPROC=$(nproc)
echo "=== Compilando con $NPROC cores ==="
make -j$NPROC

echo "=== Instalando ==="
make install

echo "=== COMPLETADO ==="
echo "OpenCV instalado en: $LIBS_DIR/opencv_cuda"
OPENCV_SCRIPT

    chmod +x "$LIBS_DIR/build_opencv_cuda.sh"

    echo ""
    log_warn "La compilación de OpenCV tarda 20-40 minutos."
    read -p "¿Compilar ahora? [y/N] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Compilando OpenCV (log en $LIBS_DIR/opencv_build.log)..."
        bash "$LIBS_DIR/build_opencv_cuda.sh" 2>&1 | tee "$LIBS_DIR/opencv_build.log"
        log_ok "OpenCV compilado"
        OPENCV_DIR="$LIBS_DIR/opencv_cuda/lib/cmake/opencv4"
    else
        log_info "Puedes compilar después con:"
        echo "  nohup bash $LIBS_DIR/build_opencv_cuda.sh > $LIBS_DIR/opencv_build.log 2>&1 &"
    fi
}

install_tensorrt() {
    log_info "TensorRT debe descargarse manualmente de NVIDIA Developer"
    echo ""
    echo "  1. Ve a: https://developer.nvidia.com/tensorrt"
    echo "  2. Descarga TensorRT 10.x para tu versión de CUDA"
    echo "  3. Extrae en: $LIBS_DIR/TensorRT-<version>"
    echo ""

    if [ "$NEEDS_TENSORRT_RTX" = true ]; then
        log_warn "Para Blackwell, necesitas TensorRT-RTX:"
        echo "  https://developer.nvidia.com/tensorrt-rtx-downloads"
    fi

    read -p "Presiona Enter cuando hayas instalado TensorRT..."
    detect_tensorrt
}

# =============================================================================
# ACTUALIZAR CMakeLists.txt
# =============================================================================
update_cmake() {
    log_info "Actualizando CMakeLists.txt..."

    CMAKE_FILE="$PROJECT_DIR/CMakeLists.txt"

    if [ ! -f "$CMAKE_FILE" ]; then
        log_error "CMakeLists.txt no encontrado"
        return 1
    fi

    # Backup
    cp "$CMAKE_FILE" "$CMAKE_FILE.bak"

    # Crear nuevo CMakeLists.txt
    cat > "$CMAKE_FILE" << CMAKE_CONTENT
cmake_minimum_required(VERSION 3.16)
project(aria_slam)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "$OPENCV_DIR")
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# TensorRT ($TENSORRT_VERSION) for $GPU_NAME (SM $SM)
set(TensorRT_DIR "$TENSORRT_DIR")
set(TensorRT_INCLUDE_DIRS "\${TensorRT_DIR}/include")
set(TensorRT_LIBS "\${TensorRT_DIR}/lib")

include_directories(
    include
    \${OpenCV_INCLUDE_DIRS}
    \${CUDA_INCLUDE_DIRS}
    \${TensorRT_INCLUDE_DIRS}
    \${EIGEN3_INCLUDE_DIR}
)

link_directories(\${TensorRT_LIBS})

add_executable(aria_slam
    src/main.cpp
    src/Frame.cpp
    src/TRTInference.cpp
    src/IMU.cpp
    src/LoopClosure.cpp
    src/Mapper.cpp
)

target_link_libraries(aria_slam
    \${OpenCV_LIBS}
    \${CUDA_LIBRARIES}
    $TENSORRT_LIB_NAME
    cudart
    Eigen3::Eigen
    g2o_core
    g2o_stuff
    g2o_types_slam3d
    g2o_solver_eigen
)

# EuRoC dataset evaluation
add_executable(euroc_eval
    src/euroc_eval.cpp
    src/EuRoCReader.cpp
    src/Frame.cpp
    src/TRTInference.cpp
    src/IMU.cpp
    src/LoopClosure.cpp
    src/Mapper.cpp
)

target_link_libraries(euroc_eval
    \${OpenCV_LIBS}
    \${CUDA_LIBRARIES}
    $TENSORRT_LIB_NAME
    cudart
    Eigen3::Eigen
    g2o_core
    g2o_stuff
    g2o_types_slam3d
    g2o_solver_eigen
)

# Experiments/benchmarks
add_executable(benchmark_imu
    experiments/benchmark_imu.cpp
    src/IMU.cpp
)

target_link_libraries(benchmark_imu
    Eigen3::Eigen
)
CMAKE_CONTENT

    log_ok "CMakeLists.txt actualizado"
    log_info "Backup guardado en CMakeLists.txt.bak"
}

# =============================================================================
# GENERAR ENGINE YOLO
# =============================================================================
generate_yolo_engine() {
    MODEL=${1:-yolo26s}
    log_info "Generando engine para $MODEL..."

    MODELS_DIR="$PROJECT_DIR/models"
    ONNX_FILE="$MODELS_DIR/${MODEL}.onnx"
    ENGINE_FILE="$MODELS_DIR/${MODEL}.engine"
    VENV_DIR="$PROJECT_DIR/.venv"

    mkdir -p "$MODELS_DIR"

    # Descargar ONNX si no existe
    if [ ! -f "$ONNX_FILE" ]; then
        log_info "Descargando modelo $MODEL..."

        # Crear venv si no existe
        if [ ! -d "$VENV_DIR" ]; then
            log_info "Creando entorno virtual en $VENV_DIR..."
            python3 -m venv "$VENV_DIR"
        fi

        # Activar venv e instalar ultralytics
        source "$VENV_DIR/bin/activate"

        # Usar /home para cache de pip (root / está lleno)
        export TMPDIR="$HOME/tmp"
        export PIP_CACHE_DIR="$HOME/.cache/pip"
        mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

        if ! python3 -c "import ultralytics" 2>/dev/null; then
            log_info "Instalando ultralytics en venv..."
            pip install --upgrade pip -q
            pip install ultralytics -q
        fi

        python3 << PYTHON_SCRIPT
from ultralytics import YOLO
import shutil
import os

model = YOLO('${MODEL}.pt')
model.export(format='onnx', imgsz=640, simplify=True)

# Mover al directorio de modelos
src = '${MODEL}.onnx'
if os.path.exists(src):
    shutil.move(src, '$ONNX_FILE')
    print(f"ONNX guardado en: $ONNX_FILE")
PYTHON_SCRIPT

        deactivate 2>/dev/null || true
    fi

    if [ ! -f "$ONNX_FILE" ]; then
        log_error "No se pudo obtener $ONNX_FILE"
        return 1
    fi

    log_ok "ONNX disponible: $ONNX_FILE"

    # Encontrar trtexec
    TRTEXEC=""
    if [ -n "$TENSORRT_DIR" ]; then
        if [ -f "$TENSORRT_DIR/bin/trtexec" ]; then
            TRTEXEC="$TENSORRT_DIR/bin/trtexec"
        elif [ -f "$TENSORRT_DIR/bin/tensorrt_rtx" ]; then
            TRTEXEC="$TENSORRT_DIR/bin/tensorrt_rtx"
        fi
    fi

    if [ -z "$TRTEXEC" ]; then
        log_error "trtexec no encontrado"
        return 1
    fi

    log_info "Usando: $TRTEXEC"
    log_info "Generando engine (esto puede tardar unos minutos)..."

    # Generar engine (TensorRT 10+ usa --memPoolSize en vez de --workspace)
    # YOLO26 tiene bloques de atención que requieren ~5GB workspace
    $TRTEXEC \
        --onnx="$ONNX_FILE" \
        --saveEngine="$ENGINE_FILE" \
        --fp16 \
        --memPoolSize=workspace:5000M

    if [ -f "$ENGINE_FILE" ]; then
        ENGINE_SIZE=$(du -h "$ENGINE_FILE" | cut -f1)
        log_ok "Engine generado: $ENGINE_FILE ($ENGINE_SIZE)"

        # Benchmark
        log_info "Benchmark:"
        $TRTEXEC --loadEngine="$ENGINE_FILE" --warmUp=500 --iterations=100 2>&1 | grep -E "Throughput|mean" | head -3
    else
        log_error "Error generando engine"
        return 1
    fi
}

# =============================================================================
# MAIN
# =============================================================================
main() {
    echo ""
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║     ARIA-SLAM Machine Setup              ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo ""

    MODE=${1:---full}

    # Siempre detectar GPU
    detect_gpu
    detect_cuda

    case $MODE in
        --config-only)
            detect_opencv || true
            detect_tensorrt || true

            if [ -n "$OPENCV_DIR" ] && [ -n "$TENSORRT_DIR" ]; then
                update_cmake
            else
                log_error "Faltan dependencias. Usa --full para instalar."
                exit 1
            fi
            ;;

        --engine-only)
            detect_tensorrt || { log_error "TensorRT requerido"; exit 1; }
            generate_yolo_engine yolo26s
            ;;

        --full|*)
            # OpenCV
            if ! detect_opencv; then
                install_opencv_cuda
            fi

            # TensorRT
            if ! detect_tensorrt; then
                install_tensorrt
            fi

            # Verificar que tenemos todo
            if [ -z "$OPENCV_DIR" ] || [ -z "$TENSORRT_DIR" ]; then
                log_error "Dependencias incompletas"
                exit 1
            fi

            # Actualizar CMake
            update_cmake

            # Generar engine
            generate_yolo_engine yolo26s

            echo ""
            echo "  ╔══════════════════════════════════════════╗"
            echo "  ║           SETUP COMPLETADO               ║"
            echo "  ╠══════════════════════════════════════════╣"
            echo "  ║  Siguiente paso:                         ║"
            echo "  ║    cd build && cmake .. && make -j       ║"
            echo "  ╚══════════════════════════════════════════╝"
            echo ""
            ;;
    esac
}

main "$@"
