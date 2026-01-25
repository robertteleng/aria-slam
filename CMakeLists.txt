cmake_minimum_required(VERSION 3.16)
project(aria_slam)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "/home/robert/libs/opencv_cuda/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# TensorRT-RTX (for Blackwell support)
set(TensorRT_DIR "/home/robert/libs/TensorRT-RTX-1.3.0.35")
set(TensorRT_INCLUDE_DIRS "${TensorRT_DIR}/include")
set(TensorRT_LIBS "${TensorRT_DIR}/lib")

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

link_directories(${TensorRT_LIBS})

add_executable(aria_slam
    src/main.cpp
    src/Frame.cpp
    src/TRTInference.cpp
    src/IMU.cpp
    src/LoopClosure.cpp
    src/Mapper.cpp
)

target_link_libraries(aria_slam
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    tensorrt_rtx
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
    src/IMU.cpp
    src/LoopClosure.cpp
    src/Mapper.cpp
)

target_link_libraries(euroc_eval
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
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
