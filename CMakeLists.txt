cmake_minimum_required(VERSION 3.16)
project(aria_slam)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "$ENV{HOME}/libs/opencv_cuda/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# TensorRT (system install supports SM 120 since v10.15)
set(TensorRT_INCLUDE_DIRS "/usr/include/x86_64-linux-gnu")
set(TensorRT_LIBS "/usr/lib/x86_64-linux-gnu")

# H12 Clean Architecture: include/core, include/interfaces, include/adapters, include/pipeline
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
    src/legacy/Frame.cpp
    src/legacy/TRTInference.cpp
    src/legacy/IMU.cpp
    src/legacy/LoopClosure.cpp
    src/legacy/Mapper.cpp
)

target_link_libraries(aria_slam
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
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
    src/legacy/EuRoCReader.cpp
    src/legacy/Frame.cpp
    src/legacy/TRTInference.cpp
    src/legacy/IMU.cpp
    src/legacy/LoopClosure.cpp
    src/legacy/Mapper.cpp
)

target_link_libraries(euroc_eval
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
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
    src/legacy/IMU.cpp
)

target_link_libraries(benchmark_imu
    Eigen3::Eigen
)
