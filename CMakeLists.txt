cmake_minimum_required(VERSION 3.16)
project(aria_slam)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "/home/roberto/libs/opencv_cuda/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# TensorRT
set(TensorRT_DIR "/home/roberto/libs/TensorRT-10.7.0.23")
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
)

target_link_libraries(aria_slam
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    cudart
    Eigen3::Eigen
)

# IMU test executable
add_executable(test_imu
    src/test_imu.cpp
    src/IMU.cpp
)

target_link_libraries(test_imu
    Eigen3::Eigen
)
