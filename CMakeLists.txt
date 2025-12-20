cmake_minimum_required(VERSION 3.16)
project(aria_slam)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "/home/roberto/libs/opencv_cuda/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(aria_slam src/main.cpp src/Frame.cpp)
target_link_libraries(aria_slam ${OpenCV_LIBS})