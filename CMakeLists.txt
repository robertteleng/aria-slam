cmake_minimum_required(VERSION 3.10)
project(aria_slam LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(aria_slam src/main.cpp)
target_link_libraries(aria_slam ${OpenCV_LIBS})
