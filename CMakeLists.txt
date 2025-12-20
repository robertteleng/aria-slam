cmake_minimum_required(VERSION 3.16)
project(aria_slam)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(aria_slam src/main.cpp)
target_link_libraries(aria_slam ${OpenCV_LIBS})