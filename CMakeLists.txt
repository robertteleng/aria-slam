cmake_minimum_required(VERSION 3.16)
project(aria_slam)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

include_directories(include)  # AÑADIR
add_executable(aria_slam src/main.cpp src/Frame.cpp)  # AÑADIR Frame.cpp
target_link_libraries(aria_slam ${OpenCV_LIBS})