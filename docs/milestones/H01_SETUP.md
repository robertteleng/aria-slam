# H01: Setup + Capture

**Status:** ✅ Completed

## Objective

Set up the development environment and basic video capture pipeline.

## Requirements

- CMake build system
- OpenCV for video I/O
- Basic project structure

## Implementation

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.16)
project(aria_slam)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(aria_slam src/main.cpp)
target_link_libraries(aria_slam ${OpenCV_LIBS})
```

### Video Capture

```cpp
#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap("../test.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::imshow("Frame", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}
```

## Files Created

- `CMakeLists.txt`
- `src/main.cpp`

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| CMake | >= 3.16 | Build system |
| OpenCV | >= 4.6 | Video capture |

## Verification

```bash
mkdir build && cd build
cmake ..
make
./aria_slam
```

## Next Steps

→ H02: Feature Extraction
