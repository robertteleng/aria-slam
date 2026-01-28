# H06: TensorRT Object Detection

**Status:** ✅ Completed

## Objective

Integrate TensorRT-accelerated YOLO for real-time object detection to filter dynamic objects from SLAM.

## Requirements

- TensorRT runtime
- YOLO model conversion to TensorRT engine
- Bounding box detection pipeline
- Dynamic object filtering

## Why Filter Dynamic Objects?

SLAM assumes a static world. Moving objects (people, cars) cause:
- False feature matches
- Drift in pose estimation
- Ghost points in map

Solution: Detect and mask dynamic objects before feature extraction.

## Installation

### TensorRT Setup

```bash
# Download TensorRT from NVIDIA
tar -xzf TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz
mv TensorRT-10.7.0.23 ~/libs/

# Add to environment
export LD_LIBRARY_PATH=~/libs/TensorRT-10.7.0.23/lib:$LD_LIBRARY_PATH
```

### Generate YOLO Engine

```bash
# Create Python venv (Ubuntu 24.04 requirement)
python3 -m venv .venv
source .venv/bin/activate

# Install ultralytics
pip install ultralytics

# Export to TensorRT
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11s.pt')  # or yolo26s for latest
model.export(format='engine',
             imgsz=640,
             device=0,
             workspace=5)  # 5GB for attention layers
"
```

## Implementation

### TensorRT Detector

```cpp
#include <NvInfer.h>
#include <cuda_runtime.h>

class YoloDetector {
public:
    YoloDetector(const std::string& engine_path) {
        loadEngine(engine_path);
        allocateBuffers();
    }

    std::vector<Detection> detect(const cv::Mat& image) {
        // Preprocess
        cv::Mat blob;
        preprocess(image, blob);

        // Copy to GPU
        cudaMemcpyAsync(d_input_, blob.data, input_size_,
                        cudaMemcpyHostToDevice, stream_);

        // Inference
        context_->enqueueV3(stream_);

        // Copy results
        cudaMemcpyAsync(h_output_, d_output_, output_size_,
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // Postprocess
        return postprocess(h_output_, image.size());
    }

private:
    void preprocess(const cv::Mat& image, cv::Mat& blob) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(640, 640));
        resized.convertTo(blob, CV_32F, 1.0/255.0);
        // HWC -> CHW, normalize
    }

    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;
    void *d_input_, *d_output_;
    float* h_output_;
};
```

### Dynamic Object Mask

```cpp
cv::Mat createDynamicMask(const cv::Mat& image,
                          const std::vector<Detection>& detections) {
    cv::Mat mask = cv::Mat::ones(image.size(), CV_8UC1) * 255;

    // Dynamic object classes (COCO)
    std::set<int> dynamic_classes = {
        0,   // person
        1,   // bicycle
        2,   // car
        3,   // motorcycle
        5,   // bus
        7,   // truck
        15,  // cat
        16,  // dog
    };

    for (const auto& det : detections) {
        if (dynamic_classes.count(det.class_id) && det.confidence > 0.5) {
            // Expand bbox slightly
            cv::Rect bbox = det.bbox;
            bbox.x = std::max(0, bbox.x - 10);
            bbox.y = std::max(0, bbox.y - 10);
            bbox.width = std::min(image.cols - bbox.x, bbox.width + 20);
            bbox.height = std::min(image.rows - bbox.y, bbox.height + 20);

            mask(bbox) = 0;  // Mask out dynamic region
        }
    }

    return mask;
}
```

### Integration with Feature Extraction

```cpp
void extractFeatures(const cv::Mat& image, ...) {
    // Detect dynamic objects
    auto detections = detector_.detect(image);
    cv::Mat mask = createDynamicMask(image, detections);

    // Extract features only in static regions
    orb_->detectAndCompute(gray, mask, keypoints, descriptors);
}
```

## CMakeLists.txt

```cmake
# TensorRT
set(TENSORRT_DIR /home/roberto/libs/TensorRT-10.7.0.23)
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)

target_link_libraries(aria_slam
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES}
)
```

## Performance

| Model | Input | GPU | Latency | FPS |
|-------|-------|-----|---------|-----|
| YOLO11s | 640x640 | RTX 2060 | 3.2ms | 312 |
| YOLO26s | 640x640 | RTX 2060 | 2.9ms | 343 |
| YOLO11s | 640x640 | Jetson Orin | 8ms | 125 |

## Engine Generation Notes

### Workspace Size

YOLO models with attention (v11, v26) need more workspace:
```bash
# Old (deprecated in TensorRT 10)
--workspace 4096

# New
--memPoolSize=workspace:5000M
```

### Platform Specific

Engines are **not portable** between:
- Different GPU architectures
- Different TensorRT versions
- Different CUDA versions

Generate engine on target platform.

## Next Steps

→ H08: Sensor Fusion (IMU)
