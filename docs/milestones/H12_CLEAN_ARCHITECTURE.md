# H12: Clean Architecture

## Overview

This document defines the architectural refactoring of aria-slam following **Hexagonal Architecture** (Ports & Adapters) and **SOLID principles**. The goal is to enable:

- **Testability**: Mock any component for unit testing
- **Flexibility**: Swap CPU/GPU implementations without changing business logic
- **Maintainability**: Clear boundaries between layers
- **Multithreading**: Thread-safe interfaces ready for H13

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ SlamPipeline│  │ EurocRunner │  │  AriaRunner │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Ports (Interfaces)                          │
│  ┌────────────────┐  ┌────────────┐  ┌─────────────────┐        │
│  │IFeatureExtractor│  │  IMatcher  │  │ ILoopDetector   │        │
│  └────────────────┘  └────────────┘  └─────────────────┘        │
│  ┌────────────────┐  ┌────────────┐  ┌─────────────────┐        │
│  │IObjectDetector │  │ISensorFusion│  │    IMapper      │        │
│  └────────────────┘  └────────────┘  └─────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
          ▲                ▲                ▲
          │                │                │
┌─────────┼────────────────┼────────────────┼─────────────────────┐
│         │    Adapters    │                │                      │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐              │
│  │  GPU Impl   │  │  CPU Impl   │  │    Mocks    │              │
│  │ OrbCuda     │  │  OrbCpu     │  │ MockExtract │              │
│  │ CudaMatcher │  │  BFMatcher  │  │ MockMatcher │              │
│  │ YoloTrt     │  │             │  │             │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
          ▲                ▲                ▲
          │                │                │
┌─────────────────────────────────────────────────────────────────┐
│                     Domain Layer (Core)                          │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐          │
│  │  Frame  │  │ KeyFrame │  │ MapPoint │  │  Pose   │          │
│  └─────────┘  └──────────┘  └──────────┘  └─────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Domain Layer (Core Entities)

Pure data structures with no dependencies on OpenCV, CUDA, or external libraries.

### Frame.hpp

```cpp
#pragma once
#include <vector>
#include <cstdint>
#include <Eigen/Dense>

namespace aria::core {

struct KeyPoint {
    float x, y;           // Position in image
    float size;           // Diameter of meaningful keypoint neighborhood
    float angle;          // Orientation in degrees [0, 360)
    float response;       // Response by which the keypoints are sorted
    int octave;           // Octave (pyramid layer) from which the keypoint was extracted
};

struct Frame {
    uint64_t id;
    double timestamp;
    int width, height;

    std::vector<KeyPoint> keypoints;
    std::vector<uint8_t> descriptors;  // Flattened: N x 32 bytes for ORB

    // Computed pose (optional, filled after tracking)
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

    size_t descriptorSize() const { return 32; }  // ORB descriptor size
    size_t numKeypoints() const { return keypoints.size(); }
};

} // namespace aria::core
```

### KeyFrame.hpp

```cpp
#pragma once
#include "Frame.hpp"
#include <memory>

namespace aria::core {

struct KeyFrame {
    uint64_t id;
    double timestamp;

    Frame frame;
    Eigen::Matrix4d pose;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;

    // Covisibility (frames that see same map points)
    std::vector<uint64_t> covisible_keyframes;

    // Map point observations
    std::vector<uint64_t> observed_mappoints;
};

} // namespace aria::core
```

### MapPoint.hpp

```cpp
#pragma once
#include <Eigen/Dense>
#include <vector>

namespace aria::core {

struct MapPoint {
    uint64_t id;
    Eigen::Vector3d position;
    Eigen::Vector3d normal;       // Mean viewing direction

    std::vector<uint8_t> descriptor;  // Representative descriptor

    // Observations: keyframe_id -> keypoint_index
    std::vector<std::pair<uint64_t, int>> observations;

    // Quality metrics
    int num_observations = 0;
    float min_distance = 0.0f;    // Scale invariance bounds
    float max_distance = 0.0f;

    bool is_bad = false;
};

} // namespace aria::core
```

### Pose.hpp

```cpp
#pragma once
#include <Eigen/Dense>

namespace aria::core {

struct Pose {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    double timestamp;

    // Covariance (6x6: position + orientation)
    Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity();

    Eigen::Matrix4d toMatrix() const {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = orientation.toRotationMatrix();
        T.block<3,1>(0,3) = position;
        return T;
    }

    static Pose fromMatrix(const Eigen::Matrix4d& T, double ts = 0.0) {
        Pose p;
        p.position = T.block<3,1>(0,3);
        p.orientation = Eigen::Quaterniond(T.block<3,3>(0,0));
        p.timestamp = ts;
        return p;
    }
};

} // namespace aria::core
```

## Ports (Interfaces)

Abstract interfaces that define contracts. No implementation details.

### IFeatureExtractor.hpp

```cpp
#pragma once
#include "core/Frame.hpp"
#include <memory>
#include <vector>

namespace aria::interfaces {

class IFeatureExtractor {
public:
    virtual ~IFeatureExtractor() = default;

    // Extract keypoints and descriptors from raw image data
    // @param image_data Raw pixel data (grayscale, row-major)
    // @param width Image width
    // @param height Image height
    // @param frame Output frame with keypoints and descriptors
    virtual void extract(
        const uint8_t* image_data,
        int width,
        int height,
        core::Frame& frame
    ) = 0;

    // Async extraction (for GPU implementations)
    // Returns immediately, results available after sync()
    virtual void extractAsync(
        const uint8_t* image_data,
        int width,
        int height,
        core::Frame& frame
    ) { extract(image_data, width, height, frame); }  // Default: sync

    // Wait for async operation to complete
    virtual void sync() {}

    // Configuration
    virtual void setMaxFeatures(int n) = 0;
    virtual int getMaxFeatures() const = 0;
};

using FeatureExtractorPtr = std::unique_ptr<IFeatureExtractor>;

} // namespace aria::interfaces
```

### IMatcher.hpp

```cpp
#pragma once
#include "core/Frame.hpp"
#include <vector>

namespace aria::interfaces {

struct Match {
    int query_idx;      // Index in query frame
    int train_idx;      // Index in train frame
    float distance;     // Descriptor distance
};

class IMatcher {
public:
    virtual ~IMatcher() = default;

    // Match descriptors between two frames
    // @param query Query frame (current)
    // @param train Train frame (previous/reference)
    // @param matches Output matches
    // @param ratio_threshold Lowe's ratio test threshold (0.0 = disabled)
    virtual void match(
        const core::Frame& query,
        const core::Frame& train,
        std::vector<Match>& matches,
        float ratio_threshold = 0.75f
    ) = 0;

    // Match one frame against multiple (for loop closure)
    virtual void matchMultiple(
        const core::Frame& query,
        const std::vector<core::Frame>& candidates,
        std::vector<std::vector<Match>>& all_matches,
        float ratio_threshold = 0.75f
    ) {
        all_matches.resize(candidates.size());
        for (size_t i = 0; i < candidates.size(); i++) {
            match(query, candidates[i], all_matches[i], ratio_threshold);
        }
    }
};

using MatcherPtr = std::unique_ptr<IMatcher>;

} // namespace aria::interfaces
```

### ILoopDetector.hpp

```cpp
#pragma once
#include "core/KeyFrame.hpp"
#include "IMatcher.hpp"
#include <optional>

namespace aria::interfaces {

struct LoopCandidate {
    uint64_t query_id;
    uint64_t match_id;
    double score;
    std::vector<Match> matches;
    Eigen::Matrix4d relative_pose;
};

class ILoopDetector {
public:
    virtual ~ILoopDetector() = default;

    // Add keyframe to database
    virtual void addKeyFrame(const core::KeyFrame& kf) = 0;

    // Detect loop closure
    // @param query Current keyframe
    // @return Loop candidate if found, nullopt otherwise
    virtual std::optional<LoopCandidate> detect(const core::KeyFrame& query) = 0;

    // Get number of detected loops
    virtual int getLoopCount() const = 0;

    // Configuration
    virtual void setMinFramesBetween(int n) = 0;
    virtual void setMinScore(double s) = 0;
    virtual void setMinMatches(int n) = 0;
};

using LoopDetectorPtr = std::unique_ptr<ILoopDetector>;

} // namespace aria::interfaces
```

### IObjectDetector.hpp

```cpp
#pragma once
#include <vector>
#include <string>

namespace aria::interfaces {

struct Detection {
    float x1, y1, x2, y2;   // Bounding box
    float confidence;
    int class_id;
    std::string class_name;
};

class IObjectDetector {
public:
    virtual ~IObjectDetector() = default;

    // Detect objects in image
    // @param image_data RGB image data (row-major, 3 channels)
    // @param width Image width
    // @param height Image height
    // @param detections Output detections
    // @param conf_threshold Confidence threshold
    // @param nms_threshold NMS IoU threshold
    virtual void detect(
        const uint8_t* image_data,
        int width,
        int height,
        std::vector<Detection>& detections,
        float conf_threshold = 0.5f,
        float nms_threshold = 0.45f
    ) = 0;

    // Async detection
    virtual void detectAsync(
        const uint8_t* image_data,
        int width,
        int height
    ) = 0;

    // Get results after async detection
    virtual void getDetections(
        std::vector<Detection>& detections,
        float conf_threshold = 0.5f,
        float nms_threshold = 0.45f
    ) = 0;

    virtual void sync() = 0;
};

using ObjectDetectorPtr = std::unique_ptr<IObjectDetector>;

} // namespace aria::interfaces
```

### ISensorFusion.hpp

```cpp
#pragma once
#include "core/Pose.hpp"
#include <Eigen/Dense>

namespace aria::interfaces {

struct ImuMeasurement {
    double timestamp;
    Eigen::Vector3d accel;      // m/s^2
    Eigen::Vector3d gyro;       // rad/s
};

class ISensorFusion {
public:
    virtual ~ISensorFusion() = default;

    // IMU prediction step (high frequency: 200Hz)
    virtual void predictIMU(const ImuMeasurement& imu) = 0;

    // Visual odometry update step (low frequency: 30Hz)
    virtual void updateVO(const core::Pose& vo_pose) = 0;

    // Get current fused state
    virtual core::Pose getFusedPose() const = 0;

    // Get velocity estimate
    virtual Eigen::Vector3d getVelocity() const = 0;

    // Reset filter
    virtual void reset() = 0;
    virtual void reset(const core::Pose& initial_pose) = 0;
};

using SensorFusionPtr = std::unique_ptr<ISensorFusion>;

} // namespace aria::interfaces
```

### IMapper.hpp

```cpp
#pragma once
#include "core/Frame.hpp"
#include "core/MapPoint.hpp"
#include "core/Pose.hpp"
#include "IMatcher.hpp"
#include <vector>
#include <string>

namespace aria::interfaces {

class IMapper {
public:
    virtual ~IMapper() = default;

    // Triangulate new map points from matched frames
    // @param frame1 First frame with pose
    // @param frame2 Second frame with pose
    // @param matches Matches between frames
    // @param K Camera intrinsic matrix (3x3)
    // @param new_points Output: newly created map points
    virtual void triangulate(
        const core::Frame& frame1,
        const core::Frame& frame2,
        const core::Pose& pose1,
        const core::Pose& pose2,
        const std::vector<Match>& matches,
        const Eigen::Matrix3d& K,
        std::vector<core::MapPoint>& new_points
    ) = 0;

    // Get all map points
    virtual const std::vector<core::MapPoint>& getMapPoints() const = 0;

    // Export to file
    virtual void exportPLY(const std::string& filename) const = 0;
    virtual void exportPCD(const std::string& filename) const = 0;

    // Clear map
    virtual void clear() = 0;

    // Statistics
    virtual size_t size() const = 0;
};

using MapperPtr = std::unique_ptr<IMapper>;

} // namespace aria::interfaces
```

## GPU Adapters

Implementations using CUDA and TensorRT.

### OrbCudaExtractor.hpp

```cpp
#pragma once
#include "interfaces/IFeatureExtractor.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include <cuda_runtime.h>

namespace aria::adapters::gpu {

class OrbCudaExtractor : public interfaces::IFeatureExtractor {
public:
    explicit OrbCudaExtractor(int max_features = 1000, cudaStream_t stream = nullptr);
    ~OrbCudaExtractor() override;

    void extract(
        const uint8_t* image_data,
        int width,
        int height,
        core::Frame& frame
    ) override;

    void extractAsync(
        const uint8_t* image_data,
        int width,
        int height,
        core::Frame& frame
    ) override;

    void sync() override;

    void setMaxFeatures(int n) override;
    int getMaxFeatures() const override { return max_features_; }

    // GPU-specific: get descriptors without download (for GPU matching)
    const cv::cuda::GpuMat& getGpuDescriptors() const { return gpu_descriptors_; }

private:
    cv::Ptr<cv::cuda::ORB> orb_;
    cv::cuda::GpuMat gpu_image_;
    cv::cuda::GpuMat gpu_keypoints_;
    cv::cuda::GpuMat gpu_descriptors_;
    cv::cuda::Stream cv_stream_;
    cudaStream_t cuda_stream_;
    int max_features_;
    bool owns_stream_;
};

} // namespace aria::adapters::gpu
```

### CudaMatcher.hpp

```cpp
#pragma once
#include "interfaces/IMatcher.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include <cuda_runtime.h>

namespace aria::adapters::gpu {

class CudaMatcher : public interfaces::IMatcher {
public:
    explicit CudaMatcher(cudaStream_t stream = nullptr);
    ~CudaMatcher() override;

    void match(
        const core::Frame& query,
        const core::Frame& train,
        std::vector<interfaces::Match>& matches,
        float ratio_threshold = 0.75f
    ) override;

    // GPU-to-GPU matching (zero-copy when used with OrbCudaExtractor)
    void matchGpu(
        const cv::cuda::GpuMat& query_desc,
        const cv::cuda::GpuMat& train_desc,
        std::vector<interfaces::Match>& matches,
        float ratio_threshold = 0.75f
    );

private:
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher_;
    cv::cuda::Stream cv_stream_;
    cudaStream_t cuda_stream_;
    bool owns_stream_;
};

} // namespace aria::adapters::gpu
```

### YoloTrtDetector.hpp

```cpp
#pragma once
#include "interfaces/IObjectDetector.hpp"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>

namespace aria::adapters::gpu {

class YoloTrtDetector : public interfaces::IObjectDetector {
public:
    explicit YoloTrtDetector(const std::string& engine_path, cudaStream_t stream = nullptr);
    ~YoloTrtDetector() override;

    void detect(
        const uint8_t* image_data,
        int width,
        int height,
        std::vector<interfaces::Detection>& detections,
        float conf_threshold = 0.5f,
        float nms_threshold = 0.45f
    ) override;

    void detectAsync(
        const uint8_t* image_data,
        int width,
        int height
    ) override;

    void getDetections(
        std::vector<interfaces::Detection>& detections,
        float conf_threshold = 0.5f,
        float nms_threshold = 0.45f
    ) override;

    void sync() override;

private:
    void preprocess(const uint8_t* image_data, int width, int height);
    void postprocess(std::vector<interfaces::Detection>& detections,
                     float conf_threshold, float nms_threshold);

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_;
    bool owns_stream_;

    // Buffers
    void* buffers_[2];
    float* output_host_;
    int input_h_, input_w_;
    int output_size_;
};

} // namespace aria::adapters::gpu
```

## Application Layer (Pipeline)

Orchestrates components using dependency injection.

### SlamPipeline.hpp

```cpp
#pragma once
#include "interfaces/IFeatureExtractor.hpp"
#include "interfaces/IMatcher.hpp"
#include "interfaces/ILoopDetector.hpp"
#include "interfaces/IObjectDetector.hpp"
#include "interfaces/ISensorFusion.hpp"
#include "interfaces/IMapper.hpp"
#include "core/Pose.hpp"
#include <memory>
#include <functional>

namespace aria::pipeline {

struct PipelineConfig {
    bool enable_loop_closure = true;
    bool enable_object_detection = true;
    bool enable_mapping = true;
    bool filter_dynamic_objects = true;

    // Camera intrinsics
    double fx = 700, fy = 700;
    double cx = 320, cy = 180;
};

class SlamPipeline {
public:
    // Dependency injection via constructor
    SlamPipeline(
        interfaces::FeatureExtractorPtr extractor,
        interfaces::MatcherPtr matcher,
        interfaces::LoopDetectorPtr loop_detector,
        interfaces::ObjectDetectorPtr object_detector,
        interfaces::SensorFusionPtr sensor_fusion,
        interfaces::MapperPtr mapper,
        const PipelineConfig& config = {}
    );

    ~SlamPipeline();

    // Process single frame
    // @param image_data RGB image data
    // @param width Image width
    // @param height Image height
    // @param timestamp Frame timestamp
    // @return Current pose estimate
    core::Pose processFrame(
        const uint8_t* image_data,
        int width,
        int height,
        double timestamp
    );

    // Process IMU measurement
    void processIMU(const interfaces::ImuMeasurement& imu);

    // Get current state
    core::Pose getCurrentPose() const;
    const std::vector<core::Pose>& getTrajectory() const;
    const interfaces::IMapper& getMapper() const;

    // Callbacks for external consumers
    using PoseCallback = std::function<void(const core::Pose&)>;
    using LoopCallback = std::function<void(const interfaces::LoopCandidate&)>;

    void setPoseCallback(PoseCallback cb) { pose_callback_ = std::move(cb); }
    void setLoopCallback(LoopCallback cb) { loop_callback_ = std::move(cb); }

private:
    // Components (injected)
    interfaces::FeatureExtractorPtr extractor_;
    interfaces::MatcherPtr matcher_;
    interfaces::LoopDetectorPtr loop_detector_;
    interfaces::ObjectDetectorPtr object_detector_;
    interfaces::SensorFusionPtr sensor_fusion_;
    interfaces::MapperPtr mapper_;

    // Configuration
    PipelineConfig config_;
    Eigen::Matrix3d K_;  // Camera intrinsics

    // State
    std::unique_ptr<core::Frame> prev_frame_;
    core::Pose current_pose_;
    std::vector<core::Pose> trajectory_;
    uint64_t frame_id_ = 0;

    // Callbacks
    PoseCallback pose_callback_;
    LoopCallback loop_callback_;

    // Internal methods
    void filterDynamicKeypoints(
        core::Frame& frame,
        const std::vector<interfaces::Detection>& detections
    );

    core::Pose estimatePose(
        const core::Frame& prev,
        const core::Frame& curr,
        const std::vector<interfaces::Match>& matches
    );
};

} // namespace aria::pipeline
```

## Factory (Dependency Injection)

Create pipeline with different configurations.

### PipelineFactory.hpp

```cpp
#pragma once
#include "pipeline/SlamPipeline.hpp"
#include <string>

namespace aria::factory {

enum class ExecutionMode {
    GPU,        // Full GPU acceleration (production)
    CPU,        // CPU-only (debugging, profiling)
    MOCK        // Mock components (unit testing)
};

struct FactoryConfig {
    ExecutionMode mode = ExecutionMode::GPU;

    // GPU settings
    std::string yolo_engine_path = "../models/yolo26s.engine";
    int cuda_device = 0;

    // Feature extraction
    int max_features = 1000;

    // Pipeline config
    pipeline::PipelineConfig pipeline_config;
};

class PipelineFactory {
public:
    static std::unique_ptr<pipeline::SlamPipeline> create(const FactoryConfig& config);

    // Convenience methods
    static std::unique_ptr<pipeline::SlamPipeline> createGpu(
        const std::string& yolo_engine = "../models/yolo26s.engine"
    );

    static std::unique_ptr<pipeline::SlamPipeline> createCpu();

    static std::unique_ptr<pipeline::SlamPipeline> createMock();
};

} // namespace aria::factory
```

### Usage Example

```cpp
#include "factory/PipelineFactory.hpp"

int main() {
    // Production: full GPU
    auto pipeline = aria::factory::PipelineFactory::createGpu();

    // Or with custom config
    aria::factory::FactoryConfig config;
    config.mode = aria::factory::ExecutionMode::GPU;
    config.max_features = 2000;
    config.pipeline_config.filter_dynamic_objects = true;

    auto custom_pipeline = aria::factory::PipelineFactory::create(config);

    // Process frames
    while (auto frame = capture.getFrame()) {
        auto pose = pipeline->processFrame(
            frame.data, frame.width, frame.height, frame.timestamp
        );
        std::cout << "Position: " << pose.position.transpose() << std::endl;
    }

    // Export map
    pipeline->getMapper().exportPLY("map.ply");
}
```

## Testing with Mocks

```cpp
#include "factory/PipelineFactory.hpp"
#include <gtest/gtest.h>

TEST(SlamPipeline, ProcessFrameReturnsPose) {
    // Create pipeline with mock components
    auto pipeline = aria::factory::PipelineFactory::createMock();

    // Create test image
    std::vector<uint8_t> test_image(640 * 480 * 3, 128);

    // Process frame
    auto pose = pipeline->processFrame(test_image.data(), 640, 480, 0.0);

    // Verify pose is valid
    EXPECT_FALSE(pose.position.hasNaN());
    EXPECT_NEAR(pose.orientation.norm(), 1.0, 1e-6);
}
```

## SOLID Principles Summary

| Principle | How Applied |
|-----------|-------------|
| **S**ingle Responsibility | `OrbCudaExtractor` only extracts, `CudaMatcher` only matches |
| **O**pen/Closed | Add `SuperPointExtractor` without modifying `SlamPipeline` |
| **L**iskov Substitution | `CudaMatcher` and `BFMatcher` are interchangeable |
| **I**nterface Segregation | `IFeatureExtractor` != `IMatcher` != `IObjectDetector` |
| **D**ependency Inversion | `SlamPipeline` depends on `IFeatureExtractor`, not `OrbCudaExtractor` |

## Migration Plan

1. **Create interfaces** in `include/interfaces/` (no code changes)
2. **Create domain entities** in `include/core/` (copy existing structs)
3. **Wrap existing code** in adapters (minimal changes)
4. **Create SlamPipeline** that uses interfaces
5. **Update main.cpp** to use factory
6. **Add tests** with mocks

Each step is a separate commit, maintaining a working build throughout.

## File Structure After H12

```
include/
├── core/
│   ├── Frame.hpp
│   ├── KeyFrame.hpp
│   ├── MapPoint.hpp
│   └── Pose.hpp
├── interfaces/
│   ├── IFeatureExtractor.hpp
│   ├── IMatcher.hpp
│   ├── ILoopDetector.hpp
│   ├── IObjectDetector.hpp
│   ├── ISensorFusion.hpp
│   └── IMapper.hpp
├── adapters/
│   ├── gpu/
│   │   ├── OrbCudaExtractor.hpp
│   │   ├── CudaMatcher.hpp
│   │   └── YoloTrtDetector.hpp
│   ├── cpu/
│   │   ├── OrbCpuExtractor.hpp
│   │   └── BruteForceMatcher.hpp
│   └── sensors/
│       └── EuRoCReader.hpp
├── pipeline/
│   └── SlamPipeline.hpp
└── factory/
    └── PipelineFactory.hpp
```

## Next Steps

After H12 is complete:
- **H13**: Add `LoopClosureThread` for async loop detection
- **H14**: Migrate loop closure matching to GPU
- **H16**: Add GoogleTest with mock-based unit tests
