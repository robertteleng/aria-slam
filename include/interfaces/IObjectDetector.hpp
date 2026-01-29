#pragma once
#include "core/Types.hpp"
#include <memory>
#include <vector>

namespace aria::interfaces {

/// Abstract interface for object detection
/// Implementations: YoloTrtDetector (GPU), MockDetector (test)
class IObjectDetector {
public:
    virtual ~IObjectDetector() = default;

    /// Detect objects in image (synchronous)
    /// @param image_data RGB image data (row-major, 3 channels)
    /// @param width Image width
    /// @param height Image height
    /// @param detections Output detections
    /// @param conf_threshold Confidence threshold
    /// @param nms_threshold NMS IoU threshold
    virtual void detect(
        const uint8_t* image_data,
        int width,
        int height,
        std::vector<core::Detection>& detections,
        float conf_threshold = 0.5f,
        float nms_threshold = 0.45f
    ) = 0;

    /// Async detection (for GPU implementations)
    virtual void detectAsync(
        const uint8_t* image_data,
        int width,
        int height
    ) = 0;

    /// Get results after async detection
    virtual void getDetections(
        std::vector<core::Detection>& detections,
        float conf_threshold = 0.5f,
        float nms_threshold = 0.45f
    ) = 0;

    /// Wait for async operation to complete
    virtual void sync() = 0;
};

using ObjectDetectorPtr = std::unique_ptr<IObjectDetector>;

} // namespace aria::interfaces
