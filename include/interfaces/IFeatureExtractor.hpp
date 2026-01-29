#pragma once
#include "core/Types.hpp"
#include <memory>

namespace aria::interfaces {

/// Abstract interface for feature extraction
/// Implementations: OrbCudaExtractor (GPU), OrbCpuExtractor (CPU), MockExtractor (test)
class IFeatureExtractor {
public:
    virtual ~IFeatureExtractor() = default;

    /// Extract keypoints and descriptors from raw image data
    /// @param image_data Raw pixel data (grayscale, row-major)
    /// @param width Image width
    /// @param height Image height
    /// @param frame Output frame with keypoints and descriptors
    virtual void extract(
        const uint8_t* image_data,
        int width,
        int height,
        core::Frame& frame
    ) = 0;

    /// Async extraction (for GPU implementations)
    /// Returns immediately, results available after sync()
    virtual void extractAsync(
        const uint8_t* image_data,
        int width,
        int height,
        core::Frame& frame
    ) { extract(image_data, width, height, frame); }  // Default: sync

    /// Wait for async operation to complete
    virtual void sync() {}

    /// Configuration
    virtual void setMaxFeatures(int n) = 0;
    virtual int getMaxFeatures() const = 0;
};

using FeatureExtractorPtr = std::unique_ptr<IFeatureExtractor>;

} // namespace aria::interfaces
