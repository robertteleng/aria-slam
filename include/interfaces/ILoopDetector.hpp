#pragma once
#include "core/Types.hpp"
#include <memory>
#include <optional>

namespace aria::interfaces {

/// Abstract interface for loop closure detection
/// Implementations: LoopDetectorGpu (GPU), MockLoopDetector (test)
class ILoopDetector {
public:
    virtual ~ILoopDetector() = default;

    /// Add keyframe to database
    virtual void addKeyFrame(const core::KeyFrame& kf) = 0;

    /// Detect loop closure
    /// @param query Current keyframe
    /// @return Loop candidate if found, nullopt otherwise
    virtual std::optional<core::LoopCandidate> detect(const core::KeyFrame& query) = 0;

    /// Get number of detected loops
    virtual int getLoopCount() const = 0;

    /// Configuration
    virtual void setMinFramesBetween(int n) = 0;
    virtual void setMinScore(double s) = 0;
    virtual void setMinMatches(int n) = 0;
};

using LoopDetectorPtr = std::unique_ptr<ILoopDetector>;

} // namespace aria::interfaces
