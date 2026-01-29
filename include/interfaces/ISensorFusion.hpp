#pragma once
#include "core/Types.hpp"
#include <memory>

namespace aria::interfaces {

/// Abstract interface for sensor fusion (IMU + Visual Odometry)
/// Implementations: EKFSensorFusion, MockSensorFusion (test)
class ISensorFusion {
public:
    virtual ~ISensorFusion() = default;

    /// IMU prediction step (high frequency: 200Hz)
    virtual void predictIMU(const core::ImuMeasurement& imu) = 0;

    /// Visual odometry update step (low frequency: 30Hz)
    virtual void updateVO(const core::Pose& vo_pose) = 0;

    /// Get current fused state
    virtual core::Pose getFusedPose() const = 0;

    /// Get velocity estimate
    virtual Eigen::Vector3d getVelocity() const = 0;

    /// Reset filter
    virtual void reset() = 0;
    virtual void reset(const core::Pose& initial_pose) = 0;
};

using SensorFusionPtr = std::unique_ptr<ISensorFusion>;

} // namespace aria::interfaces
