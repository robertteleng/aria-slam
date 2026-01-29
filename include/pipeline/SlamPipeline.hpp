#pragma once
#include "interfaces/IFeatureExtractor.hpp"
#include "interfaces/IMatcher.hpp"
#include "interfaces/ILoopDetector.hpp"
#include "interfaces/IObjectDetector.hpp"
#include "interfaces/ISensorFusion.hpp"
#include "interfaces/IMapper.hpp"
#include "core/Types.hpp"
#include <memory>
#include <functional>
#include <vector>

namespace aria::pipeline {

/// Pipeline configuration
struct PipelineConfig {
    bool enable_loop_closure = true;
    bool enable_object_detection = true;
    bool enable_mapping = true;
    bool filter_dynamic_objects = true;

    // Camera intrinsics
    double fx = 700, fy = 700;
    double cx = 320, cy = 180;
};

/// Main SLAM pipeline orchestrator
/// Uses dependency injection for all components
class SlamPipeline {
public:
    /// Dependency injection via constructor
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

    /// Process single frame
    /// @param image_data RGB image data
    /// @param width Image width
    /// @param height Image height
    /// @param timestamp Frame timestamp
    /// @return Current pose estimate
    core::Pose processFrame(
        const uint8_t* image_data,
        int width,
        int height,
        double timestamp
    );

    /// Process IMU measurement
    void processIMU(const core::ImuMeasurement& imu);

    /// Get current state
    core::Pose getCurrentPose() const;
    const std::vector<core::Pose>& getTrajectory() const;
    const interfaces::IMapper* getMapper() const;

    /// Callbacks for external consumers
    using PoseCallback = std::function<void(const core::Pose&)>;
    using LoopCallback = std::function<void(const core::LoopCandidate&)>;

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
        const std::vector<core::Detection>& detections
    );

    core::Pose estimatePose(
        const core::Frame& prev,
        const core::Frame& curr,
        const std::vector<core::Match>& matches
    );
};

} // namespace aria::pipeline
