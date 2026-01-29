#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cstdint>

namespace aria::core {

/// Keypoint in image coordinates (technology-agnostic)
struct KeyPoint {
    float x, y;           // Position in image
    float size;           // Diameter of meaningful keypoint neighborhood
    float angle;          // Orientation in degrees [0, 360)
    float response;       // Response by which the keypoints are sorted
    int octave;           // Octave (pyramid layer) from which keypoint was extracted
};

/// Visual frame with extracted features
struct Frame {
    uint64_t id = 0;
    double timestamp = 0.0;
    int width = 0;
    int height = 0;

    std::vector<KeyPoint> keypoints;
    std::vector<uint8_t> descriptors;  // Flattened: N x 32 bytes for ORB

    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

    size_t descriptorSize() const { return 32; }  // ORB descriptor size
    size_t numKeypoints() const { return keypoints.size(); }
};

/// Keyframe for loop closure and mapping
struct KeyFrame {
    uint64_t id = 0;
    double timestamp = 0.0;

    Frame frame;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();

    std::vector<uint64_t> covisible_keyframes;
    std::vector<uint64_t> observed_mappoints;
};

/// 3D map point
struct MapPoint {
    uint64_t id = 0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();

    std::vector<uint8_t> descriptor;  // Representative descriptor

    // Observations: keyframe_id -> keypoint_index
    std::vector<std::pair<uint64_t, int>> observations;

    int num_observations = 0;
    float min_distance = 0.0f;
    float max_distance = 0.0f;

    bool is_bad = false;
};

/// Camera pose with uncertainty
struct Pose {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    double timestamp = 0.0;

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

/// IMU measurement
struct ImuMeasurement {
    double timestamp = 0.0;
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();   // m/s^2
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();    // rad/s
};

/// Feature match between two frames
struct Match {
    int query_idx;      // Index in query frame
    int train_idx;      // Index in train frame
    float distance;     // Descriptor distance
};

/// Object detection result
struct Detection {
    float x1, y1, x2, y2;   // Bounding box
    float confidence;
    int class_id;

    bool contains(float x, float y) const {
        return x >= x1 && x <= x2 && y >= y1 && y <= y2;
    }
};

/// Loop closure candidate
struct LoopCandidate {
    uint64_t query_id;
    uint64_t match_id;
    double score;
    std::vector<Match> matches;
    Eigen::Matrix4d relative_pose = Eigen::Matrix4d::Identity();
};

} // namespace aria::core
