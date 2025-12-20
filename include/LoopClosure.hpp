#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <Eigen/Dense>

struct KeyFrame {
    int id;
    double timestamp;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};

struct LoopCandidate {
    int query_id;
    int match_id;
    double score;
    std::vector<cv::DMatch> matches;
    Eigen::Matrix4d relative_pose;  // Transform from match to query
};

/**
 * Loop Closure Detection
 *
 * Detects when the camera returns to a previously visited location.
 * Uses descriptor matching to find similar keyframes and geometric
 * verification to confirm loop closures.
 */
class LoopClosureDetector {
public:
    LoopClosureDetector(int min_frames_between = 30,
                        double min_score = 0.3,
                        int min_matches = 30);

    // Add keyframe to database
    void addKeyFrame(const KeyFrame& kf);

    // Detect loop closure for current frame
    bool detect(const KeyFrame& query, LoopCandidate& candidate);

    // Get all keyframes
    const std::deque<KeyFrame>& getKeyFrames() const { return keyframes_; }

    // Get number of detected loops
    int getLoopCount() const { return loop_count_; }

private:
    // Find candidate keyframes by descriptor similarity
    std::vector<std::pair<int, double>> findCandidates(const KeyFrame& query);

    // Geometric verification using fundamental matrix
    bool verifyGeometry(const KeyFrame& query, const KeyFrame& candidate,
                        std::vector<cv::DMatch>& inlier_matches);

    // Compute relative pose between keyframes
    bool computeRelativePose(const KeyFrame& query, const KeyFrame& candidate,
                             const std::vector<cv::DMatch>& matches,
                             Eigen::Matrix4d& relative_pose);

    std::deque<KeyFrame> keyframes_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;

    // Parameters
    int min_frames_between_;  // Minimum frame gap to consider loop
    double min_score_;        // Minimum similarity score
    int min_matches_;         // Minimum inlier matches for valid loop

    int loop_count_ = 0;
};

/**
 * Pose Graph Optimizer using g2o
 *
 * Optimizes camera trajectory using loop closure constraints.
 * Uses Levenberg-Marquardt optimization on SE3 poses.
 */
class PoseGraphOptimizer {
public:
    PoseGraphOptimizer();
    ~PoseGraphOptimizer();

    // Add odometry edge (sequential constraint)
    void addOdometryEdge(int from_id, int to_id,
                         const Eigen::Matrix4d& relative_pose,
                         double info_scale = 1.0);

    // Add loop closure edge
    void addLoopEdge(int from_id, int to_id,
                     const Eigen::Matrix4d& relative_pose,
                     double info_scale = 1.0);

    // Set initial pose for vertex
    void setInitialPose(int id, const Eigen::Matrix4d& pose);

    // Run optimization
    void optimize(int iterations = 10);

    // Get optimized pose
    Eigen::Matrix4d getOptimizedPose(int id) const;

    // Get all optimized poses
    std::vector<Eigen::Matrix4d> getAllPoses() const;

    // Clear graph
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
