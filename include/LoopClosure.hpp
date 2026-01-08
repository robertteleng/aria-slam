#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

struct Keyframe {
    int id;
    cv::Mat descriptors;
    Eigen::Isometry3d pose;
};

class LoopClosure {
public:
    LoopClosure();
    
    void addKeyframe(int id, const cv::Mat& descriptors, const Eigen::Isometry3d& pose);
    int detectLoop(const cv::Mat& descriptors);
    void optimize();
    
    std::vector<Eigen::Isometry3d> getOptimizedPoses() const;
    
private:
    std::vector<Keyframe> keyframes_;
    std::vector<std::pair<int, int>> loopEdges_;
    
    g2o::SparseOptimizer optimizer_;
};
