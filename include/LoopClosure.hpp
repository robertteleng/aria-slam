#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

class LoopClosure {
public:
    LoopClosure();
    
    void addKeyframe(int id, const cv::Mat& descriptors);
    int detectLoop(const cv::Mat& descriptors);
    
private:
    std::vector<cv::Mat> keyframeDescriptors_;
    std::vector<int> keyframeIds_;
};
