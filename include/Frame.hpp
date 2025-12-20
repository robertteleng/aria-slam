#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>

// Frame class with GPU-accelerated ORB detection
class Frame {
public:
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // GPU version (H5)
    Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu);

    // CPU version (legacy)
    Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb);

    Frame(const Frame& other);
};