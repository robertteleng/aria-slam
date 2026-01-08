#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

class Frame {
public:
    Frame(const cv::Mat& image);
    void extractFeatures();
    void uploadToGPU();
    
    cv::Mat image_;
    cv::cuda::GpuMat d_image_;
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;

private:
    cv::Ptr<cv::cuda::ORB> orb_;
};
