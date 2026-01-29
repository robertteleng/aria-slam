#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <vector>

class Frame {
public:
    // GPU ORB con stream opcional
    Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu, cudaStream_t stream = nullptr);
    // CPU ORB fallback
    Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb);
    // Copy constructor
    Frame(const Frame& other);
    
    void downloadResults();
    
    // Public members
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::cuda::GpuMat gpu_descriptors;

private:
    cv::Ptr<cv::cuda::ORB> orb_gpu_;
    cv::cuda::GpuMat gpu_keypoints_;
    bool downloaded_ = false;
};