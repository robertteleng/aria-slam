#include "Frame.hpp"

// GPU version - ORB detection on CUDA
Frame::Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu) {
    image = img.clone();

    // Convert to grayscale if needed
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // Upload to GPU
    cv::cuda::GpuMat gpu_img(gray);
    cv::cuda::GpuMat gpu_keypoints, gpu_descriptors;

    // Detect and compute on GPU
    orb_gpu->detectAndComputeAsync(gpu_img, cv::cuda::GpuMat(), gpu_keypoints, gpu_descriptors);

    // Download results to CPU
    orb_gpu->convert(gpu_keypoints, keypoints);
    gpu_descriptors.download(descriptors);
}

// CPU version (legacy)
Frame::Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb) {
    image = img.clone();
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

Frame::Frame(const Frame& other) {
    image = other.image.clone();
    keypoints = other.keypoints;
    descriptors = other.descriptors.clone();
}