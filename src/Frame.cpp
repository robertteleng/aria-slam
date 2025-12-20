#include "Frame.hpp"

// GPU ORB: uploads image to VRAM, detects features, downloads results
Frame::Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu) {
    image = img.clone();

    // ORB requires grayscale input
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // CPU -> GPU transfer
    cv::cuda::GpuMat gpu_img(gray);
    cv::cuda::GpuMat gpu_keypoints, gpu_descriptors;

    // Async detection on GPU (non-blocking)
    orb_gpu->detectAndComputeAsync(gpu_img, cv::cuda::GpuMat(), gpu_keypoints, this->gpu_descriptors);

    // GPU -> CPU transfer (keypoints needed for visualization)
    orb_gpu->convert(gpu_keypoints, keypoints);
    this->gpu_descriptors.download(descriptors);
}

// CPU ORB: fallback for systems without CUDA
Frame::Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb) {
    image = img.clone();
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

// Deep copy for frame history
Frame::Frame(const Frame& other) {
    image = other.image.clone();
    keypoints = other.keypoints;
    descriptors = other.descriptors.clone();
    other.gpu_descriptors.copyTo(gpu_descriptors);
}