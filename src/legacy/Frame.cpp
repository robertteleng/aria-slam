#include "legacy/Frame.hpp"
#include <opencv2/core/cuda_stream_accessor.hpp>

// GPU ORB: uploads image to VRAM, detects features
// H11: supports async execution with CUDA streams
Frame::Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu, cudaStream_t stream) {
    image = img.clone();
    orb_gpu_ = orb_gpu;

    // ORB requires grayscale input
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // CPU -> GPU transfer
    cv::cuda::GpuMat gpu_img(gray);

    if (stream) {
        // Async mode: use provided CUDA stream
        cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

        // Async detection on GPU (non-blocking)
        orb_gpu->detectAndComputeAsync(gpu_img, cv::cuda::GpuMat(),
                                        gpu_keypoints_, gpu_descriptors,
                                        false, cv_stream);
        // Results will be downloaded later via downloadResults()
        downloaded_ = false;
    } else {
        // Sync mode: original behavior (backward compatible)
        cv::cuda::GpuMat gpu_keypoints;
        orb_gpu->detectAndComputeAsync(gpu_img, cv::cuda::GpuMat(),
                                        gpu_keypoints, gpu_descriptors);

        // GPU -> CPU transfer (blocking)
        orb_gpu->convert(gpu_keypoints, keypoints);
        gpu_descriptors.download(descriptors);
        downloaded_ = true;
    }
}

// CPU ORB: fallback for systems without CUDA
Frame::Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb) {
    image = img.clone();
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    downloaded_ = true;  // CPU mode is always sync
}

// Deep copy for frame history
Frame::Frame(const Frame& other) {
    image = other.image.clone();
    keypoints = other.keypoints;
    descriptors = other.descriptors.clone();
    other.gpu_descriptors.copyTo(gpu_descriptors);
    orb_gpu_ = other.orb_gpu_;
    other.gpu_keypoints_.copyTo(gpu_keypoints_);
    downloaded_ = other.downloaded_;
}

// Download GPU results to CPU (call after stream sync)
void Frame::downloadResults() {
    if (downloaded_) return;

    if (orb_gpu_ && !gpu_keypoints_.empty()) {
        orb_gpu_->convert(gpu_keypoints_, keypoints);
    }
    if (!gpu_descriptors.empty()) {
        gpu_descriptors.download(descriptors);
    }
    downloaded_ = true;
}