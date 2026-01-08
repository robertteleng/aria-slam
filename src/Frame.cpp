#include "Frame.hpp"

Frame::Frame(const cv::Mat& image) : image_(image.clone()) {
    orb_ = cv::cuda::ORB::create(2000);
}

void Frame::uploadToGPU() {
    cv::Mat gray;
    if (image_.channels() == 3) {
        cv::cvtColor(image_, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image_;
    }
    d_image_.upload(gray);
}

void Frame::extractFeatures() {
    uploadToGPU();
    
    cv::cuda::GpuMat d_keypoints, d_descriptors;
    orb_->detectAndComputeAsync(d_image_, cv::cuda::GpuMat(), d_keypoints, d_descriptors);
    
    orb_->convert(d_keypoints, keypoints_);
    d_descriptors.download(descriptors_);
}
