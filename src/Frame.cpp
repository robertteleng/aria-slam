#include "Frame.hpp"

Frame::Frame(const cv::Mat& image) : image_(image.clone()) {
    orb_ = cv::ORB::create(2000);
}

void Frame::extractFeatures() {
    cv::Mat gray;
    if (image_.channels() == 3) {
        cv::cvtColor(image_, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image_;
    }
    
    orb_->detectAndCompute(gray, cv::noArray(), keypoints_, descriptors_);
}
