#include "Frame.hpp"

Frame::Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb) {
    image = img.clone();
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

Frame::Frame(const Frame& other) {
    image = other.image.clone();
    keypoints = other.keypoints;
    descriptors = other.descriptors.clone();
}