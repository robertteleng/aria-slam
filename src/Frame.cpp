#include "Frame.hpp"

Frame::Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb) {
    // IMPORTANT: clone the image and detect keypoints on the SAME image
    // If you detect on 'image' but draw on a different cv::Mat,
    // the keypoint coordinates won't match (black screen bug)
    image = img.clone();
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

Frame::Frame(const Frame& other) {
    image = other.image.clone();
    keypoints = other.keypoints;
    descriptors = other.descriptors.clone();
}