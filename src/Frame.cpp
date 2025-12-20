#include "Frame.hpp"

// Constructor implementation
Frame::Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb) {
    image = img.clone();  // copia imagen
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}