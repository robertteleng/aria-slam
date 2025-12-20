#pragma once              // evita incluir 2 veces
#include <opencv2/opencv.hpp>

// Frame class to hold image, keypoints, and descriptors
class Frame {
public:
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb);
    Frame(const Frame& other);  // Copy constructor
};