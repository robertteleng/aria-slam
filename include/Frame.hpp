#pragma once
#include <opencv2/opencv.hpp>

class Frame {
public:
    Frame(const cv::Mat& image);
    
    cv::Mat image_;
};
