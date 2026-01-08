#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};

class TRTInference {
public:
    TRTInference(const std::string& enginePath);
    ~TRTInference();
    
    std::vector<Detection> detect(const cv::Mat& image);
};
