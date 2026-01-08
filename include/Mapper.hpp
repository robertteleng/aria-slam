#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

struct MapPoint {
    Eigen::Vector3d position;
    cv::Vec3b color;
};

class Mapper {
public:
    Mapper();
    
    void triangulatePoints(const std::vector<cv::Point2f>& pts1,
                           const std::vector<cv::Point2f>& pts2,
                           const cv::Mat& P1, const cv::Mat& P2);
    
    std::vector<MapPoint> points_;
};
