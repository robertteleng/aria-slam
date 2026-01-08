#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>
#include <string>

struct MapPoint {
    Eigen::Vector3d position;
    cv::Vec3b color;
};

class Mapper {
public:
    Mapper();
    
    void triangulatePoints(const std::vector<cv::Point2f>& pts1,
                           const std::vector<cv::Point2f>& pts2,
                           const cv::Mat& P1, const cv::Mat& P2,
                           const cv::Mat& img1);
    
    void savePLY(const std::string& filename) const;
    
    std::vector<MapPoint> points_;
};
