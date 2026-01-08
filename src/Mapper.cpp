#include "Mapper.hpp"

Mapper::Mapper() {}

void Mapper::triangulatePoints(const std::vector<cv::Point2f>& pts1,
                               const std::vector<cv::Point2f>& pts2,
                               const cv::Mat& P1, const cv::Mat& P2) {
    if (pts1.empty() || pts2.empty()) return;
    
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, pts1, pts2, points4D);
    
    for (int i = 0; i < points4D.cols; i++) {
        cv::Mat x = points4D.col(i);
        x /= x.at<float>(3);
        
        MapPoint mp;
        mp.position = Eigen::Vector3d(x.at<float>(0), x.at<float>(1), x.at<float>(2));
        mp.color = cv::Vec3b(255, 255, 255);
        points_.push_back(mp);
    }
}
