#include "Mapper.hpp"
#include <iostream>
#include <cmath>

Mapper::Mapper(const cv::Mat& K) : K_(K.clone()) {}

int Mapper::triangulate(const std::vector<cv::KeyPoint>& kp1,
                        const std::vector<cv::KeyPoint>& kp2,
                        const std::vector<cv::DMatch>& matches,
                        const Eigen::Matrix4d& pose1,
                        const Eigen::Matrix4d& pose2,
                        const cv::Mat& image1) {
    if (matches.size() < 8) return 0;

    // Build projection matrices P = K * [R|t]
    cv::Mat R1(3, 3, CV_64F), t1(3, 1, CV_64F);
    cv::Mat R2(3, 3, CV_64F), t2(3, 1, CV_64F);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R1.at<double>(i, j) = pose1(i, j);
            R2.at<double>(i, j) = pose2(i, j);
        }
        t1.at<double>(i) = pose1(i, 3);
        t2.at<double>(i) = pose2(i, 3);
    }

    cv::Mat P1(3, 4, CV_64F), P2(3, 4, CV_64F);
    cv::hconcat(R1, t1, P1);
    cv::hconcat(R2, t2, P2);
    P1 = K_ * P1;
    P2 = K_ * P2;

    // Extract matched points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    // Triangulate
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, pts1, pts2, points4D);

    // Camera centers for parallax check
    Eigen::Vector3d C1 = -pose1.block<3,3>(0,0).transpose() * pose1.block<3,1>(0,3);
    Eigen::Vector3d C2 = -pose2.block<3,3>(0,0).transpose() * pose2.block<3,1>(0,3);

    int added = 0;
    for (int i = 0; i < points4D.cols; i++) {
        // Convert from homogeneous
        double w = points4D.at<double>(3, i);
        if (std::abs(w) < 1e-10) continue;

        Eigen::Vector3d pt(
            points4D.at<double>(0, i) / w,
            points4D.at<double>(1, i) / w,
            points4D.at<double>(2, i) / w
        );

        // Check depth (point should be in front of both cameras)
        Eigen::Vector3d pt_cam1 = pose1.block<3,3>(0,0) * pt + pose1.block<3,1>(0,3);
        Eigen::Vector3d pt_cam2 = pose2.block<3,3>(0,0) * pt + pose2.block<3,1>(0,3);

        if (pt_cam1.z() < min_depth_ || pt_cam1.z() > max_depth_) continue;
        if (pt_cam2.z() < min_depth_ || pt_cam2.z() > max_depth_) continue;

        // Check parallax angle
        Eigen::Vector3d ray1 = (pt - C1).normalized();
        Eigen::Vector3d ray2 = (pt - C2).normalized();
        double cos_parallax = ray1.dot(ray2);
        double parallax_deg = std::acos(std::min(1.0, std::abs(cos_parallax))) * 180.0 / M_PI;

        if (parallax_deg < min_parallax_) continue;

        // Check reprojection error
        double fx = K_.at<double>(0, 0);
        double fy = K_.at<double>(1, 1);
        double cx = K_.at<double>(0, 2);
        double cy = K_.at<double>(1, 2);

        double u1_proj = fx * pt_cam1.x() / pt_cam1.z() + cx;
        double v1_proj = fy * pt_cam1.y() / pt_cam1.z() + cy;
        double err1 = std::sqrt(std::pow(u1_proj - pts1[i].x, 2) +
                                std::pow(v1_proj - pts1[i].y, 2));

        double u2_proj = fx * pt_cam2.x() / pt_cam2.z() + cx;
        double v2_proj = fy * pt_cam2.y() / pt_cam2.z() + cy;
        double err2 = std::sqrt(std::pow(u2_proj - pts2[i].x, 2) +
                                std::pow(v2_proj - pts2[i].y, 2));

        if (err1 > max_reproj_error_ || err2 > max_reproj_error_) continue;

        // Get color from image if available
        Eigen::Vector3d color(0.5, 0.5, 0.5);  // Default gray
        if (!image1.empty()) {
            int px = std::clamp((int)pts1[i].x, 0, image1.cols - 1);
            int py = std::clamp((int)pts1[i].y, 0, image1.rows - 1);

            if (image1.channels() == 3) {
                cv::Vec3b bgr = image1.at<cv::Vec3b>(py, px);
                color = Eigen::Vector3d(bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0);
            } else {
                uchar gray = image1.at<uchar>(py, px);
                color = Eigen::Vector3d(gray / 255.0, gray / 255.0, gray / 255.0);
            }
        }

        // Add point
        MapPoint mp;
        mp.id = next_id_++;
        mp.position = pt;
        mp.color = color;
        mp.observations = 1;
        mp.quality = 1.0 / (err1 + err2 + 0.1);  // Higher quality = lower error

        points_.push_back(mp);
        added++;
    }

    return added;
}

void Mapper::addPoint(const Eigen::Vector3d& position, const Eigen::Vector3d& color) {
    MapPoint mp;
    mp.id = next_id_++;
    mp.position = position;
    mp.color = color;
    mp.observations = 1;
    mp.quality = 1.0;
    points_.push_back(mp);
}

void Mapper::filterOutliers(double max_reproj_error) {
    // Simple statistical outlier removal
    if (points_.size() < 10) return;

    // Compute mean position
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& p : points_) {
        mean += p.position;
    }
    mean /= points_.size();

    // Compute standard deviation
    double variance = 0;
    for (const auto& p : points_) {
        variance += (p.position - mean).squaredNorm();
    }
    double stddev = std::sqrt(variance / points_.size());

    // Remove points beyond 3 sigma
    double threshold = 3.0 * stddev;
    size_t before = points_.size();

    points_.erase(
        std::remove_if(points_.begin(), points_.end(),
            [&](const MapPoint& p) {
                return (p.position - mean).norm() > threshold;
            }),
        points_.end()
    );

    std::cout << "Filtered outliers: " << before << " -> " << points_.size() << std::endl;
}

void Mapper::filterByDistance(double max_distance) {
    points_.erase(
        std::remove_if(points_.begin(), points_.end(),
            [max_distance](const MapPoint& p) {
                return p.position.norm() > max_distance;
            }),
        points_.end()
    );
}

void Mapper::clear() {
    points_.clear();
    next_id_ = 0;
}

bool Mapper::exportPLY(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    // PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points_.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";

    // Points
    for (const auto& p : points_) {
        int r = std::clamp((int)(p.color.x() * 255), 0, 255);
        int g = std::clamp((int)(p.color.y() * 255), 0, 255);
        int b = std::clamp((int)(p.color.z() * 255), 0, 255);

        file << p.position.x() << " "
             << p.position.y() << " "
             << p.position.z() << " "
             << r << " " << g << " " << b << "\n";
    }

    file.close();
    std::cout << "Exported " << points_.size() << " points to " << filename << std::endl;
    return true;
}

bool Mapper::exportPCD(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    // PCD header
    file << "# .PCD v0.7 - Point Cloud Data\n";
    file << "VERSION 0.7\n";
    file << "FIELDS x y z rgb\n";
    file << "SIZE 4 4 4 4\n";
    file << "TYPE F F F U\n";
    file << "COUNT 1 1 1 1\n";
    file << "WIDTH " << points_.size() << "\n";
    file << "HEIGHT 1\n";
    file << "VIEWPOINT 0 0 0 1 0 0 0\n";
    file << "POINTS " << points_.size() << "\n";
    file << "DATA ascii\n";

    // Points
    for (const auto& p : points_) {
        int r = std::clamp((int)(p.color.x() * 255), 0, 255);
        int g = std::clamp((int)(p.color.y() * 255), 0, 255);
        int b = std::clamp((int)(p.color.z() * 255), 0, 255);

        // Pack RGB into single uint32
        uint32_t rgb = ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;

        file << p.position.x() << " "
             << p.position.y() << " "
             << p.position.z() << " "
             << rgb << "\n";
    }

    file.close();
    std::cout << "Exported " << points_.size() << " points to " << filename << std::endl;
    return true;
}

void Mapper::getBoundingBox(Eigen::Vector3d& min_pt, Eigen::Vector3d& max_pt) const {
    if (points_.empty()) {
        min_pt = max_pt = Eigen::Vector3d::Zero();
        return;
    }

    min_pt = max_pt = points_[0].position;
    for (const auto& p : points_) {
        min_pt = min_pt.cwiseMin(p.position);
        max_pt = max_pt.cwiseMax(p.position);
    }
}
