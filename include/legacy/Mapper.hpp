#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <fstream>

struct MapPoint {
    int id;
    Eigen::Vector3d position;
    Eigen::Vector3d color;  // RGB [0-1]
    int observations = 1;   // Number of times observed
    double quality = 1.0;   // Point quality score
};

/**
 * 3D Mapper
 *
 * Triangulates 2D feature matches into 3D map points.
 * Maintains a sparse point cloud of the environment.
 */
class Mapper {
public:
    Mapper(const cv::Mat& K);

    // Triangulate points from two frames
    int triangulate(const std::vector<cv::KeyPoint>& kp1,
                    const std::vector<cv::KeyPoint>& kp2,
                    const std::vector<cv::DMatch>& matches,
                    const Eigen::Matrix4d& pose1,
                    const Eigen::Matrix4d& pose2,
                    const cv::Mat& image1 = cv::Mat());

    // Add single point
    void addPoint(const Eigen::Vector3d& position,
                  const Eigen::Vector3d& color = Eigen::Vector3d(1, 1, 1));

    // Filter outliers based on reprojection error
    void filterOutliers(double max_reproj_error = 5.0);

    // Remove points far from camera
    void filterByDistance(double max_distance = 100.0);

    // Get all map points
    const std::vector<MapPoint>& getPoints() const { return points_; }

    // Get point count
    size_t size() const { return points_.size(); }

    // Clear all points
    void clear();

    // Export to PLY file
    bool exportPLY(const std::string& filename) const;

    // Export to PCD file (PCL format)
    bool exportPCD(const std::string& filename) const;

    // Get bounding box
    void getBoundingBox(Eigen::Vector3d& min_pt, Eigen::Vector3d& max_pt) const;

private:
    cv::Mat K_;  // Camera intrinsics
    std::vector<MapPoint> points_;
    int next_id_ = 0;

    // Triangulation parameters
    double min_parallax_ = 1.0;      // Minimum parallax angle (degrees)
    double max_reproj_error_ = 2.0;  // Maximum reprojection error (pixels)
    double min_depth_ = 0.1;         // Minimum depth (meters)
    double max_depth_ = 50.0;        // Maximum depth (meters)
};
