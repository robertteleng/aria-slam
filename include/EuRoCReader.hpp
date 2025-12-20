#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "IMU.hpp"

/**
 * EuRoC MAV Dataset Reader
 *
 * Reads ASL format datasets containing:
 * - Stereo images (cam0, cam1)
 * - IMU measurements (imu0)
 * - Ground truth poses
 *
 * Reference: https://projects.asl.ethz.ch/datasets/
 */

struct EuRoCImage {
    double timestamp;        // Nanoseconds -> seconds
    std::string filename;
    cv::Mat image;
};

struct EuRoCGroundTruth {
    double timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d velocity;
    Eigen::Vector3d bias_gyro;
    Eigen::Vector3d bias_accel;
};

class EuRoCReader {
public:
    EuRoCReader(const std::string& dataset_path);

    // Load all data
    bool load();

    // Get next synchronized data
    bool getNext(cv::Mat& image, std::vector<IMUMeasurement>& imu_data,
                 double& timestamp);

    // Get ground truth at timestamp (interpolated)
    bool getGroundTruth(double timestamp, EuRoCGroundTruth& gt) const;

    // Get camera intrinsics
    cv::Mat getCameraMatrix() const { return K_; }

    // Dataset info
    size_t numImages() const { return images_.size(); }
    size_t numIMU() const { return imu_data_.size(); }
    size_t numGroundTruth() const { return ground_truth_.size(); }

    // Reset to beginning
    void reset() { current_idx_ = 0; last_imu_idx_ = 0; }

    // Check if more data available
    bool hasNext() const { return current_idx_ < images_.size(); }

private:
    std::string dataset_path_;

    // Data containers
    std::vector<EuRoCImage> images_;
    std::vector<IMUMeasurement> imu_data_;
    std::vector<EuRoCGroundTruth> ground_truth_;

    // Camera intrinsics (from sensor.yaml)
    cv::Mat K_;
    cv::Mat dist_coeffs_;

    // Current position in dataset
    size_t current_idx_ = 0;
    size_t last_imu_idx_ = 0;

    // Parsers
    bool loadImages(const std::string& cam_path);
    bool loadIMU(const std::string& imu_path);
    bool loadGroundTruth(const std::string& gt_path);
    bool loadCameraParams(const std::string& sensor_yaml);

    // Helpers
    double parseTimestamp(const std::string& ns_str) const {
        return std::stod(ns_str) * 1e-9;  // ns to seconds
    }
};
