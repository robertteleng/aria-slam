#include "EuRoCReader.hpp"
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

EuRoCReader::EuRoCReader(const std::string& dataset_path)
    : dataset_path_(dataset_path) {

    // Default EuRoC camera intrinsics (cam0)
    // From MH_01_easy/mav0/cam0/sensor.yaml
    double fx = 458.654;
    double fy = 457.296;
    double cx = 367.215;
    double cy = 248.375;
    K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // Radial-tangential distortion
    dist_coeffs_ = (cv::Mat_<double>(4, 1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);
}

bool EuRoCReader::load() {
    std::cout << "Loading EuRoC dataset from: " << dataset_path_ << std::endl;

    // EuRoC structure: dataset_path/mav0/cam0, imu0, state_groundtruth_estimate0
    std::string mav_path = dataset_path_ + "/mav0";

    if (!fs::exists(mav_path)) {
        // Try without mav0 subdirectory
        mav_path = dataset_path_;
    }

    bool ok = true;

    // Load images from cam0
    std::string cam0_path = mav_path + "/cam0";
    if (!loadImages(cam0_path)) {
        std::cerr << "Failed to load images from " << cam0_path << std::endl;
        ok = false;
    }

    // Load IMU data
    std::string imu_path = mav_path + "/imu0";
    if (!loadIMU(imu_path)) {
        std::cerr << "Failed to load IMU from " << imu_path << std::endl;
        ok = false;
    }

    // Load ground truth
    std::string gt_path = mav_path + "/state_groundtruth_estimate0";
    if (!loadGroundTruth(gt_path)) {
        // Try alternative path
        gt_path = mav_path + "/leica0";
        if (!loadGroundTruth(gt_path)) {
            std::cerr << "Warning: No ground truth found" << std::endl;
        }
    }

    // Load camera parameters if available
    loadCameraParams(cam0_path + "/sensor.yaml");

    std::cout << "Loaded: " << images_.size() << " images, "
              << imu_data_.size() << " IMU measurements, "
              << ground_truth_.size() << " ground truth poses" << std::endl;

    return ok && !images_.empty();
}

bool EuRoCReader::loadImages(const std::string& cam_path) {
    std::string csv_path = cam_path + "/data.csv";
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        return false;
    }

    std::string line;
    std::getline(file, line);  // Skip header

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string timestamp_str, filename;

        std::getline(ss, timestamp_str, ',');
        std::getline(ss, filename, ',');

        // Remove whitespace
        filename.erase(0, filename.find_first_not_of(" \t"));
        filename.erase(filename.find_last_not_of(" \t\r\n") + 1);

        EuRoCImage img;
        img.timestamp = parseTimestamp(timestamp_str);
        img.filename = cam_path + "/data/" + filename;

        images_.push_back(img);
    }

    // Sort by timestamp
    std::sort(images_.begin(), images_.end(),
              [](const EuRoCImage& a, const EuRoCImage& b) {
                  return a.timestamp < b.timestamp;
              });

    return !images_.empty();
}

bool EuRoCReader::loadIMU(const std::string& imu_path) {
    std::string csv_path = imu_path + "/data.csv";
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        return false;
    }

    std::string line;
    std::getline(file, line);  // Skip header

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() < 7) continue;

        IMUMeasurement imu;
        imu.timestamp = parseTimestamp(tokens[0]);

        // EuRoC format: timestamp, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
        imu.gyro.x() = std::stod(tokens[1]);
        imu.gyro.y() = std::stod(tokens[2]);
        imu.gyro.z() = std::stod(tokens[3]);
        imu.accel.x() = std::stod(tokens[4]);
        imu.accel.y() = std::stod(tokens[5]);
        imu.accel.z() = std::stod(tokens[6]);

        imu_data_.push_back(imu);
    }

    // Sort by timestamp
    std::sort(imu_data_.begin(), imu_data_.end(),
              [](const IMUMeasurement& a, const IMUMeasurement& b) {
                  return a.timestamp < b.timestamp;
              });

    return !imu_data_.empty();
}

bool EuRoCReader::loadGroundTruth(const std::string& gt_path) {
    std::string csv_path = gt_path + "/data.csv";
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        return false;
    }

    std::string line;
    std::getline(file, line);  // Skip header

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() < 17) continue;

        EuRoCGroundTruth gt;
        gt.timestamp = parseTimestamp(tokens[0]);

        // Position
        gt.position.x() = std::stod(tokens[1]);
        gt.position.y() = std::stod(tokens[2]);
        gt.position.z() = std::stod(tokens[3]);

        // Orientation (quaternion: w, x, y, z)
        gt.orientation.w() = std::stod(tokens[4]);
        gt.orientation.x() = std::stod(tokens[5]);
        gt.orientation.y() = std::stod(tokens[6]);
        gt.orientation.z() = std::stod(tokens[7]);

        // Velocity
        gt.velocity.x() = std::stod(tokens[8]);
        gt.velocity.y() = std::stod(tokens[9]);
        gt.velocity.z() = std::stod(tokens[10]);

        // Biases
        gt.bias_gyro.x() = std::stod(tokens[11]);
        gt.bias_gyro.y() = std::stod(tokens[12]);
        gt.bias_gyro.z() = std::stod(tokens[13]);
        gt.bias_accel.x() = std::stod(tokens[14]);
        gt.bias_accel.y() = std::stod(tokens[15]);
        gt.bias_accel.z() = std::stod(tokens[16]);

        ground_truth_.push_back(gt);
    }

    // Sort by timestamp
    std::sort(ground_truth_.begin(), ground_truth_.end(),
              [](const EuRoCGroundTruth& a, const EuRoCGroundTruth& b) {
                  return a.timestamp < b.timestamp;
              });

    return !ground_truth_.empty();
}

bool EuRoCReader::loadCameraParams(const std::string& sensor_yaml) {
    // Simple YAML parser for camera intrinsics
    std::ifstream file(sensor_yaml);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    std::vector<double> intrinsics;
    std::vector<double> distortion;

    while (std::getline(file, line)) {
        // Look for intrinsics line: [fx, fy, cx, cy]
        if (line.find("intrinsics:") != std::string::npos) {
            size_t start = line.find('[');
            size_t end = line.find(']');
            if (start != std::string::npos && end != std::string::npos) {
                std::string values = line.substr(start + 1, end - start - 1);
                std::stringstream ss(values);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    intrinsics.push_back(std::stod(token));
                }
            }
        }

        // Look for distortion coefficients
        if (line.find("distortion_coefficients:") != std::string::npos) {
            size_t start = line.find('[');
            size_t end = line.find(']');
            if (start != std::string::npos && end != std::string::npos) {
                std::string values = line.substr(start + 1, end - start - 1);
                std::stringstream ss(values);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    distortion.push_back(std::stod(token));
                }
            }
        }
    }

    if (intrinsics.size() >= 4) {
        K_ = (cv::Mat_<double>(3, 3) <<
              intrinsics[0], 0, intrinsics[2],
              0, intrinsics[1], intrinsics[3],
              0, 0, 1);
        std::cout << "Loaded camera intrinsics: fx=" << intrinsics[0]
                  << ", fy=" << intrinsics[1] << std::endl;
    }

    if (distortion.size() >= 4) {
        dist_coeffs_ = cv::Mat(distortion).clone();
    }

    return true;
}

bool EuRoCReader::getNext(cv::Mat& image, std::vector<IMUMeasurement>& imu_data,
                          double& timestamp) {
    if (current_idx_ >= images_.size()) {
        return false;
    }

    // Load image
    EuRoCImage& img = images_[current_idx_];
    image = cv::imread(img.filename, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Failed to load: " << img.filename << std::endl;
        current_idx_++;
        return getNext(image, imu_data, timestamp);
    }

    timestamp = img.timestamp;

    // Collect IMU measurements between last image and current
    imu_data.clear();
    double prev_time = (current_idx_ > 0) ? images_[current_idx_ - 1].timestamp : 0;

    while (last_imu_idx_ < imu_data_.size() &&
           imu_data_[last_imu_idx_].timestamp <= timestamp) {
        if (imu_data_[last_imu_idx_].timestamp > prev_time) {
            imu_data.push_back(imu_data_[last_imu_idx_]);
        }
        last_imu_idx_++;
    }

    current_idx_++;
    return true;
}

bool EuRoCReader::getGroundTruth(double timestamp, EuRoCGroundTruth& gt) const {
    if (ground_truth_.empty()) {
        return false;
    }

    // Binary search for closest timestamp
    auto it = std::lower_bound(ground_truth_.begin(), ground_truth_.end(), timestamp,
                               [](const EuRoCGroundTruth& g, double t) {
                                   return g.timestamp < t;
                               });

    if (it == ground_truth_.end()) {
        gt = ground_truth_.back();
        return true;
    }

    if (it == ground_truth_.begin()) {
        gt = ground_truth_.front();
        return true;
    }

    // Interpolate between two closest poses
    auto prev = std::prev(it);
    double t0 = prev->timestamp;
    double t1 = it->timestamp;
    double alpha = (timestamp - t0) / (t1 - t0);

    gt.timestamp = timestamp;
    gt.position = (1 - alpha) * prev->position + alpha * it->position;
    gt.velocity = (1 - alpha) * prev->velocity + alpha * it->velocity;
    gt.orientation = prev->orientation.slerp(alpha, it->orientation);
    gt.bias_gyro = (1 - alpha) * prev->bias_gyro + alpha * it->bias_gyro;
    gt.bias_accel = (1 - alpha) * prev->bias_accel + alpha * it->bias_accel;

    return true;
}
