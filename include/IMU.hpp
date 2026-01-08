#pragma once
#include <Eigen/Dense>

struct IMUData {
    double timestamp;
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
};

class IMU {
public:
    IMU();
    void predict(const IMUData& data, double dt);
    
    Eigen::Vector3d position_;
    Eigen::Vector3d velocity_;
};
