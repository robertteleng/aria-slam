#pragma once
#include <Eigen/Dense>

struct IMUData {
    double timestamp;
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
};

class EKF15State {
public:
    EKF15State();
    
    void predict(const IMUData& data, double dt);
    void updateVO(const Eigen::Vector3d& position, const Eigen::Matrix3d& R);
    
    // State: [position(3), velocity(3), orientation(4), accel_bias(3), gyro_bias(3)] = 15+1
    Eigen::Matrix<double, 16, 1> state_;
    Eigen::Matrix<double, 15, 15> P_; // Covariance
    
    Eigen::Vector3d getPosition() const;
    Eigen::Vector3d getVelocity() const;
    Eigen::Quaterniond getOrientation() const;
};
