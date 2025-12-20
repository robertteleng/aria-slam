#pragma once
#include <Eigen/Dense>
#include <vector>
#include <deque>

struct IMUMeasurement {
    double timestamp;
    Eigen::Vector3d accel;  // Accelerometer (m/s^2)
    Eigen::Vector3d gyro;   // Gyroscope (rad/s)
};

struct IMUBias {
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
};

class IMUPreintegrator {
public:
    IMUPreintegrator(const Eigen::Vector3d& gravity = Eigen::Vector3d(0, 0, -9.81));

    void reset();
    void integrate(const IMUMeasurement& measurement);
    void setBias(const IMUBias& bias);

    // Preintegrated measurements
    Eigen::Vector3d getDeltaPosition() const { return delta_p_; }
    Eigen::Vector3d getDeltaVelocity() const { return delta_v_; }
    Eigen::Quaterniond getDeltaRotation() const { return delta_q_; }
    double getDeltaTime() const { return dt_sum_; }

    // Covariance for uncertainty
    Eigen::Matrix<double, 9, 9> getCovariance() const { return covariance_; }

private:
    Eigen::Vector3d gravity_;
    IMUBias bias_;

    // Preintegrated state
    Eigen::Vector3d delta_p_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d delta_v_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond delta_q_ = Eigen::Quaterniond::Identity();
    double dt_sum_ = 0.0;
    double last_timestamp_ = -1.0;

    // Covariance propagation
    Eigen::Matrix<double, 9, 9> covariance_ = Eigen::Matrix<double, 9, 9>::Zero();

    // Noise parameters (typical MEMS IMU)
    double accel_noise_ = 0.01;  // m/s^2/sqrt(Hz)
    double gyro_noise_ = 0.001;  // rad/s/sqrt(Hz)
};

class SensorFusion {
public:
    SensorFusion();

    void addIMU(const IMUMeasurement& imu);
    void addVisualPose(double timestamp, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

    // Get fused state
    Eigen::Vector3d getPosition() const { return position_; }
    Eigen::Vector3d getVelocity() const { return velocity_; }
    Eigen::Quaterniond getOrientation() const { return orientation_; }

    bool isInitialized() const { return initialized_; }

private:
    void predictIMU(const IMUMeasurement& imu);
    void updateVisual(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

    // State
    Eigen::Vector3d position_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation_ = Eigen::Quaterniond::Identity();

    // IMU buffer for preintegration
    std::deque<IMUMeasurement> imu_buffer_;
    IMUPreintegrator preintegrator_;
    IMUBias bias_;

    // Kalman filter state
    Eigen::Matrix<double, 15, 15> P_ = Eigen::Matrix<double, 15, 15>::Identity();

    // Gravity in world frame
    Eigen::Vector3d gravity_{0, 0, -9.81};

    double last_visual_time_ = -1.0;
    bool initialized_ = false;
};
