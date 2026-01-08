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

/**
 * Extended Kalman Filter for Visual-Inertial Fusion
 *
 * State vector (15 dimensions):
 *   [0-2]   position (x, y, z)
 *   [3-5]   velocity (vx, vy, vz)
 *   [6-8]   orientation error (rotation vector)
 *   [9-11]  accelerometer bias
 *   [12-14] gyroscope bias
 */
class SensorFusion {
public:
    SensorFusion();

    void addIMU(const IMUMeasurement& imu);
    void addVisualPose(double timestamp, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

    // Get fused state
    Eigen::Vector3d getPosition() const { return position_; }
    Eigen::Vector3d getVelocity() const { return velocity_; }
    Eigen::Quaterniond getOrientation() const { return orientation_; }
    IMUBias getBias() const { return bias_; }
    Eigen::Matrix<double, 15, 15> getCovariance() const { return P_; }

    bool isInitialized() const { return initialized_; }

private:
    // EKF Predict step (IMU propagation)
    void predictEKF(const IMUMeasurement& imu);

    // EKF Update step (visual measurement)
    void updateEKF(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

    // State
    Eigen::Vector3d position_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation_ = Eigen::Quaterniond::Identity();

    // IMU buffer for preintegration
    std::deque<IMUMeasurement> imu_buffer_;
    IMUBias bias_;

    // EKF covariance matrix (15x15)
    Eigen::Matrix<double, 15, 15> P_;

    // Process noise covariance
    Eigen::Matrix<double, 12, 12> Q_;  // accel_noise, gyro_noise, accel_bias_walk, gyro_bias_walk

    // Measurement noise covariance
    Eigen::Matrix<double, 6, 6> R_meas_;  // position and orientation noise

    // Gravity in world frame
    Eigen::Vector3d gravity_{0, 0, -9.81};

    // Noise parameters
    double accel_noise_ = 0.1;        // m/s^2/sqrt(Hz)
    double gyro_noise_ = 0.01;        // rad/s/sqrt(Hz)
    double accel_bias_walk_ = 0.001;  // m/s^3/sqrt(Hz)
    double gyro_bias_walk_ = 0.0001;  // rad/s^2/sqrt(Hz)
    double pos_noise_ = 0.01;         // m
    double rot_noise_ = 0.01;         // rad

    double last_imu_time_ = -1.0;
    double last_visual_time_ = -1.0;
    bool initialized_ = false;
};
