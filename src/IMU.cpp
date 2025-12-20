#include "IMU.hpp"
#include <iostream>

// Helper function for skew-symmetric matrix
static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return m;
}

// ============== IMUPreintegrator ==============

IMUPreintegrator::IMUPreintegrator(const Eigen::Vector3d& gravity)
    : gravity_(gravity) {
    reset();
}

void IMUPreintegrator::reset() {
    delta_p_ = Eigen::Vector3d::Zero();
    delta_v_ = Eigen::Vector3d::Zero();
    delta_q_ = Eigen::Quaterniond::Identity();
    dt_sum_ = 0.0;
    last_timestamp_ = -1.0;
    covariance_.setZero();
}

void IMUPreintegrator::setBias(const IMUBias& bias) {
    bias_ = bias;
}

void IMUPreintegrator::integrate(const IMUMeasurement& measurement) {
    if (last_timestamp_ < 0) {
        last_timestamp_ = measurement.timestamp;
        return;
    }

    double dt = measurement.timestamp - last_timestamp_;
    if (dt <= 0 || dt > 0.5) {
        last_timestamp_ = measurement.timestamp;
        return;
    }

    // Remove bias from measurements
    Eigen::Vector3d accel = measurement.accel - bias_.accel_bias;
    Eigen::Vector3d gyro = measurement.gyro - bias_.gyro_bias;

    // Rotation integration (mid-point)
    Eigen::Vector3d delta_angle = gyro * dt;
    double angle = delta_angle.norm();
    Eigen::Quaterniond dq;
    if (angle > 1e-10) {
        dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, delta_angle.normalized()));
    } else {
        dq = Eigen::Quaterniond::Identity();
    }

    // Rotate acceleration to initial frame
    Eigen::Vector3d accel_world = delta_q_ * accel;

    // Position and velocity integration
    delta_p_ += delta_v_ * dt + 0.5 * accel_world * dt * dt;
    delta_v_ += accel_world * dt;
    delta_q_ = delta_q_ * dq;
    delta_q_.normalize();

    // Covariance propagation (simplified)
    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(3, 6) = -delta_q_.toRotationMatrix() * skew(accel) * dt;

    Eigen::Matrix<double, 9, 6> G = Eigen::Matrix<double, 9, 6>::Zero();
    G.block<3, 3>(3, 0) = delta_q_.toRotationMatrix() * dt;
    G.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;

    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
    Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * accel_noise_ * accel_noise_;
    Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * gyro_noise_ * gyro_noise_;

    covariance_ = F * covariance_ * F.transpose() + G * Q * G.transpose();

    dt_sum_ += dt;
    last_timestamp_ = measurement.timestamp;
}

// ============== SensorFusion ==============

SensorFusion::SensorFusion() {
    P_.block<3, 3>(0, 0) *= 0.01;   // Position uncertainty
    P_.block<3, 3>(3, 3) *= 0.01;   // Velocity uncertainty
    P_.block<3, 3>(6, 6) *= 0.01;   // Orientation uncertainty
    P_.block<3, 3>(9, 9) *= 0.001;  // Accel bias uncertainty
    P_.block<3, 3>(12, 12) *= 0.0001; // Gyro bias uncertainty
}

void SensorFusion::addIMU(const IMUMeasurement& imu) {
    imu_buffer_.push_back(imu);

    // Limit buffer size
    while (imu_buffer_.size() > 1000) {
        imu_buffer_.pop_front();
    }

    if (initialized_) {
        predictIMU(imu);
    }
}

void SensorFusion::predictIMU(const IMUMeasurement& imu) {
    static double last_imu_time = -1;
    if (last_imu_time < 0) {
        last_imu_time = imu.timestamp;
        return;
    }

    double dt = imu.timestamp - last_imu_time;
    if (dt <= 0 || dt > 0.1) {
        last_imu_time = imu.timestamp;
        return;
    }

    // Remove bias
    Eigen::Vector3d accel = imu.accel - bias_.accel_bias;
    Eigen::Vector3d gyro = imu.gyro - bias_.gyro_bias;

    // Integrate orientation
    Eigen::Vector3d delta_angle = gyro * dt;
    double angle = delta_angle.norm();
    if (angle > 1e-10) {
        Eigen::Quaterniond dq(Eigen::AngleAxisd(angle, delta_angle.normalized()));
        orientation_ = orientation_ * dq;
        orientation_.normalize();
    }

    // Rotate acceleration to world frame and remove gravity
    Eigen::Vector3d accel_world = orientation_ * accel + gravity_;

    // Integrate velocity and position
    velocity_ += accel_world * dt;
    position_ += velocity_ * dt + 0.5 * accel_world * dt * dt;

    last_imu_time = imu.timestamp;
}

void SensorFusion::addVisualPose(double timestamp, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    if (!initialized_) {
        // Initialize from first visual pose
        position_ = t;
        orientation_ = Eigen::Quaterniond(R);
        velocity_ = Eigen::Vector3d::Zero();
        last_visual_time_ = timestamp;
        initialized_ = true;
        preintegrator_.reset();
        std::cout << "Sensor fusion initialized" << std::endl;
        return;
    }

    // Preintegrate IMU between visual frames
    preintegrator_.reset();
    for (const auto& imu : imu_buffer_) {
        if (imu.timestamp > last_visual_time_ && imu.timestamp <= timestamp) {
            preintegrator_.integrate(imu);
        }
    }

    // Simple fusion: weighted average between IMU prediction and visual
    double dt = timestamp - last_visual_time_;
    if (dt > 0 && dt < 1.0) {
        // Visual weight (higher = trust visual more)
        double visual_weight = 0.8;

        // IMU predicted position
        Eigen::Vector3d imu_predicted_pos = position_ + velocity_ * dt +
                                            preintegrator_.getDeltaPosition();

        // Fuse position
        position_ = visual_weight * t + (1.0 - visual_weight) * imu_predicted_pos;

        // Update velocity from visual (finite difference)
        Eigen::Vector3d visual_velocity = (t - position_) / dt;
        velocity_ = visual_weight * visual_velocity + (1.0 - visual_weight) * velocity_;

        // Fuse orientation (SLERP)
        Eigen::Quaterniond visual_q(R);
        orientation_ = orientation_.slerp(visual_weight, visual_q);
        orientation_.normalize();
    } else {
        // Large gap, reset to visual
        position_ = t;
        orientation_ = Eigen::Quaterniond(R);
    }

    // Clear old IMU data
    while (!imu_buffer_.empty() && imu_buffer_.front().timestamp < timestamp) {
        imu_buffer_.pop_front();
    }

    last_visual_time_ = timestamp;
    preintegrator_.reset();
}
