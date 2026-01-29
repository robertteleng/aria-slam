#include "legacy/IMU.hpp"
#include <iostream>

// Helper function for skew-symmetric matrix
static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return m;
}

// Convert rotation vector to quaternion
static Eigen::Quaterniond expMap(const Eigen::Vector3d& theta) {
    double angle = theta.norm();
    if (angle < 1e-10) {
        return Eigen::Quaterniond::Identity();
    }
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, theta.normalized()));
}

// Convert quaternion to rotation vector
static Eigen::Vector3d logMap(const Eigen::Quaterniond& q) {
    Eigen::AngleAxisd aa(q);
    return aa.angle() * aa.axis();
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

    // Covariance propagation
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

// ============== SensorFusion (EKF) ==============

SensorFusion::SensorFusion() {
    // Initialize state covariance
    P_.setIdentity();
    P_.block<3, 3>(0, 0) *= 0.01;     // Position (m^2)
    P_.block<3, 3>(3, 3) *= 0.01;     // Velocity (m/s)^2
    P_.block<3, 3>(6, 6) *= 0.01;     // Orientation (rad^2)
    P_.block<3, 3>(9, 9) *= 0.001;    // Accel bias
    P_.block<3, 3>(12, 12) *= 0.0001; // Gyro bias

    // Process noise (IMU noise)
    Q_.setZero();
    Q_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * accel_noise_ * accel_noise_;
    Q_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * gyro_noise_ * gyro_noise_;
    Q_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * accel_bias_walk_ * accel_bias_walk_;
    Q_.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * gyro_bias_walk_ * gyro_bias_walk_;

    // Measurement noise (visual)
    R_meas_.setZero();
    R_meas_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * pos_noise_ * pos_noise_;
    R_meas_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * rot_noise_ * rot_noise_;
}

void SensorFusion::addIMU(const IMUMeasurement& imu) {
    imu_buffer_.push_back(imu);

    // Limit buffer size
    while (imu_buffer_.size() > 1000) {
        imu_buffer_.pop_front();
    }

    if (initialized_) {
        predictEKF(imu);
    }
}

void SensorFusion::predictEKF(const IMUMeasurement& imu) {
    if (last_imu_time_ < 0) {
        last_imu_time_ = imu.timestamp;
        return;
    }

    double dt = imu.timestamp - last_imu_time_;
    if (dt <= 0 || dt > 0.1) {
        last_imu_time_ = imu.timestamp;
        return;
    }

    // Remove bias from measurements
    Eigen::Vector3d accel = imu.accel - bias_.accel_bias;
    Eigen::Vector3d gyro = imu.gyro - bias_.gyro_bias;

    // Get current rotation matrix
    Eigen::Matrix3d R = orientation_.toRotationMatrix();

    // ========== State Prediction ==========

    // Orientation prediction
    Eigen::Vector3d delta_angle = gyro * dt;
    double angle = delta_angle.norm();
    if (angle > 1e-10) {
        Eigen::Quaterniond dq(Eigen::AngleAxisd(angle, delta_angle.normalized()));
        orientation_ = orientation_ * dq;
        orientation_.normalize();
    }

    // Acceleration in world frame (remove gravity)
    Eigen::Vector3d accel_world = R * accel + gravity_;

    // Velocity and position prediction
    position_ += velocity_ * dt + 0.5 * accel_world * dt * dt;
    velocity_ += accel_world * dt;

    // ========== Covariance Prediction ==========

    // State transition Jacobian F (15x15)
    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Identity();

    // dp/dv
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;

    // dp/dtheta (position depends on orientation through acceleration)
    F.block<3, 3>(0, 6) = -0.5 * R * skew(accel) * dt * dt;

    // dp/dba (position depends on accel bias)
    F.block<3, 3>(0, 9) = -0.5 * R * dt * dt;

    // dv/dtheta
    F.block<3, 3>(3, 6) = -R * skew(accel) * dt;

    // dv/dba
    F.block<3, 3>(3, 9) = -R * dt;

    // dtheta/dbg
    F.block<3, 3>(6, 12) = -Eigen::Matrix3d::Identity() * dt;

    // Noise Jacobian G (15x12)
    Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();

    // Position affected by accel noise
    G.block<3, 3>(0, 0) = 0.5 * R * dt * dt;

    // Velocity affected by accel noise
    G.block<3, 3>(3, 0) = R * dt;

    // Orientation affected by gyro noise
    G.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;

    // Bias random walk
    G.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity() * dt;
    G.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity() * dt;

    // Propagate covariance: P = F * P * F' + G * Q * G'
    P_ = F * P_ * F.transpose() + G * Q_ * G.transpose();

    // Ensure symmetry
    P_ = 0.5 * (P_ + P_.transpose());

    last_imu_time_ = imu.timestamp;
}

void SensorFusion::addVisualPose(double timestamp, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    if (!initialized_) {
        // Initialize from first visual pose
        position_ = t;
        orientation_ = Eigen::Quaterniond(R);
        velocity_ = Eigen::Vector3d::Zero();
        last_visual_time_ = timestamp;
        last_imu_time_ = timestamp;
        initialized_ = true;
        std::cout << "EKF Sensor fusion initialized" << std::endl;
        return;
    }

    updateEKF(R, t);

    // Clear old IMU data
    while (!imu_buffer_.empty() && imu_buffer_.front().timestamp < timestamp) {
        imu_buffer_.pop_front();
    }

    last_visual_time_ = timestamp;
}

void SensorFusion::updateEKF(const Eigen::Matrix3d& R_meas, const Eigen::Vector3d& t_meas) {
    // ========== Measurement Model ==========
    // z = [position, orientation_error]
    // h(x) = [position, 0] (orientation error is zero when aligned)

    // Measurement Jacobian H (6x15)
    Eigen::Matrix<double, 6, 15> H = Eigen::Matrix<double, 6, 15>::Zero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();  // Position measurement
    H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();  // Orientation measurement

    // ========== Innovation ==========

    // Position innovation
    Eigen::Vector3d pos_innov = t_meas - position_;

    // Orientation innovation (error between measured and predicted)
    Eigen::Quaterniond q_meas(R_meas);
    Eigen::Quaterniond q_err = q_meas * orientation_.inverse();
    q_err.normalize();

    // Convert to rotation vector
    Eigen::Vector3d rot_innov = logMap(q_err);

    // Combined innovation
    Eigen::Matrix<double, 6, 1> innovation;
    innovation << pos_innov, rot_innov;

    // ========== Kalman Gain ==========
    // K = P * H' * (H * P * H' + R)^-1

    Eigen::Matrix<double, 6, 6> S = H * P_ * H.transpose() + R_meas_;
    Eigen::Matrix<double, 15, 6> K = P_ * H.transpose() * S.inverse();

    // ========== State Update ==========
    Eigen::Matrix<double, 15, 1> dx = K * innovation;

    // Update position
    position_ += dx.segment<3>(0);

    // Update velocity
    velocity_ += dx.segment<3>(3);

    // Update orientation (apply error quaternion)
    Eigen::Quaterniond dq = expMap(dx.segment<3>(6));
    orientation_ = dq * orientation_;
    orientation_.normalize();

    // Update biases
    bias_.accel_bias += dx.segment<3>(9);
    bias_.gyro_bias += dx.segment<3>(12);

    // ========== Covariance Update ==========
    // P = (I - K * H) * P * (I - K * H)' + K * R * K'  (Joseph form for stability)
    Eigen::Matrix<double, 15, 15> I_KH = Eigen::Matrix<double, 15, 15>::Identity() - K * H;
    P_ = I_KH * P_ * I_KH.transpose() + K * R_meas_ * K.transpose();

    // Ensure symmetry
    P_ = 0.5 * (P_ + P_.transpose());
}
