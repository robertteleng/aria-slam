#include "IMU.hpp"

EKF15State::EKF15State() {
    state_.setZero();
    state_(6) = 1.0; // quaternion w component
    P_.setIdentity();
    P_ *= 0.01;
}

void EKF15State::predict(const IMUData& data, double dt) {
    // Get current state
    Eigen::Vector3d pos = getPosition();
    Eigen::Vector3d vel = getVelocity();
    Eigen::Quaterniond q = getOrientation();
    Eigen::Vector3d ba = state_.segment<3>(10);
    Eigen::Vector3d bg = state_.segment<3>(13);
    
    // Remove bias
    Eigen::Vector3d accel = data.accel - ba;
    Eigen::Vector3d gyro = data.gyro - bg;
    
    // Gravity
    Eigen::Vector3d g(0, 0, -9.81);
    
    // Update position and velocity
    Eigen::Vector3d accel_world = q * accel + g;
    pos += vel * dt + 0.5 * accel_world * dt * dt;
    vel += accel_world * dt;
    
    // Update quaternion
    Eigen::Quaterniond dq;
    Eigen::Vector3d theta = gyro * dt;
    double angle = theta.norm();
    if (angle > 1e-8) {
        dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, theta / angle));
    } else {
        dq = Eigen::Quaterniond::Identity();
    }
    q = q * dq;
    q.normalize();
    
    // Store back
    state_.segment<3>(0) = pos;
    state_.segment<3>(3) = vel;
    state_.segment<4>(6) = Eigen::Vector4d(q.x(), q.y(), q.z(), q.w());
}

void EKF15State::updateVO(const Eigen::Vector3d& position, const Eigen::Matrix3d& R) {
    // Simple position update for now
    Eigen::Vector3d innovation = position - getPosition();
    state_.segment<3>(0) += 0.5 * innovation;
}

Eigen::Vector3d EKF15State::getPosition() const {
    return state_.segment<3>(0);
}

Eigen::Vector3d EKF15State::getVelocity() const {
    return state_.segment<3>(3);
}

Eigen::Quaterniond EKF15State::getOrientation() const {
    return Eigen::Quaterniond(state_(9), state_(6), state_(7), state_(8));
}
