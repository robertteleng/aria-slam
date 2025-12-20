#pragma once
#include "IMU.hpp"
#include <random>

class SyntheticIMU {
public:
    SyntheticIMU(double frequency = 200.0)
        : frequency_(frequency), dt_(1.0 / frequency),
          gen_(42), noise_accel_(0, 0.01), noise_gyro_(0, 0.001) {}

    // Generate IMU data for circular motion
    IMUMeasurement generateCircular(double t) {
        IMUMeasurement imu;
        imu.timestamp = t;

        // Circular motion parameters
        double radius = 2.0;  // meters
        double omega = 0.5;   // rad/s (angular velocity)

        // Centripetal acceleration + gravity
        double ax = -radius * omega * omega * cos(omega * t);
        double ay = -radius * omega * omega * sin(omega * t);
        double az = 9.81;  // compensating gravity

        // Add noise
        imu.accel = Eigen::Vector3d(
            ax + noise_accel_(gen_),
            ay + noise_accel_(gen_),
            az + noise_accel_(gen_)
        );

        // Constant rotation around Z
        imu.gyro = Eigen::Vector3d(
            noise_gyro_(gen_),
            noise_gyro_(gen_),
            omega + noise_gyro_(gen_)
        );

        return imu;
    }

    // Generate IMU data for linear motion with turns
    IMUMeasurement generateLinear(double t) {
        IMUMeasurement imu;
        imu.timestamp = t;

        // Forward acceleration for first 2 seconds, then coast
        double ax = (t < 2.0) ? 0.5 : 0.0;

        // Turn at t=5s
        double gz = (t > 5.0 && t < 7.0) ? 0.3 : 0.0;

        imu.accel = Eigen::Vector3d(
            ax + noise_accel_(gen_),
            noise_accel_(gen_),
            9.81 + noise_accel_(gen_)
        );

        imu.gyro = Eigen::Vector3d(
            noise_gyro_(gen_),
            noise_gyro_(gen_),
            gz + noise_gyro_(gen_)
        );

        return imu;
    }

    // Get ground truth position for circular motion
    Eigen::Vector3d getCircularPosition(double t) {
        double radius = 2.0;
        double omega = 0.5;
        return Eigen::Vector3d(
            radius * cos(omega * t),
            radius * sin(omega * t),
            0.0
        );
    }

    // Get ground truth rotation for circular motion
    Eigen::Matrix3d getCircularRotation(double t) {
        double omega = 0.5;
        double angle = omega * t;
        Eigen::Matrix3d R;
        R << cos(angle), -sin(angle), 0,
             sin(angle),  cos(angle), 0,
             0,           0,          1;
        return R;
    }

    double getDt() const { return dt_; }

private:
    double frequency_;
    double dt_;
    std::mt19937 gen_;
    std::normal_distribution<double> noise_accel_;
    std::normal_distribution<double> noise_gyro_;
};
