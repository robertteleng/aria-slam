#include <iostream>
#include "IMU.hpp"
#include "SyntheticIMU.hpp"

int main() {
    std::cout << "=== IMU Sensor Fusion Test ===" << std::endl;

    SyntheticIMU synth(200.0);  // 200Hz IMU
    SensorFusion fusion;

    double duration = 10.0;  // 10 seconds
    double visual_rate = 20.0;  // 20Hz camera
    double imu_rate = 200.0;

    double t = 0;
    int visual_count = 0;

    while (t < duration) {
        // Generate and add IMU data
        IMUMeasurement imu = synth.generateCircular(t);
        fusion.addIMU(imu);

        // Simulate visual updates at lower rate
        if ((int)(t * visual_rate) > visual_count) {
            visual_count = (int)(t * visual_rate);

            Eigen::Matrix3d R = synth.getCircularRotation(t);
            Eigen::Vector3d pos = synth.getCircularPosition(t);

            fusion.addVisualPose(t, R, pos);

            // Compare with ground truth
            Eigen::Vector3d fused_pos = fusion.getPosition();
            Eigen::Vector3d error = fused_pos - pos;

            std::cout << "t=" << t << "s | "
                      << "GT: (" << pos.x() << ", " << pos.y() << ") | "
                      << "Fused: (" << fused_pos.x() << ", " << fused_pos.y() << ") | "
                      << "Error: " << error.norm() << "m" << std::endl;
        }

        t += 1.0 / imu_rate;
    }

    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
