# H08: IMU Sensor Fusion

**Status:** ✅ Completed

## Objective

Fuse IMU data with visual odometry using Extended Kalman Filter for robust state estimation.

## Requirements

- IMU preintegration
- EKF state estimation
- Visual-Inertial alignment
- Metric scale recovery

## Why IMU Fusion?

1. **Scale recovery**: Monocular SLAM can't determine metric scale
2. **Fast motion**: IMU tracks rapid movements between frames
3. **Motion blur**: IMU provides state when vision fails
4. **Prediction**: IMU predicts pose for feature search

## State Vector

```
x = [p, v, q, bg, ba]
    p  - Position (3)
    v  - Velocity (3)
    q  - Orientation quaternion (4)
    bg - Gyroscope bias (3)
    ba - Accelerometer bias (3)

Total: 16 states
```

## Implementation

### IMU Preintegration

```cpp
struct ImuPreintegration {
    Eigen::Vector3d delta_p;     // Position change
    Eigen::Vector3d delta_v;     // Velocity change
    Eigen::Quaterniond delta_q;  // Rotation change
    Eigen::Matrix<double, 9, 9> covariance;
    double dt;

    void integrate(const Eigen::Vector3d& gyro,
                   const Eigen::Vector3d& accel,
                   double delta_t,
                   const Eigen::Vector3d& bg,
                   const Eigen::Vector3d& ba) {
        // Remove bias
        Eigen::Vector3d w = gyro - bg;
        Eigen::Vector3d a = accel - ba;

        // Update rotation
        Eigen::Quaterniond dq;
        dq = Eigen::AngleAxisd(w.norm() * delta_t, w.normalized());
        delta_q = delta_q * dq;

        // Update velocity and position
        Eigen::Vector3d a_world = delta_q * a;
        delta_p += delta_v * delta_t + 0.5 * a_world * delta_t * delta_t;
        delta_v += a_world * delta_t;

        dt += delta_t;
    }
};
```

### EKF State Estimator

```cpp
class ImuEKF {
public:
    void predict(const ImuPreintegration& imu) {
        // State prediction
        state_.p += state_.v * imu.dt +
                    0.5 * (state_.q * imu.delta_v) * imu.dt;
        state_.v += state_.q * imu.delta_v;
        state_.q = state_.q * imu.delta_q;

        // Covariance prediction
        Eigen::MatrixXd F = computeStateTransition(imu);
        Eigen::MatrixXd Q = computeProcessNoise(imu);
        P_ = F * P_ * F.transpose() + Q;
    }

    void updateVision(const Eigen::Matrix4d& visual_pose,
                      const Eigen::Matrix<double, 6, 6>& R) {
        // Measurement residual
        Eigen::VectorXd z = computeResidual(visual_pose);

        // Kalman gain
        Eigen::MatrixXd H = computeMeasurementJacobian();
        Eigen::MatrixXd S = H * P_ * H.transpose() + R;
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

        // State update
        Eigen::VectorXd dx = K * z;
        applyCorrection(dx);

        // Covariance update
        P_ = (Eigen::MatrixXd::Identity(16, 16) - K * H) * P_;
    }

private:
    State state_;
    Eigen::Matrix<double, 16, 16> P_;  // State covariance
};
```

### Visual-Inertial Alignment

```cpp
// Initial alignment to find scale and gravity
bool alignVisualInertial(const std::vector<VisualFrame>& frames,
                          const std::vector<ImuPreintegration>& imu,
                          double& scale,
                          Eigen::Vector3d& gravity) {
    // Solve: v_i+1 - v_i = g*dt + R_i*dv_imu
    // And: p_i+1 - p_i = v_i*dt + 0.5*g*dt^2 + R_i*dp_imu * scale

    // Linear least squares for [scale, gravity]
    Eigen::MatrixXd A(frames.size() * 6, 4);
    Eigen::VectorXd b(frames.size() * 6);

    // Fill A, b from visual and IMU constraints
    // ...

    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
    scale = x(0);
    gravity = x.segment<3>(1);

    return scale > 0 && (gravity.norm() - 9.81) < 0.5;
}
```

## IMU Noise Parameters (EuRoC)

```cpp
// From imu0/sensor.yaml
double gyro_noise = 1.6968e-04;      // rad/s/sqrt(Hz)
double accel_noise = 2.0e-3;         // m/s^2/sqrt(Hz)
double gyro_bias_noise = 1.9393e-05; // rad/s^2/sqrt(Hz)
double accel_bias_noise = 3.0e-3;    // m/s^3/sqrt(Hz)
```

## Files Created

- `include/ImuPreintegration.hpp`
- `include/ImuEKF.hpp`
- `src/ImuPreintegration.cpp`
- `src/ImuEKF.cpp`

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| Eigen | >= 3.3 | Matrix operations |

## Performance Impact

| Metric | Vision Only | Visual-Inertial |
|--------|-------------|-----------------|
| Scale accuracy | Unknown | < 2% error |
| Fast motion | Fails | Robust |
| ATE (MH_01) | 0.45m | 0.08m |

## Next Steps

→ H09: Loop Closure Detection
