# H04: Pose Estimation

**Status:** ✅ Completed

## Objective

Estimate camera motion from feature matches using epipolar geometry.

## Requirements

- Essential matrix computation
- RANSAC outlier rejection
- Pose recovery (R, t)
- Trajectory accumulation

## Theory

### Epipolar Geometry

For two views of a 3D point:
```
x2^T * E * x1 = 0
```

Where:
- `E` = Essential matrix (encodes rotation and translation)
- `x1`, `x2` = Normalized image coordinates

### Essential Matrix Decomposition

```
E = [t]_x * R
```

Where `[t]_x` is the skew-symmetric matrix of translation.

## Implementation

### Pose Estimator

```cpp
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class PoseEstimator {
public:
    PoseEstimator(const cv::Mat& K) : K_(K) {}

    bool estimate(const std::vector<cv::Point2f>& pts1,
                  const std::vector<cv::Point2f>& pts2,
                  cv::Mat& R, cv::Mat& t,
                  std::vector<uchar>& inliers) {

        if (pts1.size() < 8) return false;

        // Find Essential matrix with RANSAC
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K_,
                                          cv::RANSAC, 0.999, 1.0, inliers);

        if (E.empty()) return false;

        // Recover pose
        int valid = cv::recoverPose(E, pts1, pts2, K_, R, t, inliers);

        return valid > 10;  // Need sufficient inliers
    }

    // Accumulate trajectory
    void updatePose(const cv::Mat& R, const cv::Mat& t) {
        // World = World * [R|t]^-1
        cv::Mat R_inv = R.t();
        cv::Mat t_inv = -R_inv * t;

        position_ += cv::Mat(rotation_ * t_inv);
        rotation_ = rotation_ * R_inv;
    }

    cv::Point3d getPosition() const {
        return cv::Point3d(position_.at<double>(0),
                           position_.at<double>(1),
                           position_.at<double>(2));
    }

private:
    cv::Mat K_;  // Camera intrinsics
    cv::Mat rotation_ = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat position_ = cv::Mat::zeros(3, 1, CV_64F);
};
```

### Camera Intrinsics

For EuRoC dataset (cam0):
```cpp
cv::Mat K = (cv::Mat_<double>(3,3) <<
    458.654, 0, 367.215,
    0, 457.296, 248.375,
    0, 0, 1);
```

## Files Created

- `include/PoseEstimator.hpp`
- `src/PoseEstimator.cpp`

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| OpenCV | >= 4.6 | Essential matrix, RANSAC |
| Eigen | >= 3.3 | Matrix operations |

## RANSAC Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Confidence | 0.999 | Probability of finding correct model |
| Threshold | 1.0 px | Maximum reprojection error |
| Min inliers | 10 | Minimum points to accept pose |

## Scale Ambiguity

Monocular SLAM has inherent scale ambiguity - we can only recover direction of translation, not magnitude. Solutions:

1. **IMU fusion** (H08) - Accelerometer provides metric scale
2. **Known object size** - Ground plane, markers
3. **Stereo camera** - Baseline provides scale

## Next Steps

→ H05: OpenCV CUDA
