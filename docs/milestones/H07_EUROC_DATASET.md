# H07: EuRoC Dataset Integration

**Status:** ✅ Completed

## Objective

Integrate EuRoC MAV dataset for benchmarking and development with ground truth.

## About EuRoC

The EuRoC MAV Dataset provides:
- Stereo camera images (20 Hz)
- IMU measurements (200 Hz)
- Ground truth poses (Vicon/Leica)
- Various difficulty levels

## Dataset Structure

```
MH_01_easy/
├── mav0/
│   ├── cam0/           # Left camera
│   │   ├── data/       # PNG images
│   │   └── sensor.yaml # Camera intrinsics
│   ├── cam1/           # Right camera
│   ├── imu0/           # IMU data
│   │   ├── data.csv    # Measurements
│   │   └── sensor.yaml # IMU calibration
│   └── state_groundtruth_estimate0/
│       └── data.csv    # Ground truth poses
```

## Implementation

### Dataset Loader

```cpp
class EurocLoader {
public:
    EurocLoader(const std::string& dataset_path) {
        loadCameraData(dataset_path + "/mav0/cam0");
        loadImuData(dataset_path + "/mav0/imu0");
        loadGroundTruth(dataset_path + "/mav0/state_groundtruth_estimate0");
    }

    // Get synchronized frame + IMU + GT
    bool getNext(cv::Mat& image,
                 std::vector<ImuMeasurement>& imu,
                 Eigen::Matrix4d& gt_pose);

private:
    void loadCameraData(const std::string& path) {
        // Parse data.csv for timestamps
        // timestamps_ns, filename
        std::ifstream csv(path + "/data.csv");
        // ...
    }

    std::vector<std::pair<uint64_t, std::string>> images_;
    std::vector<ImuMeasurement> imu_data_;
    std::vector<std::pair<uint64_t, Eigen::Matrix4d>> ground_truth_;
};
```

### Camera Intrinsics (cam0)

```cpp
// From sensor.yaml
cv::Mat K = (cv::Mat_<double>(3,3) <<
    458.654, 0, 367.215,
    0, 457.296, 248.375,
    0, 0, 1);

// Distortion (radtan model)
cv::Mat D = (cv::Mat_<double>(1,4) <<
    -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);
```

### IMU Data Format

```cpp
struct ImuMeasurement {
    uint64_t timestamp_ns;
    Eigen::Vector3d gyro;   // rad/s
    Eigen::Vector3d accel;  // m/s^2
};

// IMU rate: 200 Hz
// Gyro noise: 1.6968e-04 rad/s/sqrt(Hz)
// Accel noise: 2.0e-3 m/s^2/sqrt(Hz)
```

### Ground Truth Format

```csv
#timestamp, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z, ...
1403636579763555584,-0.023,-0.021,0.076,0.536,-0.153,-0.827,-0.091,...
```

## Evaluation Tool

```cpp
// euroc_eval.cpp
int main(int argc, char** argv) {
    EurocLoader loader(argv[1]);
    SlamSystem slam;

    std::vector<Eigen::Vector3d> estimated, ground_truth;

    cv::Mat image;
    std::vector<ImuMeasurement> imu;
    Eigen::Matrix4d gt_pose;

    while (loader.getNext(image, imu, gt_pose)) {
        // Run SLAM
        Eigen::Matrix4d est_pose = slam.process(image, imu);

        estimated.push_back(est_pose.block<3,1>(0,3));
        ground_truth.push_back(gt_pose.block<3,1>(0,3));
    }

    // Compute ATE (Absolute Trajectory Error)
    double ate = computeATE(estimated, ground_truth);
    std::cout << "ATE: " << ate << " m" << std::endl;
}
```

## Sequences

| Sequence | Difficulty | Duration | Description |
|----------|------------|----------|-------------|
| MH_01_easy | Easy | 182s | Machine Hall slow |
| MH_02_easy | Easy | 150s | Machine Hall slow |
| MH_03_medium | Medium | 132s | Machine Hall fast |
| MH_04_difficult | Hard | 99s | Machine Hall fast |
| MH_05_difficult | Hard | 111s | Machine Hall fast |
| V1_01_easy | Easy | 144s | Vicon Room slow |
| V1_02_medium | Medium | 84s | Vicon Room fast |
| V2_01_easy | Easy | 112s | Vicon Room slow |

## Download

```bash
# MH_01_easy (~1.3GB)
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip

# Extract
unzip MH_01_easy.zip -d ~/datasets/euroc/
```

## Next Steps

→ H08: Sensor Fusion (IMU)
