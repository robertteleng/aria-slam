/**
 * @file euroc_eval.cpp
 * @brief Evaluate Aria SLAM on EuRoC MAV dataset
 *
 * Runs the full SLAM pipeline on EuRoC sequences and compares
 * estimated trajectory against ground truth.
 *
 * Usage: ./euroc_eval <dataset_path>
 * Example: ./euroc_eval ../datasets/MH_01_easy
 */

#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include "EuRoCReader.hpp"
#include "Frame.hpp"
#include "IMU.hpp"
#include "LoopClosure.hpp"
#include "Mapper.hpp"

// Compute Absolute Trajectory Error (ATE)
double computeATE(const std::vector<Eigen::Vector3d>& estimated,
                  const std::vector<Eigen::Vector3d>& ground_truth) {
    if (estimated.size() != ground_truth.size() || estimated.empty()) {
        return -1;
    }

    double sum_sq_error = 0;
    for (size_t i = 0; i < estimated.size(); i++) {
        sum_sq_error += (estimated[i] - ground_truth[i]).squaredNorm();
    }

    return std::sqrt(sum_sq_error / estimated.size());
}

// Compute Relative Pose Error (RPE) for translation
double computeRPE(const std::vector<Eigen::Vector3d>& estimated,
                  const std::vector<Eigen::Vector3d>& ground_truth,
                  int delta = 10) {
    if (estimated.size() != ground_truth.size() || estimated.size() <= delta) {
        return -1;
    }

    double sum_sq_error = 0;
    int count = 0;

    for (size_t i = delta; i < estimated.size(); i++) {
        Eigen::Vector3d est_delta = estimated[i] - estimated[i - delta];
        Eigen::Vector3d gt_delta = ground_truth[i] - ground_truth[i - delta];
        sum_sq_error += (est_delta - gt_delta).squaredNorm();
        count++;
    }

    return std::sqrt(sum_sq_error / count);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <euroc_dataset_path>" << std::endl;
        std::cout << "Example: " << argv[0] << " ../datasets/MH_01_easy" << std::endl;
        return -1;
    }

    std::string dataset_path = argv[1];
    std::cout << "=== Aria SLAM - EuRoC Evaluation ===" << std::endl;

    // Load dataset
    EuRoCReader reader(dataset_path);
    if (!reader.load()) {
        std::cerr << "Failed to load dataset" << std::endl;
        return -1;
    }

    // Verify CUDA
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_devices == 0) {
        std::cerr << "Warning: No CUDA devices, using CPU" << std::endl;
    }

    // Initialize components
    cv::Mat K = reader.getCameraMatrix();
    cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(2000);
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
        cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

    SensorFusion fusion;
    // min_frames_between=200, min_score=0.4, min_matches=50
    LoopClosureDetector loop_detector(200, 0.4, 50);
    PoseGraphOptimizer optimizer;
    Mapper mapper(K);

    // State
    std::unique_ptr<Frame> prev_frame;
    Eigen::Matrix4d current_pose = Eigen::Matrix4d::Identity();

    // Results storage
    std::vector<Eigen::Vector3d> estimated_trajectory;
    std::vector<Eigen::Vector3d> ground_truth_trajectory;
    std::vector<double> timestamps;

    // Visualization
    cv::Mat trajectory_img = cv::Mat::zeros(800, 800, CV_8UC3);
    double scale = 50.0;  // meters to pixels
    int cx = 400, cy = 400;

    // Processing stats
    int frame_count = 0;
    int loop_count = 0;
    double total_time = 0;

    std::cout << "\nProcessing " << reader.numImages() << " frames..." << std::endl;

    while (reader.hasNext()) {
        cv::Mat image;
        std::vector<IMUMeasurement> imu_data;
        double timestamp;

        if (!reader.getNext(image, imu_data, timestamp)) {
            continue;
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // Process IMU data
        for (const auto& imu : imu_data) {
            fusion.addIMU(imu);
        }

        // Extract features
        Frame current_frame(image, orb);

        // Feature matching
        std::vector<cv::DMatch> good_matches;
        if (prev_frame && !prev_frame->gpu_descriptors.empty() &&
            !current_frame.gpu_descriptors.empty()) {

            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher->knnMatch(prev_frame->gpu_descriptors,
                             current_frame.gpu_descriptors, knn_matches, 2);

            for (const auto& knn : knn_matches) {
                if (knn.size() >= 2 && knn[0].distance < 0.75 * knn[1].distance) {
                    good_matches.push_back(knn[0]);
                }
            }
        }

        // Pose estimation
        if (good_matches.size() >= 8 && prev_frame) {
            std::vector<cv::Point2f> pts1, pts2;
            for (const auto& m : good_matches) {
                pts1.push_back(prev_frame->keypoints[m.queryIdx].pt);
                pts2.push_back(current_frame.keypoints[m.trainIdx].pt);
            }

            cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0);
            if (!E.empty()) {
                cv::Mat R_cv, t_cv;
                int inliers = cv::recoverPose(E, pts1, pts2, K, R_cv, t_cv);

                if (inliers > 10) {
                    // Convert to Eigen
                    Eigen::Matrix3d R;
                    Eigen::Vector3d t;
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            R(i, j) = R_cv.at<double>(i, j);
                        }
                        t(i) = t_cv.at<double>(i);
                    }

                    // Update pose
                    Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
                    delta.block<3, 3>(0, 0) = R;
                    delta.block<3, 1>(0, 3) = t;
                    current_pose = current_pose * delta;

                    // Visual update to EKF
                    fusion.addVisualPose(timestamp, R, t);

                    // Add to pose graph
                    optimizer.setInitialPose(frame_count, current_pose);
                    if (frame_count > 0) {
                        optimizer.addOdometryEdge(frame_count - 1, frame_count, delta);
                    }

                    // Triangulate new map points
                    if (frame_count > 0) {
                        Eigen::Matrix4d prev_pose = optimizer.getOptimizedPose(frame_count - 1);
                        mapper.triangulate(prev_frame->keypoints, current_frame.keypoints,
                                          good_matches, prev_pose, current_pose, prev_frame->image);
                    }

                    // Loop closure detection
                    KeyFrame kf;
                    kf.id = frame_count;
                    kf.timestamp = timestamp;
                    kf.position = current_pose.block<3, 1>(0, 3);
                    kf.orientation = Eigen::Quaterniond(current_pose.block<3, 3>(0, 0));
                    kf.keypoints = current_frame.keypoints;
                    current_frame.descriptors.copyTo(kf.descriptors);

                    LoopCandidate candidate;
                    if (loop_detector.detect(kf, candidate)) {
                        optimizer.addLoopEdge(candidate.query_id, candidate.match_id,
                                             candidate.relative_pose);
                        optimizer.optimize(10);
                        current_pose = optimizer.getOptimizedPose(frame_count);
                        loop_count++;
                    }

                    loop_detector.addKeyFrame(kf);
                }
            }
        }

        // Get ground truth for comparison
        EuRoCGroundTruth gt;
        if (reader.getGroundTruth(timestamp, gt)) {
            estimated_trajectory.push_back(current_pose.block<3, 1>(0, 3));
            ground_truth_trajectory.push_back(gt.position);
            timestamps.push_back(timestamp);

            // Draw trajectories
            int ex = cx + (int)(current_pose(0, 3) * scale);
            int ey = cy - (int)(current_pose(1, 3) * scale);
            int gx = cx + (int)(gt.position.x() * scale);
            int gy = cy - (int)(gt.position.y() * scale);

            cv::circle(trajectory_img, cv::Point(ex, ey), 1, cv::Scalar(0, 255, 0), -1);  // Green = estimated
            cv::circle(trajectory_img, cv::Point(gx, gy), 1, cv::Scalar(0, 0, 255), -1);  // Red = ground truth
        }

        prev_frame = std::make_unique<Frame>(current_frame);
        frame_count++;

        auto t_end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // Progress update
        if (frame_count % 100 == 0) {
            double avg_fps = 1000.0 * frame_count / total_time;
            std::cout << "Frame " << frame_count << "/" << reader.numImages()
                      << " | FPS: " << std::fixed << std::setprecision(1) << avg_fps
                      << " | Map: " << mapper.size() << " points"
                      << " | Loops: " << loop_count << std::endl;
        }
    }

    // Final optimization
    std::cout << "\nFinal pose graph optimization..." << std::endl;
    optimizer.optimize(50);

    // Update trajectory with optimized poses
    for (size_t i = 0; i < estimated_trajectory.size(); i++) {
        Eigen::Matrix4d opt_pose = optimizer.getOptimizedPose(i);
        estimated_trajectory[i] = opt_pose.block<3, 1>(0, 3);
    }

    // Filter outliers from map
    mapper.filterOutliers();

    // Compute metrics
    double ate = computeATE(estimated_trajectory, ground_truth_trajectory);
    double rpe = computeRPE(estimated_trajectory, ground_truth_trajectory);

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Frames processed: " << frame_count << std::endl;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(1)
              << (1000.0 * frame_count / total_time) << std::endl;
    std::cout << "Loop closures: " << loop_count << std::endl;
    std::cout << "Map points: " << mapper.size() << std::endl;
    std::cout << "\nTrajectory Error:" << std::endl;
    std::cout << "  ATE (RMSE): " << std::setprecision(4) << ate << " m" << std::endl;
    std::cout << "  RPE (RMSE): " << std::setprecision(4) << rpe << " m" << std::endl;

    // Export results
    std::string output_dir = dataset_path + "/results/";
    system(("mkdir -p " + output_dir).c_str());

    // Save trajectory
    std::ofstream traj_file(output_dir + "estimated_trajectory.txt");
    for (size_t i = 0; i < timestamps.size(); i++) {
        traj_file << std::fixed << std::setprecision(9) << timestamps[i] << " "
                  << estimated_trajectory[i].x() << " "
                  << estimated_trajectory[i].y() << " "
                  << estimated_trajectory[i].z() << std::endl;
    }
    traj_file.close();

    // Save map
    mapper.exportPLY(output_dir + "map.ply");

    // Save trajectory image
    cv::putText(trajectory_img, "ATE: " + std::to_string(ate).substr(0, 6) + " m",
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(trajectory_img, "Green=Estimated, Red=GroundTruth",
                cv::Point(10, 780), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    cv::imwrite(output_dir + "trajectory.png", trajectory_img);

    std::cout << "\nResults saved to: " << output_dir << std::endl;

    return 0;
}
