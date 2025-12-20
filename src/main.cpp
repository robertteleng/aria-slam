/**
 * @file main.cpp
 * @brief Aria SLAM - Visual Odometry with GPU acceleration
 *
 * @details Pipeline:
 * 1. Capture frame from video
 * 2. Detect ORB features (GPU)
 * 3. Match features with previous frame (CPU)
 * 4. Estimate pose from Essential Matrix
 * 5. Accumulate trajectory and visualize
 *
 * @note Requires CUDA-enabled GPU and OpenCV built with CUDA support
 */

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <chrono>
#include "Frame.hpp"

int main() {
    std::cout << "Aria SLAM (CUDA)" << std::endl;

    // Verify CUDA is available
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_devices == 0) {
        std::cerr << "Error: No CUDA devices found!" << std::endl;
        return -1;
    }
    std::cout << "CUDA devices: " << cuda_devices << std::endl;

    cv::VideoCapture cap("../test.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }

    // Feature detection (GPU accelerated)
    cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create();

    // Feature matching (GPU accelerated)
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

    // Frame history for temporal matching
    std::unique_ptr<Frame> prev_frame;

    // Intrinsic camera matrix (approximate values for test video)
    double fx = 700, fy = 700;
    double cx = 640 / 2.0;
    double cy = 360 / 2.0;
    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // Camera pose (accumulated from frame-to-frame motion)
    cv::Mat position = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);

    // Trajectory visualization canvas
    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);

    while (true) {
        auto t1 = std::chrono::high_resolution_clock::now();

        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Extract ORB features on GPU
        Frame current_frame(frame, orb);

        // Match current frame with previous frame using Lowe's ratio test (GPU)
        std::vector<cv::DMatch> good_matches;
        if (prev_frame &&
            !prev_frame->gpu_descriptors.empty() &&
            !current_frame.gpu_descriptors.empty()) {

            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher->knnMatch(prev_frame->gpu_descriptors, current_frame.gpu_descriptors, knn_matches, 2);

            for (auto& knn : knn_matches) {
                if (knn.size() >= 2 && knn[0].distance < 0.75 * knn[1].distance) {
                    good_matches.push_back(knn[0]);
                }
            }
        }

        // Pose estimation from matched features (requires >= 8 points for Essential Matrix)
        if (good_matches.size() >= 8) {
            std::vector<cv::Point2f> pts1, pts2;
            for (auto& m : good_matches) {
                pts1.push_back(prev_frame->keypoints[m.queryIdx].pt);
                pts2.push_back(current_frame.keypoints[m.trainIdx].pt);
            }

            // Compute Essential Matrix with RANSAC for outlier rejection
            cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC);

            // Decompose E into rotation and translation
            cv::Mat R, t;
            cv::recoverPose(E, pts1, pts2, K, R, t);

            // Accumulate pose in world coordinates
            position = position + rotation * t;
            rotation = R * rotation;

            // Draw position on trajectory map (scaled and centered)
            int x = (int)(position.at<double>(0) * 100) + 300;
            int y = (int)(position.at<double>(2) * 100) + 300;
            cv::circle(trajectory, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
        }

        cv::namedWindow("Trajectory", cv::WINDOW_NORMAL);
        cv::imshow("Trajectory", trajectory);

        // Visualization: draw matches or keypoints
        cv::Mat display;
        if (prev_frame && !good_matches.empty()) {
            cv::drawMatches(prev_frame->image, prev_frame->keypoints,
                           current_frame.image, current_frame.keypoints,
                           good_matches, display);
        } else {
            cv::drawKeypoints(current_frame.image, current_frame.keypoints,
                             display, cv::Scalar(0, 255, 0));
        }

        // Store current frame for next iteration
        prev_frame = std::make_unique<Frame>(current_frame);

        // Calculate and display FPS
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        cv::putText(display, "FPS: " + std::to_string((int)(1000.0 / ms)),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, "Matches: " + std::to_string(good_matches.size()),
                    cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::namedWindow("Aria SLAM", cv::WINDOW_NORMAL);
        cv::imshow("Aria SLAM", display);

        // Process window events
        char key = cv::waitKey(1);
        if (key == 'q') break;
    }

    return 0;
}
