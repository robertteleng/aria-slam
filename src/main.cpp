/**
 * @file main.cpp
 * @brief Aria SLAM - Visual Odometry with GPU acceleration
 *
 * @details Pipeline (H11: Parallel CUDA Streams):
 * 1. Capture frame from video
 * 2. Parallel execution on GPU:
 *    - Stream 1: ORB feature detection
 *    - Stream 2: YOLO object detection
 * 3. Synchronize streams
 * 4. Match features with previous frame
 * 5. Estimate pose from Essential Matrix
 * 6. Accumulate trajectory and visualize
 *
 * @note Requires CUDA-enabled GPU and OpenCV built with CUDA support
 */

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <cuda_runtime_api.h>
#include <chrono>
#include "Frame.hpp"
#include "TRTInference.hpp"

// COCO class names for visualization
const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

int main() {
    std::cout << "Aria SLAM (CUDA + TensorRT)" << std::endl;

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

    // YOLO object detection (TensorRT)
    std::unique_ptr<TRTInference> yolo;
    try {
        yolo = std::make_unique<TRTInference>("../models/yolov12s.engine");
    } catch (const std::exception& e) {
        std::cerr << "Warning: YOLO disabled - " << e.what() << std::endl;
    }

    // H11: Create CUDA streams for parallel execution
    cudaStream_t stream_orb, stream_yolo;
    cudaStreamCreate(&stream_orb);
    cudaStreamCreate(&stream_yolo);
    std::cout << "H11: CUDA streams created for parallel ORB + YOLO" << std::endl;

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

        // H11: Launch ORB and YOLO in parallel on separate CUDA streams
        // Stream 1: ORB feature extraction (async)
        Frame current_frame(frame, orb, stream_orb);

        // Stream 2: YOLO object detection (async)
        if (yolo) {
            yolo->detectAsync(frame, stream_yolo);
        }

        // H11: Synchronize both streams before using results
        cudaStreamSynchronize(stream_orb);
        cudaStreamSynchronize(stream_yolo);

        // Download ORB results from GPU to CPU
        current_frame.downloadResults();

        // Get YOLO detections (data already on CPU after sync)
        std::vector<Detection> detections;
        if (yolo) {
            detections = yolo->getDetections(0.5f, 0.45f);
        }

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

        // Draw YOLO detections
        for (const auto& det : detections) {
            cv::rectangle(display, det.box, cv::Scalar(0, 0, 255), 2);
            std::string label = COCO_CLASSES[det.class_id] + " " +
                               std::to_string((int)(det.confidence * 100)) + "%";
            cv::putText(display, label, cv::Point(det.box.x, det.box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
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
        cv::putText(display, "Objects: " + std::to_string(detections.size()),
                    cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        cv::namedWindow("Aria SLAM", cv::WINDOW_NORMAL);
        cv::imshow("Aria SLAM", display);

        // Process window events
        char key = cv::waitKey(1);
        if (key == 'q') break;
    }

    // H11: Cleanup CUDA streams
    cudaStreamDestroy(stream_orb);
    cudaStreamDestroy(stream_yolo);

    return 0;
}
