#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "Frame.hpp"

int main() {
    std::cout << "Aria SLAM" << std::endl;

    cv::VideoCapture cap("../test.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    // ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    // Previous frame pointer
    Frame* prev_frame = nullptr;
    // BFMatcher for ORB
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // Camera matrix (ajusta cx, cy al tamaño de tu video)
    double fx = 700, fy = 700;
    double cx = 640 / 2.0;
    double cy = 360 / 2.0;
    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // Posición y rotación global
    cv::Mat position = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);

    // Imagen para trayectoria
    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);

    while (true) {
        auto t1 = std::chrono::high_resolution_clock::now();

        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Create Frame (detects keypoints internally)
        Frame current_frame(frame, orb);

        // Match with previous frame
        std::vector<cv::DMatch> good_matches;
        if (prev_frame != nullptr &&
            !prev_frame->descriptors.empty() &&
            !current_frame.descriptors.empty()) {

            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher.knnMatch(prev_frame->descriptors, current_frame.descriptors, knn_matches, 2);

            for (auto& knn : knn_matches) {
                if (knn.size() >= 2 && knn[0].distance < 0.75 * knn[1].distance) {
                    good_matches.push_back(knn[0]);
                }
            }
        }

        // H4: Pose Estimation
        if (good_matches.size() >= 8) {
            // Extraer coordenadas de los matches
            std::vector<cv::Point2f> pts1, pts2;
            for (auto& m : good_matches) {
                pts1.push_back(prev_frame->keypoints[m.queryIdx].pt);
                pts2.push_back(current_frame.keypoints[m.trainIdx].pt);
            }
            
            // Essential Matrix
            cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC);
            
            // Recover Pose
            cv::Mat R, t;
            cv::recoverPose(E, pts1, pts2, K, R, t);
            
            // Acumular posición
            position = position + rotation * t;
            rotation = R * rotation;
            
            // Dibujar trayectoria
            int x = (int)(position.at<double>(0) * 100) + 300;
            int y = (int)(position.at<double>(2) * 100) + 300;
            cv::circle(trajectory, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
        }

        cv::namedWindow("Trajectory", cv::WINDOW_NORMAL);
        cv::imshow("Trajectory", trajectory);

        // H3: Visualización matches
        // IMPORTANT: Always use current_frame.image (where keypoints were detected)
        // Using 'frame' instead causes black screen - keypoint coords don't match
        cv::Mat display;
        if (prev_frame != nullptr && !good_matches.empty()) {
            cv::drawMatches(prev_frame->image, prev_frame->keypoints,
                           current_frame.image, current_frame.keypoints,
                           good_matches, display);
        } else {
            cv::drawKeypoints(current_frame.image, current_frame.keypoints,
                             display, cv::Scalar(0, 255, 0));
        }

        // Update previous frame
        if (prev_frame != nullptr) delete prev_frame;
        prev_frame = new Frame(current_frame);

        // FPS and stats
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        cv::putText(display, "FPS: " + std::to_string((int)(1000.0 / ms)),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, "Matches: " + std::to_string(good_matches.size()),
                    cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // Display
        cv::namedWindow("Aria SLAM", cv::WINDOW_NORMAL);
        cv::imshow("Aria SLAM", display);

        // waitKey refresca TODAS las ventanas
        char key = cv::waitKey(30);  // 30ms = ~33 FPS max
        if (key == 'q') break;
    }

    delete prev_frame;
    return 0;
}
