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

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    Frame* prev_frame = nullptr;
    cv::BFMatcher matcher(cv::NORM_HAMMING);

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

        // Draw visualization
        cv::Mat display;
        if (prev_frame != nullptr && !good_matches.empty()) {
            // Draw matches side by side
            cv::drawMatches(prev_frame->image, prev_frame->keypoints,
                           current_frame.image, current_frame.keypoints,
                           good_matches, display);
        } else {
            // Just draw keypoints on current frame
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
        if (cv::waitKey(1) == 'q') break;
    }

    delete prev_frame;
    return 0;
}
