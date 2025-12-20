#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main() {
    std::cout << "Aria SLAM" << std::endl;

    // cv::VideoCapture cap(0); // Open the default camera
    cv::VideoCapture cap("../test.mp4"); // Open a video file

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    cv::Ptr<cv::ORB> orb = cv::ORB::create();


    while (true)
    {
        auto t1 = std::chrono::high_resolution_clock::now(); // Start time measurement

        cv::Mat frame; // Create a matrix to hold the frame

        cap >> frame; // Capture a frame from the webcam
        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }

        // ORB feature detection and description
        std::vector<cv::KeyPoint> keypoints; // Vector to hold keypoints
        cv::Mat descriptors; // Matrix to hold descriptors
        orb->detectAndCompute(frame, cv::noArray(), keypoints, descriptors); // Detect keypoints and compute descriptors
        cv::drawKeypoints(frame, keypoints, frame, cv::Scalar(0,255,0)); // Draw keypoints on the frame

        // Calculate and display FPS
        auto t2 = std::chrono::high_resolution_clock::now(); // End time measurement
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count(); // Calculate elapsed time in milliseconds
        double fps = 1000.0 / ms;  // Calculate frames per second
        std::string fps_text = "FPS: " + std::to_string((int)fps); // Prepare FPS text
        cv::putText(frame, fps_text, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 2); // Overlay FPS on the frame

        // Display the frame with keypoints and FPS
        cv::namedWindow("Webcam Frame", cv::WINDOW_NORMAL); // Create a window to display the frame
        cv::imshow("Webcam Frame", frame); // Display the frame in a window
        if (cv::waitKey(1) == 'q') break; // Exit loop if 'q' is pressed
    }
    
    return 0;
}