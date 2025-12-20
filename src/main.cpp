#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "Aria SLAM" << std::endl;

    // cv::VideoCapture cap(0); // Open the default camera

    cv::VideoCapture cap("../test.mp4"); // Open a video file

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    while (true)
    {
        cv::Mat frame; // Create a matrix to hold the frame

        cap >> frame; // Capture a frame from the webcam
        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }
        cv::namedWindow("Webcam Frame", cv::WINDOW_NORMAL); // Create a window to display the frame
        cv::imshow("Webcam Frame", frame); // Display the frame in a window
        if (cv::waitKey(1) == 'q') break; // Exit loop if 'q' is pressed
    }
    
    return 0;
}