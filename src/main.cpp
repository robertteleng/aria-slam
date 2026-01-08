#include <opencv2/opencv.hpp>
#include <iostream>
#include "Frame.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file>" << std::endl;
        return 1;
    }
    
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return 1;
    }
    
    cv::Mat img;
    while (cap.read(img)) {
        Frame frame(img);
        frame.extractFeatures();
        
        std::cout << "Extracted " << frame.keypoints_.size() << " keypoints" << std::endl;
    }
    
    return 0;
}
