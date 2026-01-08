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
    
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    
    cv::Mat img;
    Frame* prevFrame = nullptr;
    
    while (cap.read(img)) {
        Frame* currFrame = new Frame(img);
        currFrame->extractFeatures();
        
        if (prevFrame != nullptr && !prevFrame->descriptors_.empty() && !currFrame->descriptors_.empty()) {
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher->knnMatch(prevFrame->descriptors_, currFrame->descriptors_, knnMatches, 2);
            std::cout << "KNN Matches: " << knnMatches.size() << std::endl;
            delete prevFrame;
        }
        
        prevFrame = currFrame;
    }
    
    delete prevFrame;
    return 0;
}
