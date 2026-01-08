#include <opencv2/opencv.hpp>
#include <iostream>
#include "Frame.hpp"

const float LOWE_RATIO = 0.75f;

std::vector<cv::DMatch> filterMatches(const std::vector<std::vector<cv::DMatch>>& knnMatches) {
    std::vector<cv::DMatch> goodMatches;
    for (const auto& knn : knnMatches) {
        if (knn.size() >= 2 && knn[0].distance < LOWE_RATIO * knn[1].distance) {
            goodMatches.push_back(knn[0]);
        }
    }
    return goodMatches;
}

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
            
            auto goodMatches = filterMatches(knnMatches);
            std::cout << "Good matches: " << goodMatches.size() << " / " << knnMatches.size() << std::endl;
            
            delete prevFrame;
        }
        
        prevFrame = currFrame;
    }
    
    delete prevFrame;
    return 0;
}
