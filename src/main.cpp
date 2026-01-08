#include <opencv2/opencv.hpp>
#include <iostream>
#include "Frame.hpp"

const float LOWE_RATIO = 0.75f;
const int MIN_MATCHES = 20;

// Camera intrinsics (placeholder)
const double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;

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
    
    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    
    cv::Mat img;
    Frame* prevFrame = nullptr;
    int frameCount = 0;
    
    while (cap.read(img)) {
        Frame* currFrame = new Frame(img);
        currFrame->extractFeatures();
        
        if (prevFrame != nullptr && !prevFrame->descriptors_.empty() && !currFrame->descriptors_.empty()) {
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher->knnMatch(prevFrame->descriptors_, currFrame->descriptors_, knnMatches, 2);
            
            auto goodMatches = filterMatches(knnMatches);
            
            if (goodMatches.size() >= MIN_MATCHES) {
                std::cout << "[" << frameCount << "] Good matches: " << goodMatches.size() << std::endl;
            }
            
            delete prevFrame;
        }
        
        prevFrame = currFrame;
        frameCount++;
    }
    
    delete prevFrame;
    return 0;
}
