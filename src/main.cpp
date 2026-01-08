#include <opencv2/opencv.hpp>
#include <iostream>
#include "Frame.hpp"

const float LOWE_RATIO = 0.75f;
const int MIN_MATCHES = 20;

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

void extractMatchedPoints(const std::vector<cv::DMatch>& matches,
                          const std::vector<cv::KeyPoint>& kp1,
                          const std::vector<cv::KeyPoint>& kp2,
                          std::vector<cv::Point2f>& pts1,
                          std::vector<cv::Point2f>& pts2) {
    for (const auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }
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
    cv::Point2d pp(cx, cy);
    
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    
    cv::Mat img;
    Frame* prevFrame = nullptr;
    int frameCount = 0;
    
    cv::Mat R_total = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_total = cv::Mat::zeros(3, 1, CV_64F);
    
    while (cap.read(img)) {
        Frame* currFrame = new Frame(img);
        currFrame->extractFeatures();
        
        if (prevFrame != nullptr && !prevFrame->descriptors_.empty() && !currFrame->descriptors_.empty()) {
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher->knnMatch(prevFrame->descriptors_, currFrame->descriptors_, knnMatches, 2);
            
            auto goodMatches = filterMatches(knnMatches);
            
            if (goodMatches.size() >= MIN_MATCHES) {
                std::vector<cv::Point2f> pts1, pts2;
                extractMatchedPoints(goodMatches, prevFrame->keypoints_, currFrame->keypoints_, pts1, pts2);
                
                cv::Mat E, mask;
                E = cv::findEssentialMat(pts1, pts2, fx, pp, cv::RANSAC, 0.999, 1.0, mask);
                
                cv::Mat R, t;
                int inliers = cv::recoverPose(E, pts1, pts2, R, t, fx, pp, mask);
                
                t_total = t_total + R_total * t;
                R_total = R * R_total;
                
                std::cout << "[" << frameCount << "] Position: " 
                          << t_total.at<double>(0) << ", "
                          << t_total.at<double>(1) << ", "
                          << t_total.at<double>(2) << std::endl;
            }
            
            delete prevFrame;
        }
        
        prevFrame = currFrame;
        frameCount++;
    }
    
    delete prevFrame;
    return 0;
}
