#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include "Frame.hpp"
#include "TRTInference.hpp"

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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file> [engine_file]" << std::endl;
        return 1;
    }
    
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return 1;
    }
    
    // Create CUDA streams
    cv::cuda::Stream streamORB, streamYOLO;
    cudaStream_t yoloStream;
    cudaStreamCreate(&yoloStream);
    
    std::unique_ptr<TRTInference> yolo;
    if (argc >= 3) {
        yolo = std::make_unique<TRTInference>(argv[2]);
    }
    
    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Point2d pp(cx, cy);
    
    cv::Mat img;
    while (cap.read(img)) {
        Frame frame(img, streamORB);
        
        // Launch ORB async
        frame.uploadToGPU(streamORB);
        frame.extractFeaturesAsync(streamORB);
        
        // Launch YOLO async in parallel
        if (yolo) {
            yolo->detectAsync(img, yoloStream);
        }
        
        // Sync both streams
        streamORB.waitForCompletion();
        cudaStreamSynchronize(yoloStream);
        
        frame.downloadResults();
        std::vector<Detection> detections;
        if (yolo) {
            detections = yolo->getResults();
        }
        
        std::cout << "Features: " << frame.keypoints_.size() 
                  << " Detections: " << detections.size() << std::endl;
    }
    
    cudaStreamDestroy(yoloStream);
    return 0;
}
