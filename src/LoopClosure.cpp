#include "LoopClosure.hpp"

LoopClosure::LoopClosure() {}

void LoopClosure::addKeyframe(int id, const cv::Mat& descriptors) {
    keyframeDescriptors_.push_back(descriptors.clone());
    keyframeIds_.push_back(id);
}

int LoopClosure::detectLoop(const cv::Mat& descriptors) {
    if (keyframeDescriptors_.empty()) return -1;
    
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    
    int bestMatch = -1;
    int bestScore = 0;
    
    for (size_t i = 0; i < keyframeDescriptors_.size(); i++) {
        std::vector<cv::DMatch> matches;
        matcher->match(descriptors, keyframeDescriptors_[i], matches);
        
        int goodCount = 0;
        for (const auto& m : matches) {
            if (m.distance < 50) goodCount++;
        }
        
        if (goodCount > bestScore && goodCount > 30) {
            bestScore = goodCount;
            bestMatch = keyframeIds_[i];
        }
    }
    
    return bestMatch;
}
