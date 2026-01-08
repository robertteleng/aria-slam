#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};

class TRTInference {
public:
    TRTInference(const std::string& enginePath);
    ~TRTInference();
    
    std::vector<Detection> detect(const cv::Mat& image);
    
private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    void* buffers_[2];
    int inputIndex_, outputIndex_;
    int inputSize_, outputSize_;
};
