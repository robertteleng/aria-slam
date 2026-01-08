#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
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
    void detectAsync(const cv::Mat& image, cudaStream_t stream);
    std::vector<Detection> getResults();
    
private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    void* buffers_[2];
    int inputIndex_, outputIndex_;
    size_t inputSize_, outputSize_;
    
    std::vector<float> outputData_;
    int inputH_, inputW_;
    float scaleX_, scaleY_;
    
    void preprocess(const cv::Mat& image, float* buffer);
    std::vector<Detection> postprocess();
};
