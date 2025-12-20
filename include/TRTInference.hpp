#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

class TRTInference {
public:
    TRTInference(const std::string& engine_path);
    ~TRTInference();

    std::vector<Detection> detect(const cv::Mat& image, float conf_thresh = 0.5f, float nms_thresh = 0.45f);

private:
    void preprocess(const cv::Mat& image, float* gpu_input);
    std::vector<Detection> postprocess(float* output, int num_detections,
                                        float conf_thresh, float nms_thresh,
                                        float scale_x, float scale_y);

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    void* buffers_[2];  // input + output GPU buffers
    int input_idx_, output_idx_;
    int input_h_, input_w_;
    int output_size_;

    cudaStream_t stream_;
};
