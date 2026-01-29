#pragma once
#include "interfaces/IObjectDetector.hpp"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace aria::adapters::gpu {

/// TensorRT-accelerated YOLO object detector
/// Wraps TensorRT inference and converts to domain types
class YoloTrtDetector : public interfaces::IObjectDetector {
public:
    explicit YoloTrtDetector(const std::string& engine_path, cudaStream_t stream = nullptr);
    ~YoloTrtDetector() override;

    void detect(
        const uint8_t* image_data,
        int width,
        int height,
        std::vector<core::Detection>& detections,
        float conf_threshold = 0.5f,
        float nms_threshold = 0.45f
    ) override;

    void detectAsync(
        const uint8_t* image_data,
        int width,
        int height
    ) override;

    void getDetections(
        std::vector<core::Detection>& detections,
        float conf_threshold = 0.5f,
        float nms_threshold = 0.45f
    ) override;

    void sync() override;

private:
    void preprocess(const uint8_t* image_data, int width, int height);
    void postprocess(std::vector<core::Detection>& detections,
                     float conf_threshold, float nms_threshold);

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_;
    bool owns_stream_;

    // Buffers
    void* buffers_[2];
    float* output_host_;
    int input_h_, input_w_;
    int output_size_;
    int orig_w_, orig_h_;  // Original image dimensions for scaling boxes
};

} // namespace aria::adapters::gpu
