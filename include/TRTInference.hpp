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

/**
 * @brief TensorRT inference wrapper for YOLO object detection
 *
 * H11: Supports async execution with CUDA streams for parallel processing.
 */
class TRTInference {
public:
    TRTInference(const std::string& engine_path);
    ~TRTInference();

    /**
     * @brief Synchronous detection (original API, backward compatible)
     */
    std::vector<Detection> detect(const cv::Mat& image, float conf_thresh = 0.5f, float nms_thresh = 0.45f);

    /**
     * @brief Start async inference on provided CUDA stream
     * @param image Input BGR image
     * @param stream CUDA stream for async execution
     *
     * After calling this, use cudaStreamSynchronize() or cudaDeviceSynchronize()
     * then call getDetections() to retrieve results.
     */
    void detectAsync(const cv::Mat& image, cudaStream_t stream);

    /**
     * @brief Get detection results after async inference completes
     * @param conf_thresh Confidence threshold
     * @param nms_thresh NMS threshold
     * @return Vector of detections
     *
     * Must be called after stream synchronization when using detectAsync().
     */
    std::vector<Detection> getDetections(float conf_thresh = 0.5f, float nms_thresh = 0.45f);

private:
    void preprocess(const cv::Mat& image, float* gpu_input, cudaStream_t stream);
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

    cudaStream_t stream_;  // Internal stream for sync mode

    // Async state
    std::vector<float> output_buffer_;  // CPU buffer for async results
    float last_scale_x_ = 1.0f;
    float last_scale_y_ = 1.0f;
};
