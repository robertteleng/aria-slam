#include "TRTInference.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

TRTInference::TRTInference(const std::string& engine_path) {
    // Load engine file
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // Create runtime and engine
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    context_ = engine_->createExecutionContext();

    // Get input/output binding info
    input_idx_ = 0;  // YOLO input is always index 0
    output_idx_ = 1; // YOLO output is always index 1

    auto input_dims = engine_->getTensorShape(engine_->getIOTensorName(input_idx_));
    input_h_ = input_dims.d[2];  // NCHW format
    input_w_ = input_dims.d[3];

    auto output_dims = engine_->getTensorShape(engine_->getIOTensorName(output_idx_));
    output_size_ = 1;
    for (int i = 0; i < output_dims.nbDims; i++) {
        output_size_ *= output_dims.d[i];
    }

    // Allocate GPU buffers
    size_t input_size = 3 * input_h_ * input_w_ * sizeof(float);
    cudaMalloc(&buffers_[input_idx_], input_size);
    cudaMalloc(&buffers_[output_idx_], output_size_ * sizeof(float));

    cudaStreamCreate(&stream_);

    std::cout << "TensorRT engine loaded: " << input_w_ << "x" << input_h_
              << " -> " << output_size_ << " outputs" << std::endl;
}

TRTInference::~TRTInference() {
    cudaStreamDestroy(stream_);
    cudaFree(buffers_[0]);
    cudaFree(buffers_[1]);
    delete context_;
    delete engine_;
    delete runtime_;
}

void TRTInference::preprocess(const cv::Mat& image, float* gpu_input, cudaStream_t stream) {
    // Resize to model input size
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_w_, input_h_));

    // Convert BGR to RGB and normalize to [0, 1]
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0f / 255.0f);

    // HWC to CHW (planar format)
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> input_data(3 * input_h_ * input_w_);
    int channel_size = input_h_ * input_w_;
    for (int c = 0; c < 3; c++) {
        memcpy(input_data.data() + c * channel_size,
               channels[c].data, channel_size * sizeof(float));
    }

    // Copy to GPU using provided stream
    cudaMemcpyAsync(gpu_input, input_data.data(),
                    input_data.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
}

std::vector<Detection> TRTInference::postprocess(float* output, int num_detections,
                                                  float conf_thresh, float nms_thresh,
                                                  float scale_x, float scale_y) {
    std::vector<Detection> detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // YOLO output format: [batch, num_classes + 4, num_boxes] or [batch, num_boxes, num_classes + 4]
    // YOLOv12 uses [1, 84, 8400] format (84 = 4 box coords + 80 classes)
    int num_classes = 80;
    int num_boxes = 8400;

    for (int i = 0; i < num_boxes; i++) {
        // Get box coordinates (first 4 values)
        float cx = output[0 * num_boxes + i];
        float cy = output[1 * num_boxes + i];
        float w = output[2 * num_boxes + i];
        float h = output[3 * num_boxes + i];

        // Find best class
        float max_conf = 0;
        int max_class = 0;
        for (int c = 0; c < num_classes; c++) {
            float conf = output[(4 + c) * num_boxes + i];
            if (conf > max_conf) {
                max_conf = conf;
                max_class = c;
            }
        }

        if (max_conf >= conf_thresh) {
            // Convert center format to corner format
            int x1 = (int)((cx - w / 2) * scale_x);
            int y1 = (int)((cy - h / 2) * scale_y);
            int x2 = (int)((cx + w / 2) * scale_x);
            int y2 = (int)((cy + h / 2) * scale_y);

            boxes.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            confidences.push_back(max_conf);
            class_ids.push_back(max_class);
        }
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, nms_thresh, indices);

    for (int idx : indices) {
        Detection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }

    return detections;
}

// Synchronous detection (original API, backward compatible)
std::vector<Detection> TRTInference::detect(const cv::Mat& image, float conf_thresh, float nms_thresh) {
    // Preprocess
    preprocess(image, (float*)buffers_[input_idx_], stream_);

    // Set tensor addresses
    context_->setTensorAddress(engine_->getIOTensorName(input_idx_), buffers_[input_idx_]);
    context_->setTensorAddress(engine_->getIOTensorName(output_idx_), buffers_[output_idx_]);

    // Run inference
    context_->enqueueV3(stream_);

    // Copy output to CPU
    std::vector<float> output(output_size_);
    cudaMemcpyAsync(output.data(), buffers_[output_idx_],
                    output_size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Postprocess
    float scale_x = (float)image.cols / input_w_;
    float scale_y = (float)image.rows / input_h_;

    return postprocess(output.data(), output_size_, conf_thresh, nms_thresh, scale_x, scale_y);
}

// H11: Async detection with external CUDA stream
void TRTInference::detectAsync(const cv::Mat& image, cudaStream_t stream) {
    // Store scale factors for later postprocessing
    last_scale_x_ = (float)image.cols / input_w_;
    last_scale_y_ = (float)image.rows / input_h_;

    // Preprocess (CPU work + async copy to GPU)
    preprocess(image, (float*)buffers_[input_idx_], stream);

    // Set tensor addresses
    context_->setTensorAddress(engine_->getIOTensorName(input_idx_), buffers_[input_idx_]);
    context_->setTensorAddress(engine_->getIOTensorName(output_idx_), buffers_[output_idx_]);

    // Run inference asynchronously on provided stream
    context_->enqueueV3(stream);

    // Prepare output buffer and async copy results
    output_buffer_.resize(output_size_);
    cudaMemcpyAsync(output_buffer_.data(), buffers_[output_idx_],
                    output_size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    // No synchronization here - caller must sync before calling getDetections()
}

// H11: Get results after stream synchronization
std::vector<Detection> TRTInference::getDetections(float conf_thresh, float nms_thresh) {
    return postprocess(output_buffer_.data(), output_size_,
                       conf_thresh, nms_thresh,
                       last_scale_x_, last_scale_y_);
}
