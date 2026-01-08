#include "TRTInference.hpp"
#include <fstream>
#include <iostream>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

static Logger gLogger;

TRTInference::TRTInference(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to load engine file: " + enginePath);
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(engineData.data(), size);
    context_ = engine_->createExecutionContext();
}

TRTInference::~TRTInference() {
    delete context_;
    delete engine_;
    delete runtime_;
}

std::vector<Detection> TRTInference::detect(const cv::Mat& image) {
    // TODO: implement inference
    return {};
}
