#pragma once
#include "interfaces/IFeatureExtractor.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include <cuda_runtime.h>

namespace aria::adapters::gpu {

/// GPU-accelerated ORB feature extractor using CUDA
/// Wraps cv::cuda::ORB and converts to domain types
class OrbCudaExtractor : public interfaces::IFeatureExtractor {
public:
    explicit OrbCudaExtractor(int max_features = 1000, cudaStream_t stream = nullptr);
    ~OrbCudaExtractor() override;

    void extract(
        const uint8_t* image_data,
        int width,
        int height,
        core::Frame& frame
    ) override;

    void extractAsync(
        const uint8_t* image_data,
        int width,
        int height,
        core::Frame& frame
    ) override;

    void sync() override;

    void setMaxFeatures(int n) override;
    int getMaxFeatures() const override { return max_features_; }

    /// GPU-specific: get descriptors without download (for GPU matching)
    const cv::cuda::GpuMat& getGpuDescriptors() const { return gpu_descriptors_; }

private:
    cv::Ptr<cv::cuda::ORB> orb_;
    cv::cuda::GpuMat gpu_image_;
    cv::cuda::GpuMat gpu_keypoints_;
    cv::cuda::GpuMat gpu_descriptors_;
    cv::cuda::Stream cv_stream_;
    cudaStream_t cuda_stream_;
    int max_features_;
    bool owns_stream_;

    // Para operaciones async
    core::Frame* pending_frame_ = nullptr;
    int pending_width_ = 0;
    int pending_height_ = 0;
};

} // namespace aria::adapters::gpu
