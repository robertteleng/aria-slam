#pragma once
#include "interfaces/IMatcher.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include <cuda_runtime.h>

namespace aria::adapters::gpu {

/// GPU-accelerated descriptor matcher using CUDA
/// Uses BFMatcher with Hamming distance for ORB descriptors
class CudaMatcher : public interfaces::IMatcher {
public:
    explicit CudaMatcher(cudaStream_t stream = nullptr);
    ~CudaMatcher() override;

    void match(
        const core::Frame& query,
        const core::Frame& train,
        std::vector<core::Match>& matches,
        float ratio_threshold = 0.75f
    ) override;

    /// GPU-to-GPU matching (zero-copy when used with OrbCudaExtractor)
    void matchGpu(
        const cv::cuda::GpuMat& query_desc,
        const cv::cuda::GpuMat& train_desc,
        std::vector<core::Match>& matches,
        float ratio_threshold = 0.75f
    );

private:
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher_;
    cv::cuda::Stream cv_stream_;
    cudaStream_t cuda_stream_;
    bool owns_stream_;
};

} // namespace aria::adapters::gpu
