# H14: GPU-Accelerated Loop Closure

**Status:** 🔲 Pending

## Objective

Move loop closure detection to GPU to relieve CPU bottleneck, especially critical for Jetson Orin Nano.

## Motivation

Current CPU-based loop closure:
- Linear search through keyframes: O(n)
- Descriptor matching: ~50ms per candidate
- Blocks tracking thread on weak CPUs

GPU solution:
- Parallel search across all keyframes
- Batch descriptor matching: ~5ms total
- Runs async, doesn't block tracking

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GPU                                      │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Keyframe    │     │   Batch      │     │  Geometric   │    │
│  │  Database    │ ──► │   Matcher    │ ──► │  Verify      │    │
│  │  (GPU Mem)   │     │  (cuBLAS)    │     │  (RANSAC)    │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
         ▲                                           │
         │                                           ▼
    [Add KF]                                  [Loop Candidate]
         │                                           │
┌─────────────────────────────────────────────────────────────────┐
│                         CPU                                      │
│  Tracking Thread ────────────────────► Loop Closure Thread      │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### GPU Keyframe Database

```cpp
class GpuKeyframeDatabase {
public:
    GpuKeyframeDatabase(int max_keyframes = 500, int descriptor_size = 32) {
        // Allocate GPU memory for all descriptors
        // Shape: [max_keyframes, max_features, descriptor_size]
        d_descriptors_.create(max_keyframes * MAX_FEATURES, descriptor_size, CV_8UC1);
        d_counts_.create(1, max_keyframes, CV_32SC1);
    }

    void addKeyframe(int id, const cv::Mat& descriptors) {
        int offset = id * MAX_FEATURES;

        // Upload descriptors to GPU
        cv::cuda::GpuMat region = d_descriptors_.rowRange(offset, offset + descriptors.rows);
        region.upload(descriptors);

        // Update count
        counts_[id] = descriptors.rows;
        d_counts_.upload(cv::Mat(counts_));

        keyframe_ids_.push_back(id);
    }

    cv::cuda::GpuMat getDescriptors(int id) const {
        int offset = id * MAX_FEATURES;
        int count = counts_[id];
        return d_descriptors_.rowRange(offset, offset + count);
    }

private:
    static const int MAX_FEATURES = 2000;
    cv::cuda::GpuMat d_descriptors_;
    cv::cuda::GpuMat d_counts_;
    std::vector<int> counts_;
    std::vector<int> keyframe_ids_;
};
```

### Batch GPU Matcher

```cpp
class GpuBatchMatcher {
public:
    GpuBatchMatcher() {
        matcher_ = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    }

    // Match query against all keyframes in parallel
    std::vector<MatchResult> matchAll(const cv::cuda::GpuMat& query_desc,
                                       const GpuKeyframeDatabase& db,
                                       cv::cuda::Stream& stream) {
        std::vector<MatchResult> results;

        // Upload query once
        for (int id : db.getKeyframeIds()) {
            cv::cuda::GpuMat kf_desc = db.getDescriptors(id);

            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher_->knnMatchAsync(query_desc, kf_desc, d_matches_, 2, cv::noArray(), stream);
        }

        stream.waitForCompletion();

        // Download and apply ratio test (could also be GPU kernel)
        for (size_t i = 0; i < results.size(); i++) {
            results[i].score = applyRatioTest(results[i].matches);
        }

        return results;
    }

private:
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher_;
    cv::cuda::GpuMat d_matches_;
};
```

### GPU RANSAC (Optional Advanced)

```cpp
// Custom CUDA kernel for parallel RANSAC
__global__ void ransacFundamentalKernel(
    const float2* pts1, const float2* pts2, int num_points,
    const int* sample_indices, int num_samples,
    float* scores, float3x3* models)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_samples) return;

    // Get 8 random points for this sample
    int* idx = sample_indices + tid * 8;

    // Compute fundamental matrix from 8 points
    float3x3 F = computeFundamental8Point(pts1, pts2, idx);

    // Count inliers
    int inliers = 0;
    for (int i = 0; i < num_points; i++) {
        float error = computeEpipolarError(F, pts1[i], pts2[i]);
        if (error < 3.0f) inliers++;
    }

    scores[tid] = inliers;
    models[tid] = F;
}
```

### Integrated GPU Loop Detector

```cpp
class GpuLoopClosureDetector {
public:
    GpuLoopClosureDetector(int min_gap = 30, double min_score = 0.3)
        : min_gap_(min_gap), min_score_(min_score) {}

    void addKeyframe(const KeyFrame& kf) {
        // Upload descriptors to GPU database
        d_database_.addKeyframe(kf.id, kf.descriptors);
        keyframes_.push_back(kf);
    }

    bool detect(const KeyFrame& query, LoopCandidate& candidate) {
        // Upload query descriptors
        cv::cuda::GpuMat d_query;
        d_query.upload(query.descriptors);

        // Batch match against all keyframes
        auto results = matcher_.matchAll(d_query, d_database_, stream_);

        // Find best candidate
        for (const auto& result : results) {
            if (result.score < min_score_) continue;
            if (query.id - result.kf_id < min_gap_) continue;

            // Geometric verification (can also be GPU)
            if (verifyGeometry(query, keyframes_[result.kf_id], result.matches, candidate)) {
                return true;
            }
        }

        return false;
    }

private:
    GpuKeyframeDatabase d_database_;
    GpuBatchMatcher matcher_;
    cv::cuda::Stream stream_;

    std::vector<KeyFrame> keyframes_;
    int min_gap_;
    double min_score_;
};
```

## Memory Management

```cpp
// Estimate GPU memory for loop closure
// 500 keyframes * 2000 features * 32 bytes = 32 MB descriptors
// Plus working memory: ~16 MB
// Total: ~50 MB dedicated to loop closure

void checkGpuMemory() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);

    // Reserve 50 MB for loop closure
    if (free < 50 * 1024 * 1024) {
        // Reduce keyframe database size
        d_database_.pruneOldest(100);
    }
}
```

## Performance Comparison

| Operation | CPU | GPU (RTX 2060) | Speedup |
|-----------|-----|----------------|---------|
| Match 1 keyframe | 5ms | 0.8ms | 6x |
| Match 500 keyframes | 250ms | 8ms | 31x |
| Full loop detection | 300ms | 15ms | 20x |

### Jetson Orin Nano

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Match 500 keyframes | 800ms | 25ms | 32x |
| Full loop detection | 1000ms | 40ms | 25x |

## Integration with Async Pipeline

```cpp
class AsyncGpuLoopDetector {
public:
    void run() {
        while (!stop_) {
            auto kf = keyframe_queue_.pop();
            if (!kf) continue;

            // GPU detection (non-blocking CPU)
            LoopCandidate candidate;
            if (detector_.detect(*kf, candidate)) {
                loop_queue_.push(candidate);
            }

            // CPU is free during GPU work
        }
    }
};
```

## Dependencies

- CUDA 11.0+
- OpenCV CUDA modules
- Optional: cuBLAS for optimized matrix ops

## Next Steps

→ H15: Bundle Adjustment (GPU Ceres)
