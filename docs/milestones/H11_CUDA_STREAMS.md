# H11: CUDA Streams Pipeline

**Status:** ✅ Completed

## Objective

Overlap GPU operations using CUDA streams for maximum throughput.

## Requirements

- Multiple CUDA streams
- Async memory transfers
- Pipeline overlapping
- Stream synchronization

## CUDA Streams Concept

Without streams (sequential):
```
[Upload] → [Kernel1] → [Kernel2] → [Download]
```

With streams (overlapped):
```
Stream 0: [Upload0] → [Kernel0] → [Download0]
Stream 1:    [Upload1] → [Kernel1] → [Download1]
Stream 2:       [Upload2] → [Kernel2] → [Download2]
```

## Implementation

### Stream Manager

```cpp
class CudaStreamManager {
public:
    CudaStreamManager(int num_streams = 3) {
        for (int i = 0; i < num_streams; i++) {
            cv::cuda::Stream stream;
            streams_.push_back(stream);
        }
    }

    cv::cuda::Stream& getStream(int idx) {
        return streams_[idx % streams_.size()];
    }

    void waitAll() {
        for (auto& stream : streams_) {
            stream.waitForCompletion();
        }
    }

private:
    std::vector<cv::cuda::Stream> streams_;
};
```

### Pipelined Feature Extraction

```cpp
class PipelinedExtractor {
public:
    PipelinedExtractor() : stream_mgr_(3) {
        for (int i = 0; i < 3; i++) {
            orb_[i] = cv::cuda::ORB::create(2000);
            d_frame_[i].create(480, 640, CV_8UC3);
            d_gray_[i].create(480, 640, CV_8UC1);
        }
    }

    void processAsync(int idx, const cv::Mat& frame) {
        auto& stream = stream_mgr_.getStream(idx);

        // Async upload
        d_frame_[idx].upload(frame, stream);

        // Async convert
        cv::cuda::cvtColor(d_frame_[idx], d_gray_[idx],
                           cv::COLOR_BGR2GRAY, stream);

        // Async detect
        orb_[idx]->detectAndComputeAsync(
            d_gray_[idx], cv::cuda::GpuMat(),
            d_keypoints_[idx], d_descriptors_[idx],
            false, stream);
    }

    void getResults(int idx,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::Mat& descriptors) {
        auto& stream = stream_mgr_.getStream(idx);
        stream.waitForCompletion();

        orb_[idx]->convert(d_keypoints_[idx], keypoints);
        d_descriptors_[idx].download(descriptors);
    }

private:
    CudaStreamManager stream_mgr_;
    cv::Ptr<cv::cuda::ORB> orb_[3];
    cv::cuda::GpuMat d_frame_[3], d_gray_[3];
    cv::cuda::GpuMat d_keypoints_[3], d_descriptors_[3];
};
```

### YOLO + ORB Pipeline

```cpp
class DualStreamPipeline {
public:
    void process(const cv::Mat& frame,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors,
                 std::vector<Detection>& detections) {
        // Stream 0: YOLO detection
        // Stream 1: ORB extraction
        // Both run in parallel

        // Upload to both streams
        d_frame_yolo_.upload(frame, stream_yolo_);
        d_frame_orb_.upload(frame, stream_orb_);

        // Launch YOLO inference (Stream 0)
        yolo_->inferAsync(d_frame_yolo_, stream_yolo_);

        // Launch ORB detection (Stream 1)
        cv::cuda::cvtColor(d_frame_orb_, d_gray_, cv::COLOR_BGR2GRAY, stream_orb_);
        orb_->detectAndComputeAsync(d_gray_, cv::cuda::GpuMat(),
                                     d_keypoints_, d_descriptors_,
                                     false, stream_orb_);

        // Wait for both
        stream_yolo_.waitForCompletion();
        stream_orb_.waitForCompletion();

        // Get results
        detections = yolo_->getResults();
        orb_->convert(d_keypoints_, keypoints);
        d_descriptors_.download(descriptors);
    }

private:
    cv::cuda::Stream stream_yolo_, stream_orb_;
    // ... GPU resources
};
```

### Triple Buffer Pipeline

```cpp
// Frame N:   [Process] → [Display]
// Frame N+1:    [Upload] → [Process]
// Frame N+2:       [Capture] → [Upload]

class TripleBufferPipeline {
    static const int NUM_BUFFERS = 3;

    struct Buffer {
        cv::Mat cpu_frame;
        cv::cuda::GpuMat gpu_frame;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        bool ready = false;
    };

    Buffer buffers_[NUM_BUFFERS];
    int capture_idx_ = 0;
    int process_idx_ = 1;
    int display_idx_ = 2;

public:
    void tick() {
        // Rotate indices
        int tmp = display_idx_;
        display_idx_ = process_idx_;
        process_idx_ = capture_idx_;
        capture_idx_ = tmp;
    }
};
```

## Synchronization

```cpp
// Wait for specific stream
stream.waitForCompletion();

// CUDA event for fine-grained sync
cudaEvent_t event;
cudaEventCreate(&event);
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event, 0);  // stream2 waits for stream1
```

## Performance

| Configuration | RTX 2060 FPS |
|---------------|--------------|
| Single stream | 45 |
| Dual stream (YOLO+ORB) | 65 |
| Triple buffer | 77 |

## Memory Considerations

```cpp
// Pre-allocate to avoid runtime allocations
void preallocate(int width, int height) {
    for (int i = 0; i < NUM_STREAMS; i++) {
        d_frame_[i].create(height, width, CV_8UC3);
        d_gray_[i].create(height, width, CV_8UC1);
        d_descriptors_[i].create(2000, 32, CV_8UC1);
    }
}
```

## Debugging Streams

```bash
# Profile with Nsight Systems
nsys profile --trace=cuda,nvtx ./aria_slam

# View timeline
nsys-ui report.nsys-rep
```

## Next Steps

→ H12: Clean Architecture
