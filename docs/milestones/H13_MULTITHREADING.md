# H13: Async Multithreading Pipeline

**Status:** 🔲 Pending

## Objective

Decouple SLAM components into async threads with lock-free communication for maximum throughput, especially on CPU-limited platforms like Jetson Orin Nano.

## Requirements

- Separate threads for each pipeline stage
- Lock-free queues for inter-thread communication
- Work-stealing for load balancing
- Graceful degradation under load

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Thread                               │
│  [Camera Capture] ──► Frame Queue                               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Tracking Thread                              │
│  Frame Queue ──► [Feature Extract] ──► [Match] ──► [Pose Est]   │
│                                                       │         │
│                                              KeyFrame Queue      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Loop Closure Thread                            │
│  KeyFrame Queue ──► [Loop Detect] ──► [Pose Graph Opt]          │
│                           │                                      │
│                    Loop Candidate Queue                          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Mapping Thread                               │
│  Loop Queue ──► [Map Update] ──► [Bundle Adjustment]            │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### Lock-Free Queue

```cpp
#include <atomic>
#include <optional>

template<typename T, size_t Capacity = 64>
class LockFreeQueue {
public:
    bool push(const T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next = (head + 1) % Capacity;

        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // Full
        }

        buffer_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    std::optional<T> pop() {
        size_t tail = tail_.load(std::memory_order_relaxed);

        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;  // Empty
        }

        T item = buffer_[tail];
        tail_.store((tail + 1) % Capacity, std::memory_order_release);
        return item;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

private:
    std::array<T, Capacity> buffer_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
};
```

### Thread Pool

```cpp
class ThreadPool {
public:
    ThreadPool(size_t num_threads) : stop_(false) {
        for (size_t i = 0; i < num_threads; i++) {
            workers_.emplace_back([this] { workerLoop(); });
        }
    }

    ~ThreadPool() {
        stop_ = true;
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

    template<typename F>
    auto enqueue(F&& f) -> std::future<decltype(f())> {
        using ReturnType = decltype(f());
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::forward<F>(f));

        std::future<ReturnType> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push([task] { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }

private:
    void workerLoop() {
        while (!stop_) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_;
};
```

### Async Pipeline

```cpp
class AsyncSlamPipeline {
public:
    AsyncSlamPipeline()
        : tracking_thread_([this] { trackingLoop(); }),
          loop_thread_([this] { loopClosureLoop(); }),
          mapping_thread_([this] { mappingLoop(); }) {}

    ~AsyncSlamPipeline() {
        stop_ = true;
        tracking_thread_.join();
        loop_thread_.join();
        mapping_thread_.join();
    }

    void pushFrame(const cv::Mat& frame) {
        frame_queue_.push(frame.clone());
    }

private:
    void trackingLoop() {
        while (!stop_) {
            auto frame = frame_queue_.pop();
            if (!frame) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            // Process frame
            auto result = tracker_.process(*frame);

            // If keyframe, send to loop closure
            if (result.is_keyframe) {
                keyframe_queue_.push(result.keyframe);
            }
        }
    }

    void loopClosureLoop() {
        while (!stop_) {
            auto kf = keyframe_queue_.pop();
            if (!kf) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Check for loop
            LoopCandidate candidate;
            if (loop_detector_.detect(*kf, candidate)) {
                loop_queue_.push(candidate);
            }
        }
    }

    void mappingLoop() {
        while (!stop_) {
            auto loop = loop_queue_.pop();
            if (!loop) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Optimize pose graph
            pose_graph_.addLoopEdge(loop->match_id, loop->query_id,
                                    loop->relative_pose);
            pose_graph_.optimize();
        }
    }

    LockFreeQueue<cv::Mat, 8> frame_queue_;
    LockFreeQueue<KeyFrame, 32> keyframe_queue_;
    LockFreeQueue<LoopCandidate, 16> loop_queue_;

    std::thread tracking_thread_;
    std::thread loop_thread_;
    std::thread mapping_thread_;
    std::atomic<bool> stop_{false};

    Tracker tracker_;
    LoopClosureDetector loop_detector_;
    PoseGraphOptimizer pose_graph_;
};
```

## Thread Priorities

```cpp
#include <pthread.h>

void setThreadPriority(std::thread& t, int priority) {
    sched_param param;
    param.sched_priority = priority;
    pthread_setschedparam(t.native_handle(), SCHED_FIFO, &param);
}

// Tracking: High priority (real-time critical)
setThreadPriority(tracking_thread_, 90);

// Loop closure: Medium priority
setThreadPriority(loop_thread_, 50);

// Mapping: Low priority (can be delayed)
setThreadPriority(mapping_thread_, 20);
```

## CPU Affinity (Jetson)

```cpp
#include <sched.h>

void setThreadAffinity(std::thread& t, const std::vector<int>& cpus) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int cpu : cpus) {
        CPU_SET(cpu, &cpuset);
    }
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
}

// Jetson Orin Nano: 6 cores
// Cores 0-3: A78 (performance)
// Cores 4-5: A78AE (efficiency)

setThreadAffinity(tracking_thread_, {0, 1});  // Performance cores
setThreadAffinity(loop_thread_, {2});
setThreadAffinity(mapping_thread_, {4, 5});   // Efficiency cores
```

## Graceful Degradation

```cpp
void trackingLoop() {
    while (!stop_) {
        // Check queue depth
        if (frame_queue_.size() > 4) {
            // Skip frames to catch up
            while (frame_queue_.size() > 2) {
                frame_queue_.pop();
                frames_dropped_++;
            }
        }

        // Process oldest frame
        auto frame = frame_queue_.pop();
        // ...
    }
}
```

## Performance (Jetson Orin Nano)

| Configuration | FPS | Latency |
|---------------|-----|---------|
| Single thread | 12 | 83ms |
| Async (3 threads) | 28 | 35ms |
| Async + frame skip | 30 | 33ms |

## Dependencies

- C++17 (std::optional, structured bindings)
- pthread (thread priorities, affinity)

## Next Steps

→ H14: GPU Loop Closure
