#pragma once
#include "pipeline/SlamPipeline.hpp"
#include <string>
#include <memory>

namespace aria::factory {

/// Execution mode for pipeline creation
enum class ExecutionMode {
    GPU,        // Full GPU acceleration (production)
    CPU,        // CPU-only (debugging, profiling)
    MOCK        // Mock components (unit testing)
};

/// Factory configuration
struct FactoryConfig {
    ExecutionMode mode = ExecutionMode::GPU;

    // GPU settings
    std::string yolo_engine_path = "../models/yolo26s.engine";
    int cuda_device = 0;

    // Feature extraction
    int max_features = 1000;

    // Pipeline config
    pipeline::PipelineConfig pipeline_config;
};

/// Factory for creating SLAM pipelines with different configurations
/// Implements Dependency Injection pattern
class PipelineFactory {
public:
    /// Create pipeline with full configuration
    static std::unique_ptr<pipeline::SlamPipeline> create(const FactoryConfig& config);

    /// Convenience: Create GPU pipeline (production)
    static std::unique_ptr<pipeline::SlamPipeline> createGpu(
        const std::string& yolo_engine = "../models/yolo26s.engine"
    );

    /// Convenience: Create CPU pipeline (debugging)
    static std::unique_ptr<pipeline::SlamPipeline> createCpu();

    /// Convenience: Create mock pipeline (testing)
    static std::unique_ptr<pipeline::SlamPipeline> createMock();
};

} // namespace aria::factory
