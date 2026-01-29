#include "adapters/gpu/CudaMatcher.hpp"

namespace aria::adapters {

void CudaMatcher::match(
    const core::Frame& query,
    const core::Frame& train,
    std::vector<core::Match>& matches,
    float ratio_threshold
) {
    // TODO: tu código aquí
    std::vector<std::vector<cv::DMatch>> knn_matches;
    
}

} // namespace aria::adapters