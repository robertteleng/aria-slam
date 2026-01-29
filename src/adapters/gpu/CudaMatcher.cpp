#include "adapters/gpu/CudaMatcher.hpp"

namespace aria::adapters {

void CudaMatcher::match(
    const core::Frame& query,
    const core::Frame& train,
    std::vector<core::Match>& matches,
    float ratio_threshold
) {
    std::vector<std::vector<cv::DMatch>> knn_matches;

    // TODO: Convertir query.descriptors y train.descriptors a GpuMat
    // y llamar a matcher_->knnMatch(query_gpu, train_gpu, knn_matches, 2)

    for (auto& knn : knn_matches) {
    if (knn.size() >= 2 && knn[0].distance < ratio_threshold * knn[1].distance) {
        core::Match m;
        m.query_idx = knn[0].queryIdx;
        m.train_idx = knn[0].trainIdx;
        m.distance = knn[0].distance;
        matches.push_back(m);
    }
}
}

} // namespace aria::adapters