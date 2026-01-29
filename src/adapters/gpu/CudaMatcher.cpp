#include "adapters/gpu/CudaMatcher.hpp"
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace aria::adapters::gpu {

// Constructor: crea el matcher de OpenCV CUDA
CudaMatcher::CudaMatcher(cudaStream_t stream)
    : cuda_stream_(stream)
    , owns_stream_(stream == nullptr)
{
    // Crear el matcher BruteForce con distancia Hamming (para ORB)
    matcher_ = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

    // Si no nos pasaron un stream, creamos uno propio
    if (owns_stream_) {
        cudaStreamCreate(&cuda_stream_);
    }
    cv_stream_ = cv::cuda::StreamAccessor::wrapStream(cuda_stream_);
}

// Destructor: limpia recursos
CudaMatcher::~CudaMatcher() {
    if (owns_stream_ && cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
    }
}

void CudaMatcher::match(
    const core::Frame& query,
    const core::Frame& train,
    std::vector<core::Match>& matches,
    float ratio_threshold
) {
    // Si no hay descriptores, no hay nada que hacer
    if (query.descriptors.empty() || train.descriptors.empty()) {
        return;
    }

    // PARTE 1: Convertir std::vector<uint8_t> → cv::Mat
    // cv::Mat es una matriz de OpenCV (filas x columnas)
    int query_rows = query.numKeypoints();  // Número de descriptores
    int train_rows = train.numKeypoints();
    int cols = 32;  // ORB usa 32 bytes por descriptor

    // Crear cv::Mat que apunta a los datos existentes (sin copiar)
    cv::Mat query_mat(query_rows, cols, CV_8UC1, (void*)query.descriptors.data());
    cv::Mat train_mat(train_rows, cols, CV_8UC1, (void*)train.descriptors.data());

    // PARTE 2: Subir a GPU (cv::Mat → cv::cuda::GpuMat)
    cv::cuda::GpuMat query_gpu, train_gpu;
    query_gpu.upload(query_mat);
    train_gpu.upload(train_mat);

    // PARTE 3: Llamar al matcher de OpenCV CUDA
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(query_gpu, train_gpu, knn_matches, 2);

    // PARTE 4: Procesar resultados (ratio test + traducir tipos)
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

} // namespace aria::adapters::gpu