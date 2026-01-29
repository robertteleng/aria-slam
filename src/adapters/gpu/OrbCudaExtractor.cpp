/**
 * OrbCudaExtractor - Adapter que implementa IFeatureExtractor usando OpenCV CUDA
 *
 * ARQUITECTURA CLEAN:
 * - Recibe: uint8_t* (datos crudos, sin dependencia a cv::Mat)
 * - Retorna: core::Frame (tipo del dominio, sin cv::KeyPoint)
 * - Internamente: usa cv::cuda::ORB (detalle de implementación oculto)
 *
 * TRADUCCIÓN DE TIPOS:
 * - uint8_t* → cv::Mat → cv::cuda::GpuMat (entrada)
 * - cv::KeyPoint → core::KeyPoint (salida)
 * - cv::Mat descriptors → std::vector<uint8_t> (salida)
 */

#include "adapters/gpu/OrbCudaExtractor.hpp"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgproc.hpp>

namespace aria::adapters::gpu {

OrbCudaExtractor::OrbCudaExtractor(int max_features, cudaStream_t stream)
    : max_features_(max_features)
    , cuda_stream_(stream)
    , owns_stream_(stream == nullptr)
{
    // Si no nos pasan stream, creamos uno propio
    if (owns_stream_) {
        cudaStreamCreate(&cuda_stream_);
    }

    // Wrap CUDA stream en OpenCV stream
    cv_stream_ = cv::cuda::StreamAccessor::wrapStream(cuda_stream_);

    // Crear ORB detector GPU
    orb_ = cv::cuda::ORB::create(
        max_features_,  // nfeatures
        1.2f,           // scaleFactor
        8,              // nlevels
        31,             // edgeThreshold
        0,              // firstLevel
        2,              // WTA_K
        cv::ORB::HARRIS_SCORE,
        31,             // patchSize
        20              // fastThreshold
    );
}

OrbCudaExtractor::~OrbCudaExtractor() {
    if (owns_stream_ && cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
    }
}

/**
 * extract() - Versión SÍNCRONA
 *
 * Flujo:
 * 1. Convierte uint8_t* → cv::Mat (wrapper, no copia)
 * 2. Upload a GPU: cv::Mat → cv::cuda::GpuMat
 * 3. Detecta features en GPU
 * 4. Download resultados
 * 5. Traduce cv::KeyPoint → core::KeyPoint
 */
void OrbCudaExtractor::extract(
    const uint8_t* image_data,
    int width,
    int height,
    core::Frame& frame
) {
    // === PASO 1: Entrada (mundo externo → OpenCV) ===
    // cv::Mat wrapper sin copia (solo apunta a los datos)
    cv::Mat image(height, width, CV_8UC1, const_cast<uint8_t*>(image_data));

    // Convertir a grayscale si es necesario
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // === PASO 2: CPU → GPU ===
    gpu_image_.upload(gray);

    // === PASO 3: Detección en GPU ===
    orb_->detectAndComputeAsync(
        gpu_image_,
        cv::cuda::GpuMat(),  // mask (ninguna)
        gpu_keypoints_,
        gpu_descriptors_,
        false,               // useProvidedKeypoints
        cv_stream_
    );

    // Esperar a que termine (síncrono)
    cv_stream_.waitForCompletion();

    // === PASO 4: GPU → CPU ===
    std::vector<cv::KeyPoint> cv_keypoints;
    cv::Mat cv_descriptors;

    orb_->convert(gpu_keypoints_, cv_keypoints);
    gpu_descriptors_.download(cv_descriptors);

    // === PASO 5: Traducción OpenCV → Dominio ===
    // AQUÍ está la magia de Clean Architecture:
    // Convertimos tipos de OpenCV a tipos del dominio

    frame.width = width;
    frame.height = height;
    frame.keypoints.clear();
    frame.keypoints.reserve(cv_keypoints.size());

    for (const auto& kp : cv_keypoints) {
        core::KeyPoint domain_kp;
        domain_kp.x = kp.pt.x;
        domain_kp.y = kp.pt.y;
        domain_kp.size = kp.size;
        domain_kp.angle = kp.angle;
        domain_kp.response = kp.response;
        domain_kp.octave = kp.octave;
        frame.keypoints.push_back(domain_kp);
    }

    // Descriptores: cv::Mat (N x 32) → std::vector<uint8_t> (N * 32)
    frame.descriptors.resize(cv_descriptors.total());
    std::memcpy(frame.descriptors.data(), cv_descriptors.data, cv_descriptors.total());
}

/**
 * extractAsync() - Versión ASÍNCRONA
 *
 * Lanza la operación y retorna inmediatamente.
 * Los resultados se obtienen después de llamar sync().
 */
void OrbCudaExtractor::extractAsync(
    const uint8_t* image_data,
    int width,
    int height,
    core::Frame& frame
) {
    // Guardar referencia al frame para llenar después
    pending_frame_ = &frame;
    pending_width_ = width;
    pending_height_ = height;

    // Wrapper sin copia
    cv::Mat image(height, width, CV_8UC1, const_cast<uint8_t*>(image_data));

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // Upload async
    gpu_image_.upload(gray, cv_stream_);

    // Detección async (no bloquea)
    orb_->detectAndComputeAsync(
        gpu_image_,
        cv::cuda::GpuMat(),
        gpu_keypoints_,
        gpu_descriptors_,
        false,
        cv_stream_
    );

    // Retorna inmediatamente, el GPU trabaja en background
}

/**
 * sync() - Espera y descarga resultados
 */
void OrbCudaExtractor::sync() {
    if (!pending_frame_) return;

    // Esperar a que el GPU termine
    cv_stream_.waitForCompletion();

    // Descargar y traducir
    std::vector<cv::KeyPoint> cv_keypoints;
    cv::Mat cv_descriptors;

    orb_->convert(gpu_keypoints_, cv_keypoints);
    gpu_descriptors_.download(cv_descriptors);

    // Traducir a tipos del dominio
    pending_frame_->width = pending_width_;
    pending_frame_->height = pending_height_;
    pending_frame_->keypoints.clear();
    pending_frame_->keypoints.reserve(cv_keypoints.size());

    for (const auto& kp : cv_keypoints) {
        core::KeyPoint domain_kp;
        domain_kp.x = kp.pt.x;
        domain_kp.y = kp.pt.y;
        domain_kp.size = kp.size;
        domain_kp.angle = kp.angle;
        domain_kp.response = kp.response;
        domain_kp.octave = kp.octave;
        pending_frame_->keypoints.push_back(domain_kp);
    }

    pending_frame_->descriptors.resize(cv_descriptors.total());
    std::memcpy(pending_frame_->descriptors.data(), cv_descriptors.data, cv_descriptors.total());

    pending_frame_ = nullptr;
}

void OrbCudaExtractor::setMaxFeatures(int n) {
    max_features_ = n;
    // Recrear ORB con nuevo límite
    orb_ = cv::cuda::ORB::create(max_features_);
}

} // namespace aria::adapters::gpu
