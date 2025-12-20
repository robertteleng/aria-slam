#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>

/**
 * @brief Stores image data and ORB features for SLAM processing
 *
 * Supports both GPU (CUDA) and CPU feature detection.
 * GPU version provides ~2x speedup on RTX 2060.
 */
class Frame {
public:
    cv::Mat image;                        ///< Original BGR image
    std::vector<cv::KeyPoint> keypoints;  ///< Detected ORB keypoints
    cv::Mat descriptors;                  ///< ORB descriptors (binary, 32 bytes each)

    /**
     * @brief Construct frame with GPU-accelerated ORB detection
     * @param img Input BGR image
     * @param orb_gpu CUDA ORB detector instance
     */
    Frame(const cv::Mat& img, cv::Ptr<cv::cuda::ORB> orb_gpu);

    /**
     * @brief Construct frame with CPU ORB detection (fallback)
     * @param img Input BGR image
     * @param orb CPU ORB detector instance
     */
    Frame(const cv::Mat& img, cv::Ptr<cv::ORB> orb);

    /**
     * @brief Deep copy constructor
     * @param other Frame to copy
     */
    Frame(const Frame& other);
};