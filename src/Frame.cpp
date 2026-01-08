#include "Frame.hpp"

Frame::Frame(const cv::Mat& image) : image_(image.clone()) {
}
