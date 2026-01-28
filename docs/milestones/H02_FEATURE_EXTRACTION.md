# H02: Feature Extraction

**Status:** ✅ Completed

## Objective

Implement ORB feature detection for identifying distinctive points in video frames.

## Requirements

- ORB (Oriented FAST and Rotated BRIEF) detector
- Configurable number of features
- Keypoint visualization

## Implementation

### ORB Feature Detector

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

class FeatureExtractor {
public:
    FeatureExtractor(int nfeatures = 2000) {
        orb_ = cv::ORB::create(nfeatures);
    }

    void detect(const cv::Mat& image,
                std::vector<cv::KeyPoint>& keypoints,
                cv::Mat& descriptors) {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }

        orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    }

private:
    cv::Ptr<cv::ORB> orb_;
};
```

### Visualization

```cpp
cv::Mat visualizeKeypoints(const cv::Mat& image,
                           const std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output,
                      cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return output;
}
```

## Files Created

- `include/FeatureExtractor.hpp`
- `src/FeatureExtractor.cpp`

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| OpenCV | >= 4.6 | ORB feature detection |

## Why ORB?

- **Fast**: Suitable for real-time applications
- **Rotation invariant**: Handles camera rotation
- **Scale invariant**: Works at different distances
- **Binary descriptors**: Efficient matching with Hamming distance

## Performance

| Metric | Value |
|--------|-------|
| Features per frame | 2000 |
| Detection time (CPU) | ~15ms |
| Descriptor size | 32 bytes |

## Next Steps

→ H03: Feature Matching
