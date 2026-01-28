# H03: Feature Matching

**Status:** ✅ Completed

## Objective

Match ORB features between consecutive frames for motion tracking.

## Requirements

- Brute-force matcher with Hamming distance
- Lowe's ratio test for filtering
- Match visualization

## Implementation

### Feature Matcher

```cpp
#include <opencv2/opencv.hpp>

class FeatureMatcher {
public:
    FeatureMatcher(float ratio_thresh = 0.7f)
        : ratio_thresh_(ratio_thresh) {
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
    }

    std::vector<cv::DMatch> match(const cv::Mat& desc1,
                                   const cv::Mat& desc2) {
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(desc1, desc2, knn_matches, 2);

        // Lowe's ratio test
        std::vector<cv::DMatch> good_matches;
        for (const auto& m : knn_matches) {
            if (m.size() >= 2 &&
                m[0].distance < ratio_thresh_ * m[1].distance) {
                good_matches.push_back(m[0]);
            }
        }

        return good_matches;
    }

private:
    cv::Ptr<cv::BFMatcher> matcher_;
    float ratio_thresh_;
};
```

### Match Visualization

```cpp
cv::Mat visualizeMatches(const cv::Mat& img1,
                         const std::vector<cv::KeyPoint>& kp1,
                         const cv::Mat& img2,
                         const std::vector<cv::KeyPoint>& kp2,
                         const std::vector<cv::DMatch>& matches) {
    cv::Mat output;
    cv::drawMatches(img1, kp1, img2, kp2, matches, output,
                    cv::Scalar(0, 255, 0),
                    cv::Scalar(255, 0, 0));
    return output;
}
```

## Files Modified

- `include/FeatureExtractor.hpp` → Added matcher
- `src/FeatureExtractor.cpp` → Implemented matching

## Lowe's Ratio Test

The ratio test filters ambiguous matches:

```
if (best_match.distance < 0.7 * second_best.distance)
    accept match
else
    reject (ambiguous)
```

This removes ~40-60% of false matches while keeping most true matches.

## Performance

| Metric | Value |
|--------|-------|
| Matches per frame pair | 200-500 |
| Matching time (CPU) | ~5ms |
| Inlier ratio after ratio test | ~60-80% |

## Next Steps

→ H04: Pose Estimation
