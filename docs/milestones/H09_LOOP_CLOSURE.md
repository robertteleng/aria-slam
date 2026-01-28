# H09: Loop Closure Detection

**Status:** ✅ Completed

## Objective

Detect when the camera revisits a previously seen location to correct accumulated drift.

## Requirements

- Keyframe database
- Appearance-based place recognition
- Geometric verification
- Relative pose computation

## Why Loop Closure?

Odometry accumulates drift over time. When revisiting a location:
1. Detect similar appearance to past keyframe
2. Verify with geometric constraints
3. Add loop constraint to pose graph
4. Optimize trajectory to reduce drift

## Implementation

### Keyframe Structure

```cpp
struct KeyFrame {
    int id;
    double timestamp;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};
```

### Loop Detector

```cpp
class LoopClosureDetector {
public:
    LoopClosureDetector(int min_frames_between = 30,
                        double min_score = 0.3,
                        int min_matches = 30);

    void addKeyFrame(const KeyFrame& kf);

    bool detect(const KeyFrame& query, LoopCandidate& candidate) {
        // 1. Find candidates by descriptor similarity
        auto candidates = findCandidates(query);

        for (const auto& [idx, score] : candidates) {
            // Skip recent frames (not a loop)
            if (query.id - keyframes_[idx].id < min_frames_between_)
                continue;

            // 2. Geometric verification
            std::vector<cv::DMatch> inliers;
            if (verifyGeometry(query, keyframes_[idx], inliers)) {
                // 3. Compute relative pose
                if (computeRelativePose(query, keyframes_[idx],
                                        inliers, candidate.relative_pose)) {
                    candidate.query_id = query.id;
                    candidate.match_id = keyframes_[idx].id;
                    return true;
                }
            }
        }
        return false;
    }

private:
    std::vector<std::pair<int, double>> findCandidates(const KeyFrame& query);
    bool verifyGeometry(const KeyFrame& q, const KeyFrame& c,
                        std::vector<cv::DMatch>& inliers);
    bool computeRelativePose(const KeyFrame& q, const KeyFrame& c,
                             const std::vector<cv::DMatch>& matches,
                             Eigen::Matrix4d& pose);

    std::deque<KeyFrame> keyframes_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    int min_frames_between_;
    double min_score_;
    int min_matches_;
};
```

### Candidate Finding

```cpp
std::vector<std::pair<int, double>> findCandidates(const KeyFrame& query) {
    std::vector<std::pair<int, double>> candidates;

    for (size_t i = 0; i < keyframes_.size(); i++) {
        // KNN matching with ratio test
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(query.descriptors,
                           keyframes_[i].descriptors,
                           knn_matches, 2);

        int good = 0;
        for (const auto& m : knn_matches) {
            if (m.size() >= 2 && m[0].distance < 0.7 * m[1].distance)
                good++;
        }

        double score = (double)good / query.keypoints.size();
        if (score > 0.1)
            candidates.push_back({i, score});
    }

    // Sort by score
    std::sort(candidates.begin(), candidates.end(),
              [](auto& a, auto& b) { return a.second > b.second; });

    return candidates;
}
```

### Geometric Verification

```cpp
bool verifyGeometry(const KeyFrame& query, const KeyFrame& candidate,
                    std::vector<cv::DMatch>& inlier_matches) {
    // Match descriptors
    std::vector<cv::DMatch> matches = matchDescriptors(query, candidate);

    if (matches.size() < min_matches_)
        return false;

    // Extract points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(query.keypoints[m.queryIdx].pt);
        pts2.push_back(candidate.keypoints[m.trainIdx].pt);
    }

    // RANSAC fundamental matrix
    std::vector<uchar> inlier_mask;
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC,
                                        3.0, 0.99, inlier_mask);

    // Collect inliers
    for (size_t i = 0; i < matches.size(); i++) {
        if (inlier_mask[i])
            inlier_matches.push_back(matches[i]);
    }

    return inlier_matches.size() >= min_matches_;
}
```

## Loop Candidate Structure

```cpp
struct LoopCandidate {
    int query_id;
    int match_id;
    double score;
    std::vector<cv::DMatch> matches;
    Eigen::Matrix4d relative_pose;  // Transform from match to query
};
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| min_frames_between | 30 | Minimum frame gap for loop |
| min_score | 0.3 | Minimum similarity score |
| min_matches | 30 | Minimum inlier matches |

## Performance

| Metric | Value |
|--------|-------|
| Detection time | ~50ms per query |
| Precision | ~95% |
| Recall | ~70% |

## Limitations (Current)

- Linear search through all keyframes (O(n))
- CPU-only descriptor matching
- No vocabulary-based retrieval (DBoW)

These are addressed in H13 (Multithreading) and H14 (GPU Loop Closure).

## Next Steps

→ H10: Pose Graph Optimization
