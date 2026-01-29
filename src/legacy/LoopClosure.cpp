#include "legacy/LoopClosure.hpp"
#include <iostream>
#include <algorithm>

// g2o includes
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

// ============== LoopClosureDetector ==============

LoopClosureDetector::LoopClosureDetector(int min_frames_between,
                                         double min_score,
                                         int min_matches)
    : min_frames_between_(min_frames_between),
      min_score_(min_score),
      min_matches_(min_matches) {
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
}

void LoopClosureDetector::addKeyFrame(const KeyFrame& kf) {
    keyframes_.push_back(kf);

    // Limit database size
    while (keyframes_.size() > 500) {
        keyframes_.pop_front();
    }
}

bool LoopClosureDetector::detect(const KeyFrame& query, LoopCandidate& candidate) {
    if (keyframes_.size() < (size_t)min_frames_between_) {
        return false;
    }

    // Find candidates by descriptor similarity
    auto candidates = findCandidates(query);

    for (const auto& [idx, score] : candidates) {
        if (score < min_score_) continue;

        const KeyFrame& kf = keyframes_[idx];

        // Skip recent frames
        if (query.id - kf.id < min_frames_between_) continue;

        // Geometric verification
        std::vector<cv::DMatch> inlier_matches;
        if (verifyGeometry(query, kf, inlier_matches)) {
            // Compute relative pose
            Eigen::Matrix4d relative_pose;
            if (computeRelativePose(query, kf, inlier_matches, relative_pose)) {
                candidate.query_id = query.id;
                candidate.match_id = kf.id;
                candidate.score = score;
                candidate.matches = inlier_matches;
                candidate.relative_pose = relative_pose;

                loop_count_++;
                std::cout << "Loop detected: " << query.id << " -> " << kf.id
                          << " (score: " << score << ", matches: " << inlier_matches.size() << ")" << std::endl;
                return true;
            }
        }
    }

    return false;
}

std::vector<std::pair<int, double>> LoopClosureDetector::findCandidates(const KeyFrame& query) {
    std::vector<std::pair<int, double>> candidates;

    if (query.descriptors.empty()) return candidates;

    for (size_t i = 0; i < keyframes_.size(); i++) {
        const KeyFrame& kf = keyframes_[i];

        // Skip recent frames
        if (query.id - kf.id < min_frames_between_) continue;

        if (kf.descriptors.empty()) continue;

        // Match descriptors
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(query.descriptors, kf.descriptors, knn_matches, 2);

        // Ratio test
        int good_matches = 0;
        for (const auto& m : knn_matches) {
            if (m.size() >= 2 && m[0].distance < 0.7 * m[1].distance) {
                good_matches++;
            }
        }

        // Score = ratio of good matches
        double score = (double)good_matches / std::max(1, (int)query.keypoints.size());
        if (score > 0.1) {
            candidates.push_back({i, score});
        }
    }

    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Return top 5 candidates
    if (candidates.size() > 5) {
        candidates.resize(5);
    }

    return candidates;
}

bool LoopClosureDetector::verifyGeometry(const KeyFrame& query, const KeyFrame& candidate,
                                         std::vector<cv::DMatch>& inlier_matches) {
    // Match descriptors
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(query.descriptors, candidate.descriptors, knn_matches, 2);

    // Ratio test
    std::vector<cv::DMatch> good_matches;
    for (const auto& m : knn_matches) {
        if (m.size() >= 2 && m[0].distance < 0.7 * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    if (good_matches.size() < (size_t)min_matches_) {
        return false;
    }

    // Extract matched points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : good_matches) {
        pts1.push_back(query.keypoints[m.queryIdx].pt);
        pts2.push_back(candidate.keypoints[m.trainIdx].pt);
    }

    // RANSAC fundamental matrix estimation
    std::vector<uchar> inlier_mask;
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, inlier_mask);

    if (F.empty()) return false;

    // Collect inlier matches
    inlier_matches.clear();
    for (size_t i = 0; i < good_matches.size(); i++) {
        if (inlier_mask[i]) {
            inlier_matches.push_back(good_matches[i]);
        }
    }

    return inlier_matches.size() >= (size_t)min_matches_;
}

bool LoopClosureDetector::computeRelativePose(const KeyFrame& query, const KeyFrame& candidate,
                                               const std::vector<cv::DMatch>& matches,
                                               Eigen::Matrix4d& relative_pose) {
    if (matches.size() < 8) return false;

    // Extract matched points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(query.keypoints[m.queryIdx].pt);
        pts2.push_back(candidate.keypoints[m.trainIdx].pt);
    }

    // Approximate camera matrix
    double fx = 700, fy = 700;
    double cx = 320, cy = 180;
    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // Essential matrix
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0);
    if (E.empty()) return false;

    // Recover pose
    cv::Mat R, t;
    int inliers = cv::recoverPose(E, pts1, pts2, K, R, t);

    if (inliers < min_matches_) return false;

    // Build 4x4 transform matrix
    relative_pose = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            relative_pose(i, j) = R.at<double>(i, j);
        }
        relative_pose(i, 3) = t.at<double>(i);
    }

    return true;
}

// ============== PoseGraphOptimizer with g2o ==============

struct PoseGraphOptimizer::Impl {
    using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>>;
    using LinearSolver = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;

    std::unique_ptr<g2o::SparseOptimizer> optimizer;
    std::map<int, g2o::VertexSE3*> vertices;

    Impl() {
        optimizer = std::make_unique<g2o::SparseOptimizer>();

        auto linearSolver = std::make_unique<LinearSolver>();
        auto blockSolver = std::make_unique<BlockSolver>(std::move(linearSolver));
        auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

        optimizer->setAlgorithm(algorithm);
        optimizer->setVerbose(false);
    }

    g2o::Isometry3 toIsometry(const Eigen::Matrix4d& mat) {
        g2o::Isometry3 iso = g2o::Isometry3::Identity();
        iso.linear() = mat.block<3,3>(0,0);
        iso.translation() = mat.block<3,1>(0,3);
        return iso;
    }

    Eigen::Matrix4d fromIsometry(const g2o::Isometry3& iso) {
        Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
        mat.block<3,3>(0,0) = iso.linear();
        mat.block<3,1>(0,3) = iso.translation();
        return mat;
    }
};

PoseGraphOptimizer::PoseGraphOptimizer() : impl_(std::make_unique<Impl>()) {}

PoseGraphOptimizer::~PoseGraphOptimizer() = default;

void PoseGraphOptimizer::setInitialPose(int id, const Eigen::Matrix4d& pose) {
    if (impl_->vertices.find(id) != impl_->vertices.end()) {
        impl_->vertices[id]->setEstimate(impl_->toIsometry(pose));
        return;
    }

    auto* vertex = new g2o::VertexSE3();
    vertex->setId(id);
    vertex->setEstimate(impl_->toIsometry(pose));

    // Fix first vertex
    if (impl_->vertices.empty()) {
        vertex->setFixed(true);
    }

    impl_->optimizer->addVertex(vertex);
    impl_->vertices[id] = vertex;
}

void PoseGraphOptimizer::addOdometryEdge(int from_id, int to_id,
                                          const Eigen::Matrix4d& relative_pose,
                                          double info_scale) {
    if (impl_->vertices.find(from_id) == impl_->vertices.end() ||
        impl_->vertices.find(to_id) == impl_->vertices.end()) {
        return;
    }

    auto* edge = new g2o::EdgeSE3();
    edge->setVertex(0, impl_->vertices[from_id]);
    edge->setVertex(1, impl_->vertices[to_id]);
    edge->setMeasurement(impl_->toIsometry(relative_pose));

    // Information matrix (inverse covariance)
    Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity() * info_scale;
    edge->setInformation(info);

    impl_->optimizer->addEdge(edge);
}

void PoseGraphOptimizer::addLoopEdge(int from_id, int to_id,
                                      const Eigen::Matrix4d& relative_pose,
                                      double info_scale) {
    // Same as odometry edge but with different info scale
    addOdometryEdge(from_id, to_id, relative_pose, info_scale * 10.0);  // Higher weight for loops
}

void PoseGraphOptimizer::optimize(int iterations) {
    if (impl_->vertices.empty()) return;

    impl_->optimizer->initializeOptimization();
    impl_->optimizer->optimize(iterations);

    std::cout << "Pose graph optimized (" << iterations << " iterations, "
              << impl_->vertices.size() << " vertices)" << std::endl;
}

Eigen::Matrix4d PoseGraphOptimizer::getOptimizedPose(int id) const {
    auto it = impl_->vertices.find(id);
    if (it != impl_->vertices.end()) {
        return impl_->fromIsometry(it->second->estimate());
    }
    return Eigen::Matrix4d::Identity();
}

std::vector<Eigen::Matrix4d> PoseGraphOptimizer::getAllPoses() const {
    std::vector<Eigen::Matrix4d> result;
    for (const auto& [id, vertex] : impl_->vertices) {
        result.push_back(impl_->fromIsometry(vertex->estimate()));
    }
    return result;
}

void PoseGraphOptimizer::clear() {
    impl_->optimizer->clear();
    impl_->vertices.clear();
    impl_ = std::make_unique<Impl>();
}
