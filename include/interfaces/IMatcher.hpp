#pragma once
#include "core/Types.hpp"
#include <memory>
#include <vector>

namespace aria::interfaces {

/// Abstract interface for descriptor matching
/// Implementations: CudaMatcher (GPU), BruteForceMatcher (CPU), MockMatcher (test)
class IMatcher {
public:
    virtual ~IMatcher() = default;

    /// Match descriptors between two frames
    /// @param query Query frame (current)
    /// @param train Train frame (previous/reference)
    /// @param matches Output matches
    /// @param ratio_threshold Lowe's ratio test threshold (0.0 = disabled)
    virtual void match(
        const core::Frame& query,
        const core::Frame& train,
        std::vector<core::Match>& matches,
        float ratio_threshold = 0.75f
    ) = 0;

    /// Match one frame against multiple (for loop closure)
    virtual void matchMultiple(
        const core::Frame& query,
        const std::vector<core::Frame>& candidates,
        std::vector<std::vector<core::Match>>& all_matches,
        float ratio_threshold = 0.75f
    ) {
        all_matches.resize(candidates.size());
        for (size_t i = 0; i < candidates.size(); i++) {
            match(query, candidates[i], all_matches[i], ratio_threshold);
        }
    }
};

using MatcherPtr = std::unique_ptr<IMatcher>;

} // namespace aria::interfaces
