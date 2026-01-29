#pragma once
#include "core/Types.hpp"
#include <memory>
#include <vector>
#include <string>

namespace aria::interfaces {

/// Abstract interface for 3D mapping
/// Implementations: Mapper, MockMapper (test)
class IMapper {
public:
    virtual ~IMapper() = default;

    /// Triangulate new map points from matched frames
    /// @param frame1 First frame with pose
    /// @param frame2 Second frame with pose
    /// @param pose1 Pose of first frame
    /// @param pose2 Pose of second frame
    /// @param matches Matches between frames
    /// @param K Camera intrinsic matrix (3x3)
    /// @param new_points Output: newly created map points
    virtual void triangulate(
        const core::Frame& frame1,
        const core::Frame& frame2,
        const core::Pose& pose1,
        const core::Pose& pose2,
        const std::vector<core::Match>& matches,
        const Eigen::Matrix3d& K,
        std::vector<core::MapPoint>& new_points
    ) = 0;

    /// Get all map points
    virtual const std::vector<core::MapPoint>& getMapPoints() const = 0;

    /// Export to file
    virtual void exportPLY(const std::string& filename) const = 0;
    virtual void exportPCD(const std::string& filename) const = 0;

    /// Clear map
    virtual void clear() = 0;

    /// Statistics
    virtual size_t size() const = 0;
};

using MapperPtr = std::unique_ptr<IMapper>;

} // namespace aria::interfaces
