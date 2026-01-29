#pragma once
#include "core/Types.hpp"
#include <functional>
#include <memory>
#include <string>

namespace aria::interfaces {

/// Camera identifier for Meta Aria glasses
enum class AriaCamera {
    RGB,        // Center RGB camera (1408x1408 fisheye)
    SLAM_LEFT,  // Left SLAM camera (640x480 fisheye)
    SLAM_RIGHT  // Right SLAM camera (640x480 fisheye)
};

/// Raw image data from Aria camera
struct AriaImage {
    const uint8_t* data;
    int width;
    int height;
    int channels;
    uint64_t timestamp_ns;
    AriaCamera camera;
};

/// Callback types for streaming data
using ImageCallback = std::function<void(const AriaImage&)>;
using ImuCallback = std::function<void(const core::ImuMeasurement&)>;

/// Abstract interface for Meta Aria device connection
/// Implementations: AriaDeviceAdapter (pybind11), MockAriaDevice (test)
class IAriaDevice {
public:
    virtual ~IAriaDevice() = default;

    /// Connect to Aria glasses
    /// @param ip_address Optional IP for WiFi connection (empty = USB)
    /// @return true if connection successful
    virtual bool connect(const std::string& ip_address = "") = 0;

    /// Disconnect from glasses
    virtual void disconnect() = 0;

    /// Check connection status
    virtual bool isConnected() const = 0;

    /// Start streaming from specified cameras
    /// @param cameras Which cameras to stream (can combine multiple)
    /// @return true if streaming started
    virtual bool startStreaming(int cameras = 0x7) = 0;  // Default: all cameras

    /// Stop streaming
    virtual void stopStreaming() = 0;

    /// Register callback for image data
    virtual void setImageCallback(ImageCallback callback) = 0;

    /// Register callback for IMU data
    virtual void setImuCallback(ImuCallback callback) = 0;

    /// Get camera calibration (intrinsics)
    /// @param camera Which camera
    /// @param fx, fy Focal lengths
    /// @param cx, cy Principal point
    /// @return true if calibration available
    virtual bool getCalibration(
        AriaCamera camera,
        float& fx, float& fy,
        float& cx, float& cy
    ) const = 0;

    /// Process pending callbacks (call in main loop)
    virtual void spinOnce() = 0;
};

using AriaDevicePtr = std::unique_ptr<IAriaDevice>;

} // namespace aria::interfaces
