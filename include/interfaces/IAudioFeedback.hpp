#pragma once
#include <memory>
#include <string>

namespace aria::interfaces {

/// Audio feedback priority levels
enum class AudioPriority {
    LOW,        // Background info
    MEDIUM,     // Normal navigation
    HIGH,       // Important alerts
    CRITICAL    // Immediate danger
};

/// Spatial direction for audio cues
enum class AudioDirection {
    CENTER,
    LEFT,
    RIGHT,
    BEHIND
};

/// Abstract interface for audio feedback system
/// Implementations: PulseAudioAdapter (Linux), MockAudio (test)
class IAudioFeedback {
public:
    virtual ~IAudioFeedback() = default;

    /// Initialize audio system
    /// @return true if initialization successful
    virtual bool initialize() = 0;

    /// Shutdown audio system
    virtual void shutdown() = 0;

    /// Check if audio system is ready
    virtual bool isReady() const = 0;

    /// Speak text using TTS
    /// @param text Message to speak
    /// @param priority Message priority (higher can interrupt lower)
    /// @param interrupt If true, stop current speech immediately
    virtual void speak(
        const std::string& text,
        AudioPriority priority = AudioPriority::MEDIUM,
        bool interrupt = false
    ) = 0;

    /// Play spatial beep for directional alert
    /// @param direction Where the sound should appear to come from
    /// @param frequency_hz Tone frequency (higher = more urgent)
    /// @param duration_ms How long to play
    /// @param volume 0.0 to 1.0
    virtual void playBeep(
        AudioDirection direction,
        int frequency_hz = 800,
        int duration_ms = 200,
        float volume = 0.7f
    ) = 0;

    /// Play critical alert (obstacle very close)
    /// @param direction Direction of danger
    virtual void playCriticalAlert(AudioDirection direction) = 0;

    /// Set global volume
    /// @param volume 0.0 to 1.0
    virtual void setVolume(float volume) = 0;

    /// Get current global volume
    virtual float getVolume() const = 0;

    /// Enable/disable all audio
    virtual void setMuted(bool muted) = 0;
    virtual bool isMuted() const = 0;

    /// Process pending audio (call in main loop if needed)
    virtual void spinOnce() = 0;
};

using AudioFeedbackPtr = std::unique_ptr<IAudioFeedback>;

} // namespace aria::interfaces
