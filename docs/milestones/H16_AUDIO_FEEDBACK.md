# H16: Audio Feedback System

**Status:** 📋 Planned
**Dependencies:** H12 (Clean Architecture), H15 (Meta Aria - opcional)

---

## Objetivo

Implementar sistema de audio feedback en C++ nativo para:
- TTS (Text-to-Speech) para anuncios de navegación
- Beeps espaciales (stereo) para alertas direccionales
- Sistema de prioridades y cooldowns
- Latencia mínima para alertas críticas

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                     Audio Feedback System                        │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │  SlamPipeline    │                                           │
│  │  (detecciones)   │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐      ┌──────────────────┐                │
│  │ NavigationEngine │─────►│  AudioRouter     │                │
│  │ (decide qué      │      │  (prioriza +     │                │
│  │  anunciar)       │      │   cooldowns)     │                │
│  └──────────────────┘      └────────┬─────────┘                │
│                                     │                           │
│                     ┌───────────────┼───────────────┐          │
│                     ▼               ▼               ▼          │
│           ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│           │ TTS Thread  │  │ Beep Thread │  │ Alert Queue │   │
│           │ (espeak-ng) │  │ (PulseAudio)│  │ (críticos)  │   │
│           └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Interface Definida

```cpp
// include/interfaces/IAudioFeedback.hpp

enum class AudioPriority {
    LOW,        // Background info
    MEDIUM,     // Normal navigation
    HIGH,       // Important alerts
    CRITICAL    // Immediate danger
};

enum class AudioDirection {
    CENTER,
    LEFT,
    RIGHT,
    BEHIND
};

class IAudioFeedback {
public:
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;

    // TTS
    virtual void speak(
        const std::string& text,
        AudioPriority priority = AudioPriority::MEDIUM,
        bool interrupt = false
    ) = 0;

    // Spatial beeps
    virtual void playBeep(
        AudioDirection direction,
        int frequency_hz = 800,
        int duration_ms = 200,
        float volume = 0.7f
    ) = 0;

    // Critical alert (high priority, can't be ignored)
    virtual void playCriticalAlert(AudioDirection direction) = 0;

    // Volume control
    virtual void setVolume(float volume) = 0;
    virtual void setMuted(bool muted) = 0;

    virtual void spinOnce() = 0;
};
```

## Implementación: PulseAudioAdapter

### Dependencias Linux

```bash
# Ubuntu/Debian
sudo apt install libpulse-dev espeak-ng libespeak-ng-dev

# O con PipeWire (moderno)
sudo apt install libpipewire-0.3-dev
```

### Estructura

```
src/adapters/audio/
├── PulseAudioAdapter.hpp
├── PulseAudioAdapter.cpp     # Implementación principal
├── TTSEngine.hpp             # Wrapper para espeak-ng
├── TTSEngine.cpp
├── BeepGenerator.hpp         # Generación de tonos
├── BeepGenerator.cpp
└── MockAudioFeedback.cpp     # Para testing
```

### Código Principal

```cpp
// src/adapters/audio/PulseAudioAdapter.cpp

#include <pulse/simple.h>
#include <pulse/error.h>
#include <espeak-ng/speak_lib.h>
#include <thread>
#include <queue>
#include <mutex>
#include <cmath>

class PulseAudioAdapter : public IAudioFeedback {
public:
    bool initialize() override {
        // Inicializar PulseAudio para beeps
        pa_sample_spec spec;
        spec.format = PA_SAMPLE_S16LE;
        spec.channels = 2;  // Stereo para audio espacial
        spec.rate = 44100;

        int error;
        pulse_ = pa_simple_new(
            nullptr,            // Servidor por defecto
            "aria-slam",        // Nombre de la app
            PA_STREAM_PLAYBACK,
            nullptr,            // Device por defecto
            "navigation",       // Descripción del stream
            &spec,
            nullptr,            // Channel map por defecto
            nullptr,            // Buffering por defecto
            &error
        );

        if (!pulse_) {
            std::cerr << "PulseAudio error: " << pa_strerror(error) << std::endl;
            return false;
        }

        // Inicializar espeak-ng para TTS
        int result = espeak_Initialize(
            AUDIO_OUTPUT_SYNTH_BUF,  // Buffer interno
            500,                      // Buffer length
            nullptr,                  // Path por defecto
            0                         // Options
        );

        if (result == EE_INTERNAL_ERROR) {
            std::cerr << "espeak-ng initialization failed" << std::endl;
            return false;
        }

        // Configurar voz
        espeak_SetVoiceByName("en");  // Inglés
        espeak_SetParameter(espeakRATE, 175, 0);     // Velocidad
        espeak_SetParameter(espeakVOLUME, 100, 0);   // Volumen
        espeak_SetParameter(espeakPITCH, 50, 0);     // Tono

        // Iniciar threads
        running_ = true;
        tts_thread_ = std::thread(&PulseAudioAdapter::ttsWorker, this);
        beep_thread_ = std::thread(&PulseAudioAdapter::beepWorker, this);

        initialized_ = true;
        return true;
    }

    void speak(const std::string& text, AudioPriority priority, bool interrupt) override {
        if (!initialized_ || muted_) return;

        std::lock_guard<std::mutex> lock(tts_mutex_);

        // Si es crítico e interrupt, limpiar cola
        if (interrupt && priority >= AudioPriority::HIGH) {
            std::queue<TTSMessage> empty;
            std::swap(tts_queue_, empty);
            espeak_Cancel();
        }

        // Verificar cooldown
        auto now = std::chrono::steady_clock::now();
        if (now - last_tts_ < tts_cooldown_) {
            return;  // Skip si está en cooldown
        }

        tts_queue_.push({text, priority});
        tts_cv_.notify_one();
    }

    void playBeep(AudioDirection dir, int freq, int duration_ms, float vol) override {
        if (!initialized_ || muted_) return;

        std::lock_guard<std::mutex> lock(beep_mutex_);
        beep_queue_.push({dir, freq, duration_ms, vol * master_volume_});
        beep_cv_.notify_one();
    }

    void playCriticalAlert(AudioDirection direction) override {
        // Beep urgente: alto, rápido, repetido
        playBeep(direction, 1000, 100, 1.0f);
        playBeep(direction, 1200, 100, 1.0f);
        playBeep(direction, 1000, 100, 1.0f);
    }

private:
    void ttsWorker() {
        while (running_) {
            TTSMessage msg;
            {
                std::unique_lock<std::mutex> lock(tts_mutex_);
                tts_cv_.wait(lock, [this] {
                    return !tts_queue_.empty() || !running_;
                });

                if (!running_) break;
                msg = tts_queue_.front();
                tts_queue_.pop();
            }

            // Sintetizar y reproducir
            espeak_Synth(
                msg.text.c_str(),
                msg.text.length() + 1,
                0,              // Position
                POS_CHARACTER,
                0,              // End position
                espeakCHARS_AUTO,
                nullptr,        // Unique identifier
                nullptr         // User data
            );
            espeak_Synchronize();

            last_tts_ = std::chrono::steady_clock::now();
        }
    }

    void beepWorker() {
        while (running_) {
            BeepRequest beep;
            {
                std::unique_lock<std::mutex> lock(beep_mutex_);
                beep_cv_.wait(lock, [this] {
                    return !beep_queue_.empty() || !running_;
                });

                if (!running_) break;
                beep = beep_queue_.front();
                beep_queue_.pop();
            }

            // Generar y reproducir beep
            playBeepInternal(beep);
        }
    }

    void playBeepInternal(const BeepRequest& beep) {
        const int sample_rate = 44100;
        const int num_samples = (sample_rate * beep.duration_ms) / 1000;

        // Buffer stereo: izquierda + derecha intercalados
        std::vector<int16_t> buffer(num_samples * 2);

        // Calcular panning según dirección
        float left_vol = 1.0f, right_vol = 1.0f;
        switch (beep.direction) {
            case AudioDirection::LEFT:
                left_vol = 1.0f;
                right_vol = 0.2f;
                break;
            case AudioDirection::RIGHT:
                left_vol = 0.2f;
                right_vol = 1.0f;
                break;
            case AudioDirection::CENTER:
                left_vol = right_vol = 0.8f;
                break;
            case AudioDirection::BEHIND:
                // Efecto "behind": volumen reducido, ambos canales
                left_vol = right_vol = 0.4f;
                break;
        }

        // Generar onda sinusoidal con envelope
        for (int i = 0; i < num_samples; i++) {
            float t = static_cast<float>(i) / sample_rate;
            float sample = std::sin(2.0f * M_PI * beep.frequency * t);

            // Envelope: attack-sustain-release
            float env = 1.0f;
            int attack = num_samples / 10;
            int release = num_samples / 5;
            if (i < attack) {
                env = static_cast<float>(i) / attack;
            } else if (i > num_samples - release) {
                env = static_cast<float>(num_samples - i) / release;
            }

            int16_t amplitude = static_cast<int16_t>(sample * env * beep.volume * 32767);

            buffer[i * 2] = static_cast<int16_t>(amplitude * left_vol);      // Left
            buffer[i * 2 + 1] = static_cast<int16_t>(amplitude * right_vol); // Right
        }

        // Enviar a PulseAudio
        int error;
        pa_simple_write(pulse_, buffer.data(), buffer.size() * sizeof(int16_t), &error);
    }

    struct TTSMessage {
        std::string text;
        AudioPriority priority;
    };

    struct BeepRequest {
        AudioDirection direction;
        int frequency;
        int duration_ms;
        float volume;
    };

    pa_simple* pulse_ = nullptr;
    bool initialized_ = false;
    bool running_ = false;
    bool muted_ = false;
    float master_volume_ = 0.7f;

    std::thread tts_thread_;
    std::thread beep_thread_;

    std::queue<TTSMessage> tts_queue_;
    std::mutex tts_mutex_;
    std::condition_variable tts_cv_;
    std::chrono::steady_clock::time_point last_tts_;
    std::chrono::milliseconds tts_cooldown_{800};

    std::queue<BeepRequest> beep_queue_;
    std::mutex beep_mutex_;
    std::condition_variable beep_cv_;
};
```

## CMake Configuration

```cmake
# CMakeLists.txt

# Encontrar dependencias de audio
find_package(PkgConfig REQUIRED)
pkg_check_modules(PULSE REQUIRED libpulse-simple)
pkg_check_modules(ESPEAK REQUIRED espeak-ng)

# Agregar adapter
add_library(audio_adapter
    src/adapters/audio/PulseAudioAdapter.cpp
)

target_include_directories(audio_adapter PRIVATE
    ${PULSE_INCLUDE_DIRS}
    ${ESPEAK_INCLUDE_DIRS}
)

target_link_libraries(audio_adapter
    ${PULSE_LIBRARIES}
    ${ESPEAK_LIBRARIES}
)

# Main executable
target_link_libraries(aria_slam
    audio_adapter
    # ... otras libs
)
```

## Sistema de Prioridades

| Prioridad | Uso | Cooldown | Puede interrumpir |
|-----------|-----|----------|-------------------|
| CRITICAL | Obstáculo inminente | 0ms | Sí, todo |
| HIGH | Objeto peligroso cerca | 500ms | Sí, MEDIUM/LOW |
| MEDIUM | Navegación normal | 800ms | No |
| LOW | Información contextual | 2000ms | No |

### Ejemplos de Mensajes

```cpp
// Crítico: obstáculo muy cerca
audio->speak("Stop!", AudioPriority::CRITICAL, true);
audio->playCriticalAlert(AudioDirection::CENTER);

// Alto: persona aproximándose
audio->speak("Person ahead, 2 meters", AudioPriority::HIGH);
audio->playBeep(AudioDirection::CENTER, 800, 200, 0.7f);

// Medio: navegación normal
audio->speak("Door on your right", AudioPriority::MEDIUM);
audio->playBeep(AudioDirection::RIGHT, 600, 150, 0.5f);

// Bajo: contexto
audio->speak("Entering kitchen area", AudioPriority::LOW);
```

## Integración con Detecciones

```cpp
// NavigationAudioEngine.hpp

class NavigationAudioEngine {
public:
    NavigationAudioEngine(IAudioFeedback* audio)
        : audio_(audio) {}

    void processDetections(
        const std::vector<Detection>& detections,
        const std::vector<float>& depths  // Opcional: profundidad estimada
    ) {
        for (size_t i = 0; i < detections.size(); i++) {
            const auto& det = detections[i];
            float depth = (i < depths.size()) ? depths[i] : 5.0f;

            // Determinar dirección basada en bounding box
            float center_x = (det.x1 + det.x2) / 2.0f;
            AudioDirection dir = getDirection(center_x, image_width_);

            // Determinar prioridad basada en distancia y clase
            AudioPriority prio = getPriority(det.class_id, depth);

            // Generar mensaje
            std::string msg = formatMessage(det, depth);

            // Verificar cooldown por clase
            if (canAnnounce(det.class_id)) {
                audio_->speak(msg, prio);

                if (depth < 1.5f) {
                    audio_->playBeep(dir, 800, 200, 0.8f);
                }

                updateCooldown(det.class_id);
            }
        }
    }

private:
    AudioDirection getDirection(float x, int width) {
        float normalized = x / width;
        if (normalized < 0.35f) return AudioDirection::LEFT;
        if (normalized > 0.65f) return AudioDirection::RIGHT;
        return AudioDirection::CENTER;
    }

    AudioPriority getPriority(int class_id, float depth) {
        // Clases peligrosas
        static const std::set<int> dangerous = {0, 1, 2, 3, 5, 7};  // person, bike, car, etc.

        if (depth < 1.0f) return AudioPriority::CRITICAL;
        if (depth < 2.0f && dangerous.count(class_id)) return AudioPriority::HIGH;
        if (depth < 3.0f) return AudioPriority::MEDIUM;
        return AudioPriority::LOW;
    }

    std::string formatMessage(const Detection& det, float depth) {
        std::ostringstream oss;
        oss << det.class_name;
        if (depth < 5.0f) {
            oss << ", " << std::fixed << std::setprecision(1) << depth << " meters";
        }
        return oss.str();
    }

    IAudioFeedback* audio_;
    int image_width_ = 640;
    std::map<int, std::chrono::steady_clock::time_point> class_cooldowns_;
};
```

## Testing

### MockAudioFeedback

```cpp
class MockAudioFeedback : public IAudioFeedback {
public:
    bool initialize() override { return true; }
    void shutdown() override {}

    void speak(const std::string& text, AudioPriority prio, bool) override {
        spoken_messages_.push_back({text, prio});
    }

    void playBeep(AudioDirection dir, int freq, int dur, float vol) override {
        played_beeps_.push_back({dir, freq, dur, vol});
    }

    void playCriticalAlert(AudioDirection dir) override {
        critical_alerts_.push_back(dir);
    }

    // Para verificación en tests
    std::vector<std::pair<std::string, AudioPriority>> spoken_messages_;
    std::vector<std::tuple<AudioDirection, int, int, float>> played_beeps_;
    std::vector<AudioDirection> critical_alerts_;
};

// Test
TEST(AudioFeedback, CriticalAlertInterrupts) {
    MockAudioFeedback audio;
    NavigationAudioEngine engine(&audio);

    // Simular detección muy cercana
    Detection det{100, 100, 200, 200, 0.9f, 0, "person"};
    engine.processDetections({det}, {0.5f});  // 0.5 metros

    EXPECT_EQ(audio.spoken_messages_.back().second, AudioPriority::CRITICAL);
}
```

## Checklist

- [ ] Instalar libpulse-dev y espeak-ng
- [ ] Implementar PulseAudioAdapter
- [ ] Implementar generador de beeps stereo
- [ ] Integrar TTS con espeak-ng
- [ ] Sistema de prioridades y cooldowns
- [ ] NavigationAudioEngine para procesar detecciones
- [ ] MockAudioFeedback para testing
- [ ] Tests de integración con SlamPipeline

## Alternativas

### Para macOS
```cpp
// Usar AVSpeechSynthesizer via Objective-C++
// O: flite (portable TTS)
```

### Para Windows
```cpp
// Usar SAPI (Speech API)
// #include <sapi.h>
```

### Cross-platform
```cpp
// Considerar SDL2_mixer + flite para máxima portabilidad
```

## Referencias

- [PulseAudio API](https://freedesktop.org/software/pulseaudio/doxygen/)
- [espeak-ng Documentation](https://github.com/espeak-ng/espeak-ng)
- [Audio Panning Theory](https://en.wikipedia.org/wiki/Panning_(audio))
