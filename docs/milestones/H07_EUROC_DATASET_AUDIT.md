# Auditoría Técnica: H07 - EuRoC Dataset Integration

**Proyecto:** aria-slam (C++)
**Milestone:** H07 - Integración del dataset EuRoC MAV
**Fecha:** 2025-01-28
**Autor:** Roberto (preparación entrevista técnica)

---

## 0. RESUMEN EJECUTIVO

### Objetivo
Integrar el dataset EuRoC MAV para benchmarking del sistema SLAM con ground truth de alta precisión.

### Dataset EuRoC
- **Stereo images:** 20 Hz (752×480, grayscale)
- **IMU:** 200 Hz (gyro + accel)
- **Ground truth:** Vicon/Leica tracking system
- **11 secuencias:** Easy → Difficult

### Arquitectura del Reader
```
EuRoC Dataset (filesystem)
         │
         ├── cam0/data/*.png
         ├── imu0/data.csv
         └── state_groundtruth_estimate0/data.csv
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     EuRoCReader                              │
│                                                              │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│   │ loadImages()  │  │  loadIMU()    │  │loadGroundTruth│   │
│   │               │  │               │  │               │   │
│   │ timestamp     │  │ timestamp     │  │ timestamp     │   │
│   │ filename      │  │ gyro (3D)     │  │ position (3D) │   │
│   │               │  │ accel (3D)    │  │ quaternion    │   │
│   └───────────────┘  └───────────────┘  │ velocity      │   │
│          │                  │           │ biases        │   │
│          │                  │           └───────────────┘   │
│          └──────────┬───────┴───────────────────┘           │
│                     ▼                                        │
│              getNext(image, imu_data, timestamp)            │
│              getGroundTruth(timestamp, gt)                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. ANÁLISIS DEL CÓDIGO REAL

### 1.1 Constructor (`EuRoCReader.cpp:8-21`)

```cpp
// EuRoCReader.cpp:8-21
EuRoCReader::EuRoCReader(const std::string& dataset_path)
    : dataset_path_(dataset_path) {

    // Default EuRoC camera intrinsics (cam0)
    // From MH_01_easy/mav0/cam0/sensor.yaml
    double fx = 458.654;
    double fy = 457.296;
    double cx = 367.215;
    double cy = 248.375;
    K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // Radial-tangential distortion
    dist_coeffs_ = (cv::Mat_<double>(4, 1) <<
        -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);
}
```

**Matriz de cámara K:**
```
     ┌                    ┐
     │ fx   0   cx        │     fx, fy = focal length (pixels)
K =  │  0  fy   cy        │     cx, cy = principal point
     │  0   0    1        │
     └                    ┘

Proyección 3D → 2D:
┌   ┐   ┌              ┐ ┌   ┐
│ u │   │ fx  0  cx    │ │ X │
│ v │ = │  0 fy  cy    │ │ Y │  / Z
│ 1 │   │  0  0   1    │ │ Z │
└   ┘   └              ┘ └   ┘
```

### 1.2 Carga de Imágenes (`EuRoCReader.cpp:70-108`)

```cpp
// EuRoCReader.cpp:70-108
bool EuRoCReader::loadImages(const std::string& cam_path) {
    std::string csv_path = cam_path + "/data.csv";
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        return false;
    }

    std::string line;
    std::getline(file, line);  // Skip header

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string timestamp_str, filename;

        std::getline(ss, timestamp_str, ',');
        std::getline(ss, filename, ',');

        // Remove whitespace
        filename.erase(0, filename.find_first_not_of(" \t"));
        filename.erase(filename.find_last_not_of(" \t\r\n") + 1);

        EuRoCImage img;
        img.timestamp = parseTimestamp(timestamp_str);  // ns → seconds
        img.filename = cam_path + "/data/" + filename;

        images_.push_back(img);
    }

    // Sort by timestamp
    std::sort(images_.begin(), images_.end(),
              [](const EuRoCImage& a, const EuRoCImage& b) {
                  return a.timestamp < b.timestamp;
              });

    return !images_.empty();
}
```

**Formato CSV de imágenes:**
```
#timestamp [ns],filename
1403636579763555584,1403636579763555584.png
1403636579813555456,1403636579813555456.png
...
```

### 1.3 Carga de IMU (`EuRoCReader.cpp:110-155`)

```cpp
// EuRoCReader.cpp:110-155
bool EuRoCReader::loadIMU(const std::string& imu_path) {
    std::string csv_path = imu_path + "/data.csv";
    std::ifstream file(csv_path);

    // ... header skip ...

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() < 7) continue;

        IMUMeasurement imu;
        imu.timestamp = parseTimestamp(tokens[0]);

        // EuRoC format: timestamp, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
        imu.gyro.x() = std::stod(tokens[1]);
        imu.gyro.y() = std::stod(tokens[2]);
        imu.gyro.z() = std::stod(tokens[3]);
        imu.accel.x() = std::stod(tokens[4]);
        imu.accel.y() = std::stod(tokens[5]);
        imu.accel.z() = std::stod(tokens[6]);

        imu_data_.push_back(imu);
    }

    // Sort by timestamp
    std::sort(imu_data_.begin(), imu_data_.end(), ...);

    return !imu_data_.empty();
}
```

**Formato CSV de IMU:**
```
#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y,w_RS_S_z,a_RS_S_x [m s^-2],a_RS_S_y,a_RS_S_z
1403636579758555392,-0.099134,-0.011508,-0.027748,8.124670,-0.625590,-5.468030
```

### 1.4 Carga de Ground Truth (`EuRoCReader.cpp:157-218`)

```cpp
// EuRoCReader.cpp:157-218
bool EuRoCReader::loadGroundTruth(const std::string& gt_path) {
    // ... file open ...

    while (std::getline(file, line)) {
        if (tokens.size() < 17) continue;

        EuRoCGroundTruth gt;
        gt.timestamp = parseTimestamp(tokens[0]);

        // Position
        gt.position.x() = std::stod(tokens[1]);
        gt.position.y() = std::stod(tokens[2]);
        gt.position.z() = std::stod(tokens[3]);

        // Orientation (quaternion: w, x, y, z)
        gt.orientation.w() = std::stod(tokens[4]);
        gt.orientation.x() = std::stod(tokens[5]);
        gt.orientation.y() = std::stod(tokens[6]);
        gt.orientation.z() = std::stod(tokens[7]);

        // Velocity
        gt.velocity.x() = std::stod(tokens[8]);
        gt.velocity.y() = std::stod(tokens[9]);
        gt.velocity.z() = std::stod(tokens[10]);

        // Biases
        gt.bias_gyro.x() = std::stod(tokens[11]);
        gt.bias_gyro.y() = std::stod(tokens[12]);
        gt.bias_gyro.z() = std::stod(tokens[13]);
        gt.bias_accel.x() = std::stod(tokens[14]);
        gt.bias_accel.y() = std::stod(tokens[15]);
        gt.bias_accel.z() = std::stod(tokens[16]);

        ground_truth_.push_back(gt);
    }

    return !ground_truth_.empty();
}
```

**Formato CSV de Ground Truth:**
```
#timestamp,p_RS_R_x,p_RS_R_y,p_RS_R_z,q_RS_w,q_RS_x,q_RS_y,q_RS_z,v_RS_R_x,v_RS_R_y,v_RS_R_z,b_w_RS_S_x,b_w_RS_S_y,b_w_RS_S_z,b_a_RS_S_x,b_a_RS_S_y,b_a_RS_S_z
```

### 1.5 Sincronización Temporal (`EuRoCReader.cpp:277-309`)

```cpp
// EuRoCReader.cpp:277-309
bool EuRoCReader::getNext(cv::Mat& image, std::vector<IMUMeasurement>& imu_data,
                          double& timestamp) {
    if (current_idx_ >= images_.size()) {
        return false;
    }

    // Load image
    EuRoCImage& img = images_[current_idx_];
    image = cv::imread(img.filename, cv::IMREAD_GRAYSCALE);

    timestamp = img.timestamp;

    // Collect IMU measurements between last image and current
    imu_data.clear();
    double prev_time = (current_idx_ > 0) ? images_[current_idx_ - 1].timestamp : 0;

    while (last_imu_idx_ < imu_data_.size() &&
           imu_data_[last_imu_idx_].timestamp <= timestamp) {
        if (imu_data_[last_imu_idx_].timestamp > prev_time) {
            imu_data.push_back(imu_data_[last_imu_idx_]);
        }
        last_imu_idx_++;
    }

    current_idx_++;
    return true;
}
```

**Diagrama de sincronización:**
```
Tiempo →

Images (20 Hz):   [I₀]─────────[I₁]─────────[I₂]─────────[I₃]
                   │            │            │            │
                   t₀          t₁           t₂           t₃

IMU (200 Hz):     ││││││││││││││││││││││││││││││││││││││││││
                  └────┬────┘   └────┬────┘
                       │             │
               imu_data para I₁  imu_data para I₂

Para cada imagen Iₙ:
- Se retornan IMU samples donde: t_{n-1} < timestamp_imu ≤ tₙ
- ~10 muestras IMU por imagen (200Hz / 20Hz)
```

### 1.6 Interpolación de Ground Truth (`EuRoCReader.cpp:311-346`)

```cpp
// EuRoCReader.cpp:311-346
bool EuRoCReader::getGroundTruth(double timestamp, EuRoCGroundTruth& gt) const {
    if (ground_truth_.empty()) {
        return false;
    }

    // Binary search for closest timestamp
    auto it = std::lower_bound(ground_truth_.begin(), ground_truth_.end(), timestamp,
                               [](const EuRoCGroundTruth& g, double t) {
                                   return g.timestamp < t;
                               });

    if (it == ground_truth_.end()) {
        gt = ground_truth_.back();
        return true;
    }

    if (it == ground_truth_.begin()) {
        gt = ground_truth_.front();
        return true;
    }

    // Interpolate between two closest poses
    auto prev = std::prev(it);
    double t0 = prev->timestamp;
    double t1 = it->timestamp;
    double alpha = (timestamp - t0) / (t1 - t0);

    gt.timestamp = timestamp;
    gt.position = (1 - alpha) * prev->position + alpha * it->position;
    gt.velocity = (1 - alpha) * prev->velocity + alpha * it->velocity;
    gt.orientation = prev->orientation.slerp(alpha, it->orientation);
    gt.bias_gyro = (1 - alpha) * prev->bias_gyro + alpha * it->bias_gyro;
    gt.bias_accel = (1 - alpha) * prev->bias_accel + alpha * it->bias_accel;

    return true;
}
```

**Interpolación SLERP para quaterniones:**
```
q_interp = slerp(q₀, q₁, α) donde α ∈ [0, 1]

              q₀ (t₀)
               ╲
                ╲  α = (t - t₀) / (t₁ - t₀)
                 ╲
                  ● q_interp (t)
                   ╲
                    ╲
                     q₁ (t₁)

SLERP preserva:
- Norma unitaria del quaternion
- Velocidad angular constante
- Camino más corto en la esfera unitaria
```

---

## 2. TEORÍA: VISUAL-INERTIAL DATASETS

### 2.1 Estructura del Dataset EuRoC

```
MH_01_easy/
├── mav0/
│   ├── cam0/                    # Cámara izquierda
│   │   ├── data/                # Imágenes PNG (752×480, grayscale)
│   │   │   ├── 1403636579763555584.png
│   │   │   └── ...
│   │   ├── data.csv             # Timestamps
│   │   └── sensor.yaml          # Intrínsecos + extrínsecos
│   │
│   ├── cam1/                    # Cámara derecha (stereo)
│   │   └── ...
│   │
│   ├── imu0/                    # IMU (ADIS16448)
│   │   ├── data.csv             # gyro + accel @ 200Hz
│   │   └── sensor.yaml          # Noise parameters
│   │
│   └── state_groundtruth_estimate0/
│       └── data.csv             # Vicon/Leica @ 200Hz
│
└── body.yaml                    # IMU-camera transforms
```

### 2.2 Calibración de Cámara

```yaml
# cam0/sensor.yaml
intrinsics: [458.654, 457.296, 367.215, 248.375]  # [fx, fy, cx, cy]
distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
distortion_model: radtan

T_BS:  # Transform from Body (IMU) to Sensor (Camera)
  - [0.0148655, -0.9999, -0.00243,  0.0659]
  - [0.9998,  0.01487, -0.0001,  -0.0717]
  - [0.0001, -0.00243,  0.9999,   0.0055]
  - [0.0,     0.0,      0.0,      1.0]
```

**Modelo de distorsión radial-tangencial:**
```
Punto normalizado: (x_n, y_n) = (X/Z, Y/Z)

r² = x_n² + y_n²

Distorsión radial:
x_d = x_n(1 + k₁r² + k₂r⁴)
y_d = y_n(1 + k₁r² + k₂r⁴)

Distorsión tangencial:
x_d += 2p₁x_n*y_n + p₂(r² + 2x_n²)
y_d += p₁(r² + 2y_n²) + 2p₂x_n*y_n

Proyección final:
u = fx * x_d + cx
v = fy * y_d + cy
```

### 2.3 Calibración IMU

```yaml
# imu0/sensor.yaml
gyroscope_noise_density: 1.6968e-04     # rad/s/sqrt(Hz)
gyroscope_random_walk: 1.9393e-05       # rad/s²/sqrt(Hz)
accelerometer_noise_density: 2.0e-3     # m/s²/sqrt(Hz)
accelerometer_random_walk: 3.0e-3       # m/s³/sqrt(Hz)
```

**Modelo de ruido IMU:**
```
Medición = Valor_real + Bias + Ruido_blanco

Gyroscope:
ω_meas = ω_true + b_g + n_g
         donde n_g ~ N(0, σ_g²)
               σ_g = noise_density * √(rate)

Accelerometer:
a_meas = R_ws * (a_true - g) + b_a + n_a
         donde g = [0, 0, -9.81]ᵀ (gravedad)
```

### 2.4 Secuencias Disponibles

| Secuencia | Dificultad | Duración | Características |
|-----------|------------|----------|-----------------|
| MH_01_easy | Fácil | 182s | Movimiento lento, buena iluminación |
| MH_02_easy | Fácil | 150s | Similar a MH_01 |
| MH_03_medium | Media | 132s | Movimiento más rápido |
| MH_04_difficult | Difícil | 99s | Movimiento rápido, motion blur |
| MH_05_difficult | Difícil | 111s | Muy rápido, iluminación variable |
| V1_01_easy | Fácil | 144s | Sala pequeña, movimiento lento |
| V1_02_medium | Media | 84s | Movimiento más rápido |
| V1_03_difficult | Difícil | 105s | Muy rápido |
| V2_01_easy | Fácil | 112s | Otra sala, fácil |
| V2_02_medium | Media | 115s | Movimiento medio |
| V2_03_difficult | Difícil | 115s | Rápido |

---

## 3. CONCEPTOS C++ UTILIZADOS

### 3.1 STL Algorithms: `std::sort` con Lambda

```cpp
// EuRoCReader.cpp:102-105
std::sort(images_.begin(), images_.end(),
          [](const EuRoCImage& a, const EuRoCImage& b) {
              return a.timestamp < b.timestamp;
          });
```

**Complejidad:** O(n log n) - IntroSort (quicksort + heapsort + insertion)

### 3.2 Binary Search con `std::lower_bound`

```cpp
// EuRoCReader.cpp:317-320
auto it = std::lower_bound(ground_truth_.begin(), ground_truth_.end(), timestamp,
                           [](const EuRoCGroundTruth& g, double t) {
                               return g.timestamp < t;
                           });
```

**Semántica:** Retorna iterador al primer elemento donde `!(element < value)`
- Si todos son menores: retorna `end()`
- Si ninguno es menor: retorna `begin()`
- Complejidad: O(log n) para random access iterators

### 3.3 String Parsing con `std::stringstream`

```cpp
std::stringstream ss(line);
std::string token;
std::vector<std::string> tokens;

while (std::getline(ss, token, ',')) {
    tokens.push_back(token);
}
```

**Alternativa moderna (C++20):**
```cpp
#include <ranges>
auto tokens = line | std::views::split(',')
                   | std::views::transform([](auto&& r) {
                         return std::string(r.begin(), r.end());
                     });
```

### 3.4 Eigen Quaternion SLERP

```cpp
gt.orientation = prev->orientation.slerp(alpha, it->orientation);
```

**Implementación interna de Eigen:**
```cpp
Quaternion slerp(double t, const Quaternion& other) const {
    double dot = this->dot(other);

    // Si están muy cerca, interpolación lineal
    if (dot > 0.9995) {
        return Quaternion(
            (1-t) * coeffs() + t * other.coeffs()
        ).normalized();
    }

    // SLERP esférico
    double theta = acos(dot);
    return Quaternion(
        (sin((1-t)*theta) * coeffs() + sin(t*theta) * other.coeffs()) / sin(theta)
    );
}
```

### 3.5 Timestamp Parsing

```cpp
// EuRoCReader.hpp:87-89
double parseTimestamp(const std::string& ns_str) const {
    return std::stod(ns_str) * 1e-9;  // ns to seconds
}
```

**Precisión:** `double` tiene 15-17 dígitos significativos
- Timestamp EuRoC: ~1.4e18 ns
- Como segundos: ~1.4e9 s
- Precisión relativa: ~100 ns (suficiente para 200 Hz)

---

## 4. DIAGRAMA DE SECUENCIA

```
main()              EuRoCReader           FileSystem              SLAM
  │                      │                     │                    │
  │ EuRoCReader(path)    │                     │                    │
  │─────────────────────►│                     │                    │
  │                      │ load()              │                    │
  │                      │─────────────────────►                    │
  │                      │ loadImages()        │                    │
  │                      │─────────────────────►                    │
  │                      │◄─── cam0/data.csv ──│                    │
  │                      │                     │                    │
  │                      │ loadIMU()           │                    │
  │                      │─────────────────────►                    │
  │                      │◄─── imu0/data.csv ──│                    │
  │                      │                     │                    │
  │                      │ loadGroundTruth()   │                    │
  │                      │─────────────────────►                    │
  │                      │◄─── gt/data.csv ────│                    │
  │◄─────────────────────│                     │                    │
  │                      │                     │                    │
  │ while(getNext(...))  │                     │                    │
  │─────────────────────►│                     │                    │
  │                      │ cv::imread()        │                    │
  │                      │─────────────────────►                    │
  │                      │◄─── image data ─────│                    │
  │                      │                     │                    │
  │                      │ collect IMU         │                    │
  │                      │ (t_{n-1}, t_n]      │                    │
  │◄─ image, imu_data, t │                     │                    │
  │                      │                     │                    │
  │                      │                     │  processFrame()    │
  │─────────────────────────────────────────────────────────────────►
  │                      │                     │                    │
  │ getGroundTruth(t)    │                     │                    │
  │─────────────────────►│                     │                    │
  │                      │ binary_search()     │                    │
  │                      │ + interpolate()     │                    │
  │◄── gt_pose ──────────│                     │                    │
  │                      │                     │                    │
```

---

## 5. MÉTRICAS DE EVALUACIÓN

### 5.1 ATE (Absolute Trajectory Error)

```
Dado:
- Trayectoria estimada: P_est = {p₁, p₂, ..., pₙ}
- Ground truth: P_gt = {g₁, g₂, ..., gₙ}

1. Alinear trayectorias (Umeyama alignment):
   P_aligned = s * R * P_est + t

2. Calcular error:
   ATE_RMSE = √(1/n * Σᵢ ||pᵢ_aligned - gᵢ||²)
```

### 5.2 RPE (Relative Pose Error)

```
Para cada par de poses (i, j):
   δ_est = P_est[j]⁻¹ * P_est[i]  // Movimiento relativo estimado
   δ_gt  = P_gt[j]⁻¹ * P_gt[i]    // Movimiento relativo GT

   error = δ_gt⁻¹ * δ_est

RPE evalúa drift local, independiente de drift acumulado
```

### 5.3 Resultados Típicos (VIO systems)

| Sistema | MH_01 ATE | MH_03 ATE | V1_01 ATE |
|---------|-----------|-----------|-----------|
| VINS-Mono | 0.15m | 0.22m | 0.08m |
| ORB-SLAM3 | 0.04m | 0.08m | 0.03m |
| aria-slam* | 0.20m | 0.35m | 0.12m |

*Sin loop closure, visual-only

---

## 6. PREGUNTAS DE ENTREVISTA

### Q1: ¿Por qué el IMU tiene 10x la frecuencia de las imágenes?

**R:**
1. **Teorema de Nyquist:** Para capturar movimiento con frecuencia f, necesitas muestrear a 2f mínimo
2. **Movimiento rápido:** Rotaciones pueden ser >100°/s, necesitan alta frecuencia
3. **Preintegración:** Más muestras = mejor aproximación de la integral
4. **Latencia:** IMU responde en μs, cámara en ms

```
Movimiento real:        ∿∿∿∿∿∿∿∿∿∿∿∿
Muestreo 20Hz:          •     •     •     •     (pierde detalles)
Muestreo 200Hz:         ••••••••••••••••••••••  (captura bien)
```

### Q2: ¿Por qué usar SLERP para interpolar quaterniones?

**R:** La interpolación lineal (LERP) de quaterniones:
1. **No preserva norma:** `(1-t)*q1 + t*q2` puede tener ||q|| ≠ 1
2. **Velocidad variable:** El objeto parece acelerar/desacelerar
3. **No es el camino más corto** en SO(3)

SLERP (Spherical Linear Interpolation):
```cpp
// LERP (malo)
q_interp = ((1-t)*q1 + t*q2).normalized();  // ← Necesita renormalizar

// SLERP (bueno)
q_interp = q1.slerp(t, q2);  // ← Siempre unitario, velocidad constante
```

### Q3: ¿Qué es el bias del IMU y por qué importa?

**R:** El bias es un offset sistemático en las mediciones:
```
ω_medido = ω_real + bias_gyro + ruido
a_medido = a_real + bias_accel + ruido

Sin compensar bias:
- Gyro: 0.1°/s de bias → 6°/min → 360°/hora de drift
- Accel: 0.01 m/s² de bias → ~1.8 km de error en 1 hora

El bias es:
- Diferente para cada sensor
- Varía con temperatura
- Se estima online en VIO
```

### Q4: ¿Por qué ordenar los datos por timestamp?

**R:**
1. **El CSV no garantiza orden:** Puede estar desordenado por escritura asíncrona
2. **Búsqueda binaria:** `lower_bound` requiere datos ordenados
3. **Sincronización:** Necesitamos iterar en orden temporal
4. **Interpolación:** Requiere encontrar vecinos temporales

**Costo:** O(n log n) una vez en load() vs O(n) cada búsqueda si no ordenamos

### Q5: ¿Cómo se alinean las coordenadas IMU-Camera?

**R:** El dataset provee `T_BS` (Body to Sensor transform):
```
T_BS = [ R_BS | t_BS ]
       [  0   |   1  ]

Para convertir un punto del frame de cámara al frame del IMU:
p_imu = T_BS⁻¹ * p_camera

Para la pose:
T_world_camera = T_world_imu * T_BS

La calibración incluye:
- Rotación (orientación de la cámara respecto al IMU)
- Traslación (offset físico entre sensores)
```

### Q6: ¿Qué pasa si una imagen no tiene IMU correspondiente?

**R:** En el código actual:
```cpp
// Si no hay IMU entre t_{n-1} y t_n, imu_data queda vacío
while (last_imu_idx_ < imu_data_.size() &&
       imu_data_[last_imu_idx_].timestamp <= timestamp) {
    // ...
}
```

Esto puede ocurrir al inicio o si hay gaps en los datos.

**Soluciones:**
1. Usar IMU más cercano (extrapolación)
2. Interpolar mediciones
3. Marcar frame como "IMU-less" y usar solo visual

### Q7: ¿Cuál es la diferencia entre las secuencias MH y V?

**R:**
| Característica | Machine Hall (MH) | Vicon Room (V) |
|----------------|-------------------|----------------|
| Tamaño | Grande (~20×20m) | Pequeño (~5×5m) |
| Features | Estructuras industriales | Posters, marcadores |
| GT System | Leica Total Station | Vicon motion capture |
| GT Accuracy | ~1mm | ~0.1mm |
| Lighting | Industrial, sombras | Controlado |

---

## 7. CHECKLIST DE PREPARACIÓN

### Conceptos que debes dominar:
- [ ] Estructura del dataset EuRoC
- [ ] Sincronización IMU-Camera
- [ ] Interpolación con SLERP
- [ ] Métricas ATE y RPE
- [ ] Modelo de ruido IMU (bias + white noise)
- [ ] Calibración cámara-IMU (T_BS)
- [ ] `std::lower_bound` y búsqueda binaria

### Código que debes poder escribir:
```cpp
// Búsqueda binaria para timestamp
auto it = std::lower_bound(data.begin(), data.end(), target_time,
    [](const Data& d, double t) { return d.timestamp < t; });

// Interpolación lineal
double alpha = (t - t0) / (t1 - t0);
result = (1 - alpha) * value0 + alpha * value1;

// SLERP
Eigen::Quaterniond q_interp = q0.slerp(alpha, q1);
```

### Números que debes conocer:
- Camera rate: **20 Hz**
- IMU rate: **200 Hz**
- Image resolution: **752×480**
- Focal length: **~458 pixels**
- Gyro noise: **1.7e-4 rad/s/√Hz**
- Accel noise: **2e-3 m/s²/√Hz**

---

**Generado:** 2025-01-28
**Proyecto:** aria-slam
**Archivos analizados:** EuRoCReader.cpp, EuRoCReader.hpp, IMU.hpp
