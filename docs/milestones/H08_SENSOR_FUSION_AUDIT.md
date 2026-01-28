# Auditoría Técnica: H08 - Visual-Inertial Sensor Fusion (EKF)

**Proyecto:** aria-slam (C++)
**Milestone:** H08 - Fusión IMU + Visual Odometry con EKF
**Fecha:** 2025-01-28
**Autor:** Roberto (preparación entrevista técnica)

---

## 0. RESUMEN EJECUTIVO

### Objetivo
Fusionar datos IMU con odometría visual usando un Extended Kalman Filter (EKF) para estimación de estado robusta con escala métrica.

### Por qué Sensor Fusion
| Problema | Solo Visual | Visual + IMU |
|----------|-------------|--------------|
| Escala métrica | Indeterminada | Conocida |
| Movimiento rápido | Motion blur | Robusto |
| Textura pobre | Falla | IMU prediction |
| Frecuencia | 20-30 Hz | 200 Hz |

### Arquitectura del EKF
```
                    ┌─────────────────────────────────────────┐
                    │           EKF State Vector               │
                    │                                          │
                    │  x = [p, v, θ, b_a, b_g]                │
                    │      ─┬─ ─┬─ ─┬─  ─┬─  ─┬─              │
                    │       │   │   │    │    │               │
                    │      pos vel rot accel gyro             │
                    │      (3) (3) (3) bias  bias             │
                    │                  (3)   (3)              │
                    │                                          │
                    │              Total: 15 estados           │
                    └─────────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            │                                                      │
            ▼                                                      ▼
┌─────────────────────┐                              ┌─────────────────────┐
│   PREDICT (IMU)     │                              │   UPDATE (Visual)   │
│                     │                              │                     │
│   200 Hz            │                              │   20-30 Hz          │
│                     │                              │                     │
│   x̂ = f(x, u_imu)   │                              │   K = PH'(HPH'+R)⁻¹│
│   P = FPF' + Q      │                              │   x = x + K(z - h(x))│
│                     │                              │   P = (I-KH)P       │
└─────────────────────┘                              └─────────────────────┘
```

---

## 1. ANÁLISIS DEL CÓDIGO REAL

### 1.1 IMU Preintegrator - Reset y Bias (`IMU.cpp:35-46`)

```cpp
// IMU.cpp:35-46
void IMUPreintegrator::reset() {
    delta_p_ = Eigen::Vector3d::Zero();
    delta_v_ = Eigen::Vector3d::Zero();
    delta_q_ = Eigen::Quaterniond::Identity();
    dt_sum_ = 0.0;
    last_timestamp_ = -1.0;
    covariance_.setZero();
}

void IMUPreintegrator::setBias(const IMUBias& bias) {
    bias_ = bias;
}
```

**Estado preintegrado:**
```
Δp = ∫∫ a dt²    (cambio de posición)
Δv = ∫ a dt       (cambio de velocidad)
Δq = ∏ exp(ω dt)  (cambio de rotación)
```

### 1.2 IMU Integration (`IMU.cpp:48-100`)

```cpp
// IMU.cpp:48-100
void IMUPreintegrator::integrate(const IMUMeasurement& measurement) {
    if (last_timestamp_ < 0) {
        last_timestamp_ = measurement.timestamp;
        return;
    }

    double dt = measurement.timestamp - last_timestamp_;
    if (dt <= 0 || dt > 0.5) {  // Guard: dt inválido
        last_timestamp_ = measurement.timestamp;
        return;
    }

    // Remove bias from measurements
    Eigen::Vector3d accel = measurement.accel - bias_.accel_bias;
    Eigen::Vector3d gyro = measurement.gyro - bias_.gyro_bias;

    // Rotation integration (mid-point)
    Eigen::Vector3d delta_angle = gyro * dt;
    double angle = delta_angle.norm();
    Eigen::Quaterniond dq;
    if (angle > 1e-10) {
        dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, delta_angle.normalized()));
    } else {
        dq = Eigen::Quaterniond::Identity();
    }

    // Rotate acceleration to initial frame
    Eigen::Vector3d accel_world = delta_q_ * accel;

    // Position and velocity integration
    delta_p_ += delta_v_ * dt + 0.5 * accel_world * dt * dt;
    delta_v_ += accel_world * dt;
    delta_q_ = delta_q_ * dq;
    delta_q_.normalize();

    // Covariance propagation
    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(3, 6) = -delta_q_.toRotationMatrix() * skew(accel) * dt;

    // ... noise matrices ...

    covariance_ = F * covariance_ * F.transpose() + G * Q * G.transpose();

    dt_sum_ += dt;
    last_timestamp_ = measurement.timestamp;
}
```

**Diagrama de integración:**
```
Mediciones IMU brutas:
  ω_meas = ω_true + b_g + n_g
  a_meas = a_true + b_a + n_a

Compensación de bias:
  ω = ω_meas - b_g
  a = a_meas - b_a

Integración de rotación:
  Δq_{k+1} = Δq_k ⊗ exp(ω * dt)
              │
              └── Quaternion multiplication

Aceleración en frame inicial:
  a_world = Δq * a

Integración cinemática:
  Δv_{k+1} = Δv_k + a_world * dt
  Δp_{k+1} = Δp_k + Δv_k * dt + 0.5 * a_world * dt²
```

### 1.3 Skew-Symmetric Matrix (`IMU.cpp:5-11`)

```cpp
// IMU.cpp:5-11
static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return m;
}
```

**Uso en cross product:**
```
         ┌              ┐
         │  0  -vz  vy  │
[v]× =   │  vz  0  -vx  │
         │ -vy  vx  0   │
         └              ┘

Propiedad: [v]× * u = v × u  (cross product como multiplicación de matrices)
```

### 1.4 SensorFusion EKF Constructor (`IMU.cpp:104-124`)

```cpp
// IMU.cpp:104-124
SensorFusion::SensorFusion() {
    // Initialize state covariance
    P_.setIdentity();
    P_.block<3, 3>(0, 0) *= 0.01;     // Position (m²)
    P_.block<3, 3>(3, 3) *= 0.01;     // Velocity (m/s)²
    P_.block<3, 3>(6, 6) *= 0.01;     // Orientation (rad²)
    P_.block<3, 3>(9, 9) *= 0.001;    // Accel bias
    P_.block<3, 3>(12, 12) *= 0.0001; // Gyro bias

    // Process noise (IMU noise)
    Q_.setZero();
    Q_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * accel_noise_ * accel_noise_;
    Q_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * gyro_noise_ * gyro_noise_;
    Q_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * accel_bias_walk_ * accel_bias_walk_;
    Q_.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * gyro_bias_walk_ * gyro_bias_walk_;

    // Measurement noise (visual)
    R_meas_.setZero();
    R_meas_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * pos_noise_ * pos_noise_;
    R_meas_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * rot_noise_ * rot_noise_;
}
```

**Estructura del estado (15D):**
```
        ┌─────────────────────────────────────────────────────┐
Estado: │ p_x p_y p_z │ v_x v_y v_z │ θ_x θ_y θ_z │ ba │ bg │
        └─────────────┴─────────────┴─────────────┴────┴────┘
         posición(3)   velocidad(3)   rotación(3)  bias bias
                                      (error rot)  (3)  (3)
```

### 1.5 EKF Predict Step (`IMU.cpp:139-222`)

```cpp
// IMU.cpp:139-222
void SensorFusion::predictEKF(const IMUMeasurement& imu) {
    double dt = imu.timestamp - last_imu_time_;
    if (dt <= 0 || dt > 0.1) {
        last_imu_time_ = imu.timestamp;
        return;
    }

    // Remove bias from measurements
    Eigen::Vector3d accel = imu.accel - bias_.accel_bias;
    Eigen::Vector3d gyro = imu.gyro - bias_.gyro_bias;

    // Get current rotation matrix
    Eigen::Matrix3d R = orientation_.toRotationMatrix();

    // ========== State Prediction ==========

    // Orientation prediction
    Eigen::Vector3d delta_angle = gyro * dt;
    double angle = delta_angle.norm();
    if (angle > 1e-10) {
        Eigen::Quaterniond dq(Eigen::AngleAxisd(angle, delta_angle.normalized()));
        orientation_ = orientation_ * dq;
        orientation_.normalize();
    }

    // Acceleration in world frame (remove gravity)
    Eigen::Vector3d accel_world = R * accel + gravity_;

    // Velocity and position prediction
    position_ += velocity_ * dt + 0.5 * accel_world * dt * dt;
    velocity_ += accel_world * dt;

    // ========== Covariance Prediction ==========

    // State transition Jacobian F (15x15)
    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Identity();

    // dp/dv
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;

    // dp/dtheta
    F.block<3, 3>(0, 6) = -0.5 * R * skew(accel) * dt * dt;

    // dp/dba
    F.block<3, 3>(0, 9) = -0.5 * R * dt * dt;

    // dv/dtheta
    F.block<3, 3>(3, 6) = -R * skew(accel) * dt;

    // dv/dba
    F.block<3, 3>(3, 9) = -R * dt;

    // dtheta/dbg
    F.block<3, 3>(6, 12) = -Eigen::Matrix3d::Identity() * dt;

    // Noise Jacobian G (15x12)
    // ... (similar structure) ...

    // Propagate covariance: P = F * P * F' + G * Q * G'
    P_ = F * P_ * F.transpose() + G * Q_ * G.transpose();

    // Ensure symmetry
    P_ = 0.5 * (P_ + P_.transpose());

    last_imu_time_ = imu.timestamp;
}
```

**Jacobiano de transición F:**
```
          p    v    θ   b_a  b_g
      ┌─────────────────────────────┐
   p  │  I   I*dt -½R[a]×dt²  -½Rdt²  0   │
   v  │  0    I    -R[a]×dt    -Rdt    0   │
F = θ  │  0    0      I          0    -Idt │
  b_a │  0    0      0          I      0   │
  b_g │  0    0      0          0      I   │
      └─────────────────────────────┘

[a]× = skew(accel) = matriz antisimétrica
```

### 1.6 EKF Update Step (`IMU.cpp:247-305`)

```cpp
// IMU.cpp:247-305
void SensorFusion::updateEKF(const Eigen::Matrix3d& R_meas, const Eigen::Vector3d& t_meas) {
    // ========== Measurement Model ==========
    // z = [position, orientation_error]

    // Measurement Jacobian H (6x15)
    Eigen::Matrix<double, 6, 15> H = Eigen::Matrix<double, 6, 15>::Zero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();  // Position
    H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();  // Orientation

    // ========== Innovation ==========

    // Position innovation
    Eigen::Vector3d pos_innov = t_meas - position_;

    // Orientation innovation
    Eigen::Quaterniond q_meas(R_meas);
    Eigen::Quaterniond q_err = q_meas * orientation_.inverse();
    q_err.normalize();

    // Convert to rotation vector
    Eigen::Vector3d rot_innov = logMap(q_err);

    // Combined innovation
    Eigen::Matrix<double, 6, 1> innovation;
    innovation << pos_innov, rot_innov;

    // ========== Kalman Gain ==========
    // K = P * H' * (H * P * H' + R)^-1

    Eigen::Matrix<double, 6, 6> S = H * P_ * H.transpose() + R_meas_;
    Eigen::Matrix<double, 15, 6> K = P_ * H.transpose() * S.inverse();

    // ========== State Update ==========
    Eigen::Matrix<double, 15, 1> dx = K * innovation;

    // Update position
    position_ += dx.segment<3>(0);

    // Update velocity
    velocity_ += dx.segment<3>(3);

    // Update orientation (apply error quaternion)
    Eigen::Quaterniond dq = expMap(dx.segment<3>(6));
    orientation_ = dq * orientation_;
    orientation_.normalize();

    // Update biases
    bias_.accel_bias += dx.segment<3>(9);
    bias_.gyro_bias += dx.segment<3>(12);

    // ========== Covariance Update (Joseph form) ==========
    Eigen::Matrix<double, 15, 15> I_KH =
        Eigen::Matrix<double, 15, 15>::Identity() - K * H;
    P_ = I_KH * P_ * I_KH.transpose() + K * R_meas_ * K.transpose();

    // Ensure symmetry
    P_ = 0.5 * (P_ + P_.transpose());
}
```

**Joseph form para estabilidad numérica:**
```
Standard:  P = (I - KH)P           ← Puede perder simetría/positividad
Joseph:    P = (I-KH)P(I-KH)' + KRK'  ← Siempre simétrica positiva definida
```

### 1.7 Exponential Map (`IMU.cpp:14-20`)

```cpp
// IMU.cpp:14-20
static Eigen::Quaterniond expMap(const Eigen::Vector3d& theta) {
    double angle = theta.norm();
    if (angle < 1e-10) {
        return Eigen::Quaterniond::Identity();
    }
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, theta.normalized()));
}
```

**Exponential map SO(3):**
```
exp: so(3) → SO(3)
     R³    → Quaternion

θ = [θx, θy, θz]  (rotation vector)
angle = ||θ||
axis = θ / angle

q = [cos(angle/2), sin(angle/2) * axis]
```

---

## 2. TEORÍA: EXTENDED KALMAN FILTER

### 2.1 Modelo de Estado

```
Estado: x = [p, v, θ, b_a, b_g]ᵀ ∈ R¹⁵

Dinámica (modelo de proceso):
  ṗ = v
  v̇ = R(q) * (a_meas - b_a - n_a) + g
  q̇ = ½ q ⊗ (ω_meas - b_g - n_g)
  ḃ_a = n_{ba}  (random walk)
  ḃ_g = n_{bg}  (random walk)
```

### 2.2 Ecuaciones del EKF

```
PREDICT (cada IMU sample @ 200 Hz):
┌──────────────────────────────────────────┐
│  x̂⁻ = f(x̂⁺, u)      ← State prediction │
│  P⁻ = F P⁺ Fᵀ + Q    ← Covariance pred  │
└──────────────────────────────────────────┘

UPDATE (cada visual pose @ 20-30 Hz):
┌──────────────────────────────────────────┐
│  y = z - h(x̂⁻)       ← Innovation       │
│  S = H P⁻ Hᵀ + R     ← Innovation cov   │
│  K = P⁻ Hᵀ S⁻¹       ← Kalman gain      │
│  x̂⁺ = x̂⁻ + K y      ← State update     │
│  P⁺ = (I - KH) P⁻    ← Covariance update│
└──────────────────────────────────────────┘
```

### 2.3 Modelo de Ruido IMU

```
Medición de acelerómetro:
a_meas = a_true + b_a + n_a
         donde:
         - b_a: bias (varía lentamente)
         - n_a: ruido blanco gaussiano

Modelo de bias (random walk):
ḃ_a = n_{ba}
donde n_{ba} ~ N(0, σ²_{ba})

Parámetros típicos (EuRoC):
┌────────────────────────────────────────┐
│ accel_noise = 2.0e-3 m/s²/√Hz          │
│ gyro_noise = 1.7e-4 rad/s/√Hz          │
│ accel_bias_walk = 3.0e-3 m/s³/√Hz      │
│ gyro_bias_walk = 1.9e-5 rad/s²/√Hz     │
└────────────────────────────────────────┘
```

### 2.4 Timeline de Fusión

```
Tiempo →

Visual:        [V₀]────────────────[V₁]────────────────[V₂]
                │                    │                    │
               t₀                   t₁                   t₂

IMU:          ││││││││││││││││││││││││││││││││││││││││││││
               └─┬─────────────────┘   └─┬─────────────────┘
                 │                       │
            Predict ×10              Predict ×10
                 │                       │
                 └── Update(V₁)          └── Update(V₂)

Entre cada visual update:
- ~10 IMU predictions
- Estado propagado a alta frecuencia
- Update fusiona con visual cuando disponible
```

---

## 3. CONCEPTOS C++ UTILIZADOS

### 3.1 Eigen Block Operations

```cpp
// Acceso a bloques de matrices
P_.block<3, 3>(0, 0) *= 0.01;  // Posición (filas 0-2, cols 0-2)
P_.block<3, 3>(3, 3) *= 0.01;  // Velocidad (filas 3-5, cols 3-5)

// Equivalente a:
P_.topLeftCorner<3, 3>() *= 0.01;  // Para esquina superior izquierda

// Segmento de vector
position_ += dx.segment<3>(0);  // Primeros 3 elementos
velocity_ += dx.segment<3>(3);  // Elementos 3-5
```

### 3.2 Quaternion Operations (Eigen)

```cpp
// Multiplicación de quaterniones
Eigen::Quaterniond q_result = q1 * q2;  // Hamilton convention

// Inverso (conjugado para quaternion unitario)
Eigen::Quaterniond q_inv = q.inverse();  // = q.conjugate() si ||q||=1

// Rotación de vector
Eigen::Vector3d v_rotated = q * v;  // = q.toRotationMatrix() * v

// Normalización
q.normalize();  // In-place
Eigen::Quaterniond q_norm = q.normalized();  // Copy
```

### 3.3 Fixed-Size Matrices

```cpp
// Mejor performance que dinámico cuando el tamaño es conocido
Eigen::Matrix<double, 15, 15> F;  // Stack allocation
Eigen::Matrix<double, 6, 15> H;   // Conocido en compile-time

// vs dinámico (heap allocation)
Eigen::MatrixXd F_dyn(15, 15);    // Heap, más overhead
```

### 3.4 Buffer con `std::deque`

```cpp
// IMU.hpp:92
std::deque<IMUMeasurement> imu_buffer_;

// IMU.cpp:126-132
void SensorFusion::addIMU(const IMUMeasurement& imu) {
    imu_buffer_.push_back(imu);

    // Limit buffer size
    while (imu_buffer_.size() > 1000) {
        imu_buffer_.pop_front();  // O(1) en deque
    }
}
```

**¿Por qué `deque` y no `vector`?**
- `push_back`: O(1) amortizado (igual que vector)
- `pop_front`: O(1) (vector sería O(n))
- Ideal para sliding window / buffer circular

---

## 4. DIAGRAMA DE SECUENCIA

```
main()          SensorFusion       IMUPreintegrator        Visual VO
  │                  │                    │                    │
  │ addIMU(imu)      │                    │                    │
  │─────────────────►│                    │                    │
  │                  │ predictEKF(imu)    │                    │
  │                  │───────────────────►│                    │
  │                  │                    │ integrate()        │
  │                  │                    │────────►          │
  │                  │◄───────────────────│                    │
  │                  │ P = FPF' + Q       │                    │
  │                  │                    │                    │
  │ (repite 200 Hz)  │                    │                    │
  │                  │                    │                    │
  │                  │                    │                    │
  │ addVisualPose(R,t)                    │                    │
  │─────────────────►│                    │                    │
  │                  │                    │                    │
  │                  │ updateEKF(R, t)    │                    │
  │                  │                    │                    │
  │                  │ y = z - h(x)       │                    │
  │                  │ K = PH'(HPH'+R)⁻¹  │                    │
  │                  │ x = x + Ky         │                    │
  │                  │ P = (I-KH)P        │                    │
  │                  │                    │                    │
  │◄─────────────────│ getPosition()      │                    │
  │                  │ getVelocity()      │                    │
  │                  │ getOrientation()   │                    │
  │                  │                    │                    │
```

---

## 5. PREGUNTAS DE ENTREVISTA

### Q1: ¿Por qué usar EKF en lugar de Kalman Filter lineal?

**R:** El modelo de proceso es **no lineal**:
```cpp
// Rotación: no lineal
q_{k+1} = q_k ⊗ exp(ω * dt)

// Aceleración en world frame: depende de rotación
a_world = R(q) * a_body
```

EKF linealiza alrededor del estado actual:
```
f(x) ≈ f(x̂) + F(x - x̂)  donde F = ∂f/∂x |_{x=x̂}
```

### Q2: ¿Qué es el bias del IMU y por qué se estima?

**R:** El bias es un offset sistemático que:
- Varía con temperatura y tiempo
- Es diferente para cada sensor
- Causa drift si no se compensa

```
Sin compensación de bias:
- Gyro bias 0.1°/s → 6°/min de drift
- Accel bias 0.01 m/s² → 1.8 km en 1 hora

EKF estima bias como parte del estado:
x = [p, v, θ, b_a, b_g]
           └────┴────┘
           bias estimados online
```

### Q3: ¿Por qué se usa Joseph form para actualizar la covarianza?

**R:**
```cpp
// Standard form (puede causar problemas numéricos)
P = (I - KH) * P;

// Joseph form (numéricamente estable)
P = (I - KH) * P * (I - KH)' + K * R * K';
```

**Problemas con standard form:**
1. Errores de redondeo pueden hacer P no simétrica
2. P puede perder positividad definida
3. EKF diverge si P tiene eigenvalues negativos

**Joseph form garantiza:**
- Simetría: `(I-KH)P(I-KH)' + KRK'` es simétrica por construcción
- Positividad: suma de matrices positivas semidefinidas

### Q4: ¿Cuál es la diferencia entre preintegración y integración directa?

**R:**
```
INTEGRACIÓN DIRECTA:
- Integra IMU en world frame
- Necesita pose actual para cada muestra
- Si pose cambia (loop closure), hay que re-integrar todo

PREINTEGRACIÓN:
- Integra en frame local del IMU
- Independiente de pose actual
- Resultado se transforma al recibir pose

Δp_preint = ∫∫ R_local * a dt²  (en frame local)
p_world = p₀ + R₀ * Δp_preint   (transformado después)
```

### Q5: ¿Cómo funciona el exponential map para rotaciones?

**R:** Convierte un vector de rotación (eje * ángulo) a quaternion:

```cpp
// rotation vector: θ = axis * angle
Eigen::Vector3d theta(0.1, 0.0, 0.0);  // 0.1 rad around X

// exponential map
double angle = theta.norm();  // = 0.1
Eigen::Vector3d axis = theta / angle;  // = [1, 0, 0]

Eigen::Quaterniond q = Eigen::Quaterniond(
    Eigen::AngleAxisd(angle, axis)
);
// q = [cos(0.05), sin(0.05), 0, 0]
```

**Propiedad:** `exp(θ₁) ⊗ exp(θ₂) ≈ exp(θ₁ + θ₂)` para θ pequeños

### Q6: ¿Por qué la medición visual actualiza posición y orientación pero no velocidad?

**R:** La odometría visual mide pose (R, t), no velocidad:
```cpp
// Jacobiano H (6x15)
H.block<3, 3>(0, 0) = I;  // z mide posición
H.block<3, 3>(3, 6) = I;  // z mide orientación
// columnas de velocidad (3-5) son cero
```

La velocidad se estima indirectamente:
1. IMU integra aceleración → predice velocidad
2. Visual corrige posición → velocidad se ajusta vía correlación en P

### Q7: ¿Cómo afecta la frecuencia de IMU a la precisión?

**R:**
```
Frecuencia alta (200 Hz):
+ Mejor captura de movimiento rápido
+ Menor error de discretización
+ Mejor predicción entre frames visuales

Frecuencia baja (50 Hz):
- Puede perder movimientos rápidos
- Mayor error en integración numérica
- Pero suficiente para movimiento lento

Trade-off:
- Más frecuencia = más cómputo
- Jetson: balance entre CPU load y precisión
```

---

## 6. PERFORMANCE

### 6.1 Complejidad Computacional

| Operación | Complejidad | Tiempo típico |
|-----------|-------------|---------------|
| IMU Predict | O(1) | ~10 μs |
| Visual Update | O(n³) para n=15 | ~100 μs |
| Jacobian computation | O(1) | ~5 μs |

### 6.2 Mejora con Fusion vs Visual-Only

| Métrica | Visual Only | Visual + IMU |
|---------|-------------|--------------|
| Scale error | Indeterminado | < 2% |
| Fast motion | Falla | Robusto |
| ATE (MH_01) | 0.45m | 0.08m |
| Max rotation rate | ~50°/s | ~500°/s |

---

## 7. CHECKLIST DE PREPARACIÓN

### Conceptos que debes dominar:
- [ ] Ecuaciones del EKF (predict + update)
- [ ] Modelo de ruido IMU (bias + white noise)
- [ ] Preintegración vs integración directa
- [ ] Exponential map para rotaciones
- [ ] Joseph form y estabilidad numérica
- [ ] Jacobiano de transición F

### Código que debes poder escribir:
```cpp
// EKF Predict
x_pred = f(x, u);
P_pred = F * P * F.transpose() + Q;

// EKF Update
y = z - h(x_pred);  // Innovation
S = H * P_pred * H.transpose() + R;
K = P_pred * H.transpose() * S.inverse();
x = x_pred + K * y;
P = (I - K * H) * P_pred;

// Quaternion from rotation vector
double angle = theta.norm();
Quaterniond q(AngleAxisd(angle, theta.normalized()));
```

### Números que debes conocer:
- Estado EKF: **15 dimensiones** (pos, vel, rot, biases)
- IMU rate: **200 Hz**
- Visual rate: **20-30 Hz**
- Gyro noise típico: **~0.017°/s/√Hz** (1e-4 rad/s/√Hz)
- Accel noise típico: **~0.002 m/s²/√Hz**

---

**Generado:** 2025-01-28
**Proyecto:** aria-slam
**Archivos analizados:** IMU.cpp, IMU.hpp
