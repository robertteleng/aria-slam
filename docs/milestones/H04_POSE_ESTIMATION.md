# Auditoría Técnica: H04 - Pose Estimation (Essential Matrix + RANSAC)

**Proyecto:** aria-slam (C++)
**Milestone:** H04 - Estimación de pose con geometría epipolar
**Fecha:** 2025-01
**Autor:** Roberto (preparación entrevista técnica)

---

## 0. RESUMEN EJECUTIVO

### Objetivo
Calcular el movimiento de la cámara (rotación R y traslación t) entre frames consecutivos usando correspondencias de features.

### Resultado
- **Essential Matrix** con RANSAC para robustez
- **Pose recovery** (R, t) con hasta 4 soluciones, selección automática
- **Acumulación de trayectoria** en coordenadas world

---

## 1. ANÁLISIS DEL CÓDIGO REAL

### 1.1 Intrínsecos de Cámara (`main.cpp:109-113`)

```cpp
// Intrinsic camera matrix (approximate values for test video)
double fx = 700, fy = 700;
double cx = 640 / 2.0;
double cy = 360 / 2.0;
cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
```

**Matriz K (Camera Intrinsics):**
```
K = | fx   0  cx |
    |  0  fy  cy |
    |  0   0   1 |

fx, fy = focal lengths (pixels)
cx, cy = principal point (center of image)
```

### 1.2 Estado de Pose (`main.cpp:115-117`)

```cpp
// Camera pose (accumulated from frame-to-frame motion)
cv::Mat position = cv::Mat::zeros(3, 1, CV_64F);
cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
```

**Estado inicial:**
- Posición: origen (0, 0, 0)
- Rotación: identidad (sin rotación)

### 1.3 Pose Estimation (`main.cpp:175-198`)

```cpp
// Pose estimation from matched features (requires >= 8 points for Essential Matrix)
if (good_matches.size() >= 8) {
    std::vector<cv::Point2f> pts1, pts2;
    for (auto& m : good_matches) {
        pts1.push_back(prev_frame->keypoints[m.queryIdx].pt);
        pts2.push_back(current_frame.keypoints[m.trainIdx].pt);
    }

    // Compute Essential Matrix with RANSAC for outlier rejection
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC);

    // Decompose E into rotation and translation
    cv::Mat R, t;
    cv::recoverPose(E, pts1, pts2, K, R, t);

    // Accumulate pose in world coordinates
    position = position + rotation * t;
    rotation = R * rotation;

    // Draw position on trajectory map (scaled and centered)
    int x = (int)(position.at<double>(0) * 100) + 300;
    int y = (int)(position.at<double>(2) * 100) + 300;
    cv::circle(trajectory, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
}
```

**Análisis paso a paso:**

| Paso | Línea | Operación | Output |
|------|-------|-----------|--------|
| 1 | 177-180 | Extraer coordenadas 2D | pts1, pts2 |
| 2 | 183 | Essential Matrix + RANSAC | E (3×3) |
| 3 | 186-187 | Descomponer E | R (3×3), t (3×1) |
| 4 | 190-191 | Acumular en world frame | position, rotation |

---

## 2. TEORÍA DE GEOMETRÍA EPIPOLAR

### 2.1 Essential Matrix

La Essential Matrix E relaciona puntos correspondientes entre dos vistas:

```
x₂ᵀ · E · x₁ = 0

Donde:
- x₁ = punto normalizado en imagen 1
- x₂ = punto correspondiente en imagen 2
- E = Essential Matrix (3×3, rank 2)
```

**Relación con Fundamental Matrix:**
```
E = Kᵀ · F · K

F = Fundamental Matrix (trabaja con píxeles)
E = Essential Matrix (trabaja con coordenadas normalizadas)
```

### 2.2 Descomposición de E

```
E = [t]ₓ · R

Donde [t]ₓ es la matriz skew-symmetric de t:
       |  0  -t₃  t₂ |
[t]ₓ = |  t₃  0  -t₁ |
       | -t₂  t₁  0  |
```

**4 posibles soluciones:**
```
(R₁, t)   (R₁, -t)   (R₂, t)   (R₂, -t)

cv::recoverPose() selecciona la correcta:
- Puntos deben tener depth positivo en ambas cámaras
- Triangula puntos y verifica z > 0
```

### 2.3 RANSAC para Essential Matrix

```
Parámetros por defecto de cv::findEssentialMat():

method = cv::RANSAC
prob = 0.999      // Probabilidad de encontrar modelo correcto
threshold = 1.0   // Error máximo en píxeles

Iteraciones = log(1-prob) / log(1 - (inlier_ratio)^n)
            = log(0.001) / log(1 - 0.5^5)
            ≈ 220 iteraciones para 50% inliers, 5 puntos
```

---

## 3. ACUMULACIÓN DE POSE

### 3.1 Transformación Frame-to-Frame → World

```cpp
// Movimiento relativo: frame N-1 → frame N
cv::Mat R, t;  // R rota de N-1 a N, t traslada en frame N-1

// Acumulación en world frame
position = position + rotation * t;
rotation = R * rotation;
```

**Derivación:**
```
Sea:
- P_world = posición en world frame
- R_world = rotación acumulada (world → camera actual)

Nuevo frame:
- R_rel = rotación relativa (camera N-1 → camera N)
- t_rel = traslación en frame de camera N-1

Actualización:
P_world_new = P_world + R_world * t_rel
R_world_new = R_rel * R_world
```

### 3.2 Ambigüedad de Escala

**Problema:** Visual odometry monocular no puede determinar escala absoluta.

```
Si (R, t) es solución, también lo es (R, λ·t) para cualquier λ > 0
```

**Soluciones:**
1. **IMU** (H08): Acelerómetro da escala métrica
2. **Objeto conocido**: Tamaño de landmark
3. **Stereo**: Baseline conocido

### 3.3 Visualización de Trayectoria

```cpp
// Proyección 2D (vista top-down)
int x = (int)(position.at<double>(0) * 100) + 300;  // X en world
int y = (int)(position.at<double>(2) * 100) + 300;  // Z en world (profundidad)
cv::circle(trajectory, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
```

**Mapeo:**
- Canvas 600×600 píxeles
- Escala: 100 px = 1 unidad
- Centro: (300, 300)
- X: horizontal, Z: vertical (forward)

---

## 4. FLUJO DE DATOS

```
┌─────────────────────────────────────────────────────────────────┐
│   good_matches (200-500 cv::DMatch)                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│   Extraer coordenadas 2D                                        │
│                                                                  │
│   pts1 = keypoints[match.queryIdx].pt  (prev frame)             │
│   pts2 = keypoints[match.trainIdx].pt  (current frame)          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│   cv::findEssentialMat(pts1, pts2, K, RANSAC)                   │
│                                                                  │
│   - Normaliza puntos: x_norm = K⁻¹ · [u, v, 1]ᵀ                 │
│   - RANSAC: 5-point algorithm × N iteraciones                   │
│   - Retorna E (3×3) + inlier mask                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│   cv::recoverPose(E, pts1, pts2, K, R, t)                       │
│                                                                  │
│   - SVD de E → 4 posibles (R, t)                                │
│   - Triangula puntos con cada solución                          │
│   - Selecciona la que tiene más puntos con z > 0                │
│   - Retorna R (3×3), t (3×1), inlier_count                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│   Acumulación de pose                                           │
│                                                                  │
│   position = position + rotation * t                            │
│   rotation = R * rotation                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. CONCEPTOS C++ UTILIZADOS

### 5.1 cv::Mat Initialization

```cpp
// Matriz de ceros
cv::Mat position = cv::Mat::zeros(3, 1, CV_64F);

// Matriz identidad
cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);

// Matriz desde valores literales
cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
```

### 5.2 Acceso a elementos

```cpp
// Acceso tipado (más seguro)
double x = position.at<double>(0);
double y = position.at<double>(1);
double z = position.at<double>(2);

// Acceso por puntero (más rápido, menos seguro)
double* ptr = position.ptr<double>(0);
double x = ptr[0];
```

### 5.3 Operaciones matriciales

```cpp
// OpenCV sobrecarga operadores para cv::Mat
position = position + rotation * t;  // Multiplicación matriz-vector
rotation = R * rotation;             // Multiplicación matriz-matriz
```

---

## 6. PREGUNTAS DE ENTREVISTA

### Q1: ¿Por qué se necesitan mínimo 8 puntos para Essential Matrix?

**R:**
Essential Matrix tiene 9 elementos pero:
- Rank 2 (determinante = 0)
- Escala arbitraria
- → 5 DOF (grados de libertad)

El **5-point algorithm** es el mínimo teórico, pero **8-point algorithm** es más estable numéricamente y se usa con RANSAC.

### Q2: ¿Cuál es la diferencia entre Essential y Fundamental Matrix?

**R:**
| Aspecto | Essential (E) | Fundamental (F) |
|---------|---------------|-----------------|
| Coordenadas | Normalizadas | Píxeles |
| Requiere K | Sí | No |
| DOF | 5 | 7 |
| Constraints | Rank 2, 2 singular values iguales | Rank 2 |

```
E = Kᵀ · F · K
```

### Q3: ¿Qué pasa si la cámara solo rota (sin traslación)?

**R:**
- E = [t]ₓ · R
- Si t = 0, entonces E = 0
- No hay restricción epipolar
- **findEssentialMat falla o da resultados incorrectos**

**Solución:** Detectar rotación pura (poco parallax) y usar homografía en su lugar.

### Q4: ¿Por qué RANSAC y no least squares directo?

**R:**
- Matches tienen outliers (~20-40%)
- Least squares minimiza error de **todos** los puntos
- Un outlier puede sesgar completamente el resultado
- RANSAC encuentra el modelo que maximiza **inliers**

### Q5: ¿Cómo sabes si la pose estimada es correcta?

**R:**
Indicadores de calidad:
1. **Número de inliers**: `recoverPose` retorna count > 0
2. **Ratio inliers**: inliers/total_matches > 0.5
3. **Magnitud de t**: |t| ≈ constante entre frames
4. **Consistencia temporal**: Trayectoria suave

### Q6: ¿Qué es el "chirality check" en recoverPose?

**R:**
De las 4 soluciones (R₁, ±t), (R₂, ±t), solo una tiene **todos los puntos con z > 0** en ambas cámaras.

```cpp
// Internamente, recoverPose hace:
for each (R, t) solution:
    triangulate points
    count = points with z > 0 in both cameras
select solution with max count
```

---

## 7. ERRORES COMUNES Y DEBUGGING

### 7.1 Drift acumulado

```
Frame 1: error = 0.1°
Frame 100: error = 10° (acumulado)
Frame 1000: error = 100° → trayectoria totalmente incorrecta
```

**Solución:** Loop closure (H09) para corregir drift.

### 7.2 Escala inconsistente

```cpp
// Frame 1→2: t = [0.1, 0, 0.5]  (escala arbitraria)
// Frame 2→3: t = [0.2, 0, 0.8]  (otra escala)

// Solución: Normalizar t antes de acumular
t = t / cv::norm(t);  // Asume velocidad constante
```

### 7.3 Verificación de matriz de rotación

```cpp
// R debe ser ortogonal: R·Rᵀ = I
cv::Mat shouldBeIdentity = R * R.t();

// R debe tener det(R) = 1 (no -1)
double det = cv::determinant(R);
assert(std::abs(det - 1.0) < 1e-6);
```

---

## 8. COMPARACIÓN CON OTROS MÉTODOS

| Método | Mínimo puntos | Robustez | Velocidad |
|--------|---------------|----------|-----------|
| 5-point Essential | 5 | Alta | Media |
| 8-point Essential | 8 | Media | Alta |
| Homografía | 4 | Baja (solo planar) | Muy alta |
| PnP (con mapa) | 4 | Alta | Alta |

---

## 9. CHECKLIST DE PREPARACIÓN

- [ ] Entender geometría epipolar (E, F)
- [ ] Saber qué es Essential Matrix y sus constraints
- [ ] Explicar descomposición E → (R, t)
- [ ] Conocer problema de las 4 soluciones
- [ ] Entender RANSAC y por qué es necesario
- [ ] Saber acumular pose en world frame
- [ ] Conocer ambigüedad de escala y soluciones
- [ ] Entender chirality check
- [ ] Poder debuggear problemas de drift

---

**Generado:** 2025-01
**Proyecto:** aria-slam (C++)
