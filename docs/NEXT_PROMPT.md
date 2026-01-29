# Siguiente Sesión: Continuando H12 Clean Architecture

## Estado Actual

**H12 (Clean Architecture)** sigue en progreso. Se agregaron H15 y H16 como hitos futuros.

### Estructura H12 creada:
- `include/core/Types.hpp` - Tipos del dominio
- `include/interfaces/*.hpp` - Interfaces abstractas
- `include/adapters/gpu/*.hpp` - Headers de adaptadores GPU
- `src/adapters/gpu/OrbCudaExtractor.cpp` - Primer adapter implementado

### Nuevos hitos planificados (NO implementados aún):
- **H15**: Meta Aria Integration - Interface `IAriaDevice.hpp` creada
- **H16**: Audio Feedback - Interface `IAudioFeedback.hpp` creada
- `.venv/` con SDK de Meta Aria instalado (para cuando llegue H15)

## Ejercicio Pendiente: Implementar CudaMatcher

El código actual en main.cpp (líneas 90 y 159):
```cpp
cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
matcher->knnMatch(prev_frame->gpu_descriptors, current_frame.gpu_descriptors, knn_matches, 2);
```

### Preguntas a responder:

1. **¿Qué tipos de OpenCV ves ahí que NO deberían estar en main.cpp?**
   - (pista: todo lo que tenga `cv::` o `cuda::`)

2. **¿Qué debería recibir y devolver la interface `IMatcher`?**
   - Entrada: ¿qué datos necesita para hacer matching?
   - Salida: ¿qué devuelve? (piensa en tipos del dominio, no cv::DMatch)

3. **¿Qué traducción tiene que hacer el adapter `CudaMatcher`?**
   - Entrada: ¿de qué tipo a qué tipo?
   - Salida: ¿de qué tipo a qué tipo?

## Metodología de Aprendizaje

**Problema anterior:** Claude escribía código, yo copiaba sin entender.

**Solución:** Aprendizaje guiado con preguntas. Claude pregunta, yo respondo, luego escribo el código.

## Cómo Continuar

1. Responde las 3 preguntas en el chat
2. Basado en tus respuestas, implementa `CudaMatcher.cpp`
3. Claude solo corrige si hay errores conceptuales

## Regla de Oro

> "¿Este código depende de una librería externa?"
> - Sí → va en `adapters/`
> - No → va en `core/` o `interfaces/`

## Archivos Relevantes

- `docs/H12_LEARN.md` - Explicación completa de Clean Architecture
- `src/adapters/gpu/OrbCudaExtractor.cpp` - Ejemplo de adapter
- `include/interfaces/IMatcher.hpp` - Interface que debes implementar
- `docs/AUDIT.md` - Overview del proyecto (actualizado con H15-H16)
