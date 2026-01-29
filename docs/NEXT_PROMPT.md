# Siguiente Sesión: Aprendiendo Clean Architecture

## Contexto

Estamos en H12 (Clean Architecture). La estructura base está creada:
- `include/core/Types.hpp` - Tipos del dominio
- `include/interfaces/*.hpp` - Interfaces abstractas
- `include/adapters/gpu/*.hpp` - Headers de adaptadores GPU
- `src/adapters/gpu/OrbCudaExtractor.cpp` - Primer adapter implementado

## Nueva Metodología de Aprendizaje

**Problema anterior:** Claude escribía código, yo copiaba sin entender.

**Solución:** Aprendizaje guiado con preguntas. Claude pregunta, yo respondo, luego escribo el código.

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

## Cómo Continuar

1. Abre este archivo
2. Responde las 3 preguntas (escríbelas en el chat)
3. Basado en tus respuestas, implementa `CudaMatcher.cpp`
4. Claude solo corrige si hay errores conceptuales

## Regla de Oro

> "¿Este código depende de una librería externa?"
> - Sí → va en `adapters/`
> - No → va en `core/` o `interfaces/`

## Archivos Relevantes

- Ver `docs/H12_LEARN.md` para la explicación completa
- Ver `src/adapters/gpu/OrbCudaExtractor.cpp` como ejemplo de adapter
- Ver `include/interfaces/IMatcher.hpp` para la interface que debes implementar
