# Aprendizaje C++ Básico

## Estructura

```
learn/cpp_basics/
├── 01_clases/        # Clases básicas
├── 02_herencia/      # Herencia y virtual
├── 03_interfaces/    # Interfaces (= 0) y polimorfismo
└── README.md
```

---

## 01_clases - Clases básicas

**Archivos:** `Contador.hpp`, `Contador.cpp`

**Conceptos:**
- `.hpp` = declara qué existe
- `.cpp` = implementa cómo funciona
- `public:` = accesible desde fuera
- `private:` = solo accesible dentro de la clase
- `valor_` = convención para atributos
- `Clase::funcion()` = "esta función pertenece a Clase"

---

## 02_herencia - Herencia y virtual

**Archivos:** `Animal.hpp`, `Perro.hpp`, `Perro.cpp`

**Conceptos:**
- `: public Animal` = hereda de Animal (como extends en Java)
- `virtual` = permite que los hijos reemplacen la función
- `override` = indica que reemplaza función del padre

---

## 03_interfaces - Interfaces y polimorfismo

**Archivos:** `IFigura.hpp`, `Rectangulo.hpp`, `Rectangulo.cpp`, `Circulo.hpp`, `Circulo.cpp`

**Conceptos:**
- `= 0` = función sin implementación, DEBE implementarse
- Interface = clase con solo funciones `= 0`
- Polimorfismo = usar `IFigura*` para cualquier figura

---

## 04_adapter - Patrón Adapter (traducir tipos)

**Archivos:** `ITermometro.hpp`, `SensorFahrenheit.hpp`, `AdapterTermometro.hpp`, `AdapterTermometro.cpp`

**Conceptos:**
- Adapter = traduce entre dos interfaces incompatibles
- Tu programa usa `ITermometro` (Celsius)
- Librería externa usa `SensorFahrenheit` (Fahrenheit)
- `AdapterTermometro` traduce F → C

**Conexión con Clean Architecture:**
| Termómetro | SLAM |
|------------|------|
| `ITermometro` | `IMatcher` |
| `SensorFahrenheit` | OpenCV CUDA |
| `AdapterTermometro` | `CudaMatcher` |

---

## Resumen rápido

| Concepto | Significado |
|----------|-------------|
| `.hpp` | Declara |
| `.cpp` | Implementa |
| `private:` | Solo dentro de la clase |
| `valor_` | Atributo (convención) |
| `virtual` | Hijos pueden reemplazar |
| `= 0` | Sin implementación |
| `: public X` | Hereda de X |
| `override` | Reemplaza del padre |
