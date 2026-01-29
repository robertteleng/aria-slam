#include "Rectangulo.hpp"

Rectangulo::Rectangulo(double ancho, double alto) {
    ancho_ = ancho;
    alto_ = alto;
}

double Rectangulo::calcularArea() {
    return ancho_ * alto_;
}
