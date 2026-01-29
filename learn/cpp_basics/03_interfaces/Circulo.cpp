#include "Circulo.hpp"

Circulo::Circulo(double radio) {
    radio_ = radio;
}

double Circulo::calcularArea() {
    return 3.14159 * radio_ * radio_;
}
