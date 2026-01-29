#pragma once
#include "IFigura.hpp"

class Rectangulo : public IFigura {
public:
    Rectangulo(double ancho, double alto);
    double calcularArea() override;
private:
    double ancho_;
    double alto_;
};
