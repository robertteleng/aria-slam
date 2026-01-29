#pragma once
#include "IFigura.hpp"

class Circulo : public IFigura {
public:
    Circulo(double radio);
    double calcularArea() override;
private:
    double radio_;
};
