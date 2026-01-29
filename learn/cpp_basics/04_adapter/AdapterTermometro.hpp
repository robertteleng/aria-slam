#pragma once
#include "ITermometro.hpp"
#include "SensorFahrenheit.hpp"

class AdapterTermometro : public ITermometro {
public:
    double leerCelsius() override;
private:
    SensorFahrenheit sensor_;  // Usa el sensor externo
};
