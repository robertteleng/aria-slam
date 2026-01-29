#include "AdapterTermometro.hpp"

AdapterTermometro::AdapterTermometro() : 
sensor_() {}

double AdapterTermometro::leerCelsius() {
    double fahrenheit = sensor_.leerFahrenheit();
    return (fahrenheit - 32) * 5.0 / 9.0;  // Conversión a Celsius
}