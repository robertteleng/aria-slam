#include "Contador.hpp"

Contador::Contador() {
    valor_ = 0;
} // Inicializar con valor a cero

void Contador::incrementar() {
    valor_ = valor_ + 1;
}  // Incrementar el valor del contador en 1

int Contador::getValor() {
    return valor_;
}   // Obtener el valor actual del contador