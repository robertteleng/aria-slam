#pragma once

class Contador {
public:
    Contador(); //Se inicia asi misma
    void incrementar(); // funcion para incrementar el valor
    int getValor(); // funcion para obtener el valor actual
private:
    int valor_;     // variable para almacenar el valor del contador ¿No se puede acceder private? ¿porque _?
};