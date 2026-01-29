#pragma once
#include <string>

class Animal {
public:
    virtual std::string hacerSonido() = 0;  // = 0 significa "sin implementación"
};