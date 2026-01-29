#pragma once
#include "Animal.hpp"

class Perro : public Animal {
public:
    std::string hacerSonido() override;
};