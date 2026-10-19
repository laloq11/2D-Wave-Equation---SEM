#pragma once

#include <vector>
#include "stddef.h"
#include <alias.hpp>

namespace cie
{
namespace spectralkernel
{

double evaluateBSplineBasis( double t, size_t i, size_t p, const std::vector<double>& knotVector );

double evaluateBSplineDerivative(double t,
    size_t functionIndex,
    size_t degree,
    const std::vector<double>& knotVector,
    size_t diffOrder);

std::vector<double> arange(int number, double addition);

} // splinekernel
} // cie

void runtime_check(bool result, const char message[]);