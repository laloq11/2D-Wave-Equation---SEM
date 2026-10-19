#include "basisfunctions.hpp"

#include <string>
#include <cmath>
#include <stdexcept>

namespace cie
{
namespace spectralkernel
{

double evaluateBSplineBasis( double t, 
                             size_t i, 
                             size_t p, 
                             const std::vector<double>& knotVector )
{
    double tolerance = 1e-12;

    if( p == 0 )
    {
        return ( t >= knotVector[i] && t < knotVector[i + 1] ) ||
               ( std::abs( t - knotVector[i + 1] ) < tolerance && 
                 std::abs( t - knotVector.back( ) ) < tolerance );
    }
    else
    {
        double a1 = t - knotVector[i];
        double b1 = knotVector[i + p] - knotVector[i];

        double result = 0.0;

        if( std::abs( b1 ) > tolerance )
        {
            result += a1 / b1 * evaluateBSplineBasis( t, i, p - 1, knotVector );
        }

        double a2 = knotVector[i + p + 1] - t;
        double b2 = knotVector[i + p + 1] - knotVector[i + 1];

        if( std::abs( b2 ) > tolerance )
        {
            result += a2 / b2 * evaluateBSplineBasis( t, i + 1, p - 1, knotVector );
        }

        return result;
    }
}

// This function evaluates the one-dimensional B-Spline basis functions 
// or their first derivative. Should have been in basisfunctions.cpp,
// but we forgot :)
double evaluateBSplineDerivative(double t,
    size_t functionIndex,
    size_t degree,
    const std::vector<double>& knotVector,
    size_t diffOrder)
{
    if (diffOrder == 0)
    {
        return evaluateBSplineBasis(t, functionIndex, degree, knotVector);
    }
    else if (diffOrder == 1)
    {
        if (degree == 0)
        {
            throw std::runtime_error("Invalid polynomial degree!");
        }

        double factor1 = knotVector[functionIndex + degree] - knotVector[functionIndex];
        double factor2 = knotVector[functionIndex + degree + 1] - knotVector[functionIndex + 1];

        if (std::abs(factor1) > 1e-10)
        {
            factor1 = degree / factor1 * evaluateBSplineBasis(t, functionIndex, degree - 1, knotVector);
        }

        if (std::abs(factor2) > 1e-10)
        {
            factor2 = degree / factor2 * evaluateBSplineBasis(t, functionIndex + 1, degree - 1, knotVector);
        }

        return factor1 - factor2;
    }
    else
    {
        throw std::runtime_error("Invalid diff order.");
    }
}

std::vector<double> arange(int number,double addition) {
    std::vector<double> outV;
    for (int i = 0; i < number; ++i) {
        outV.push_back(i+addition);
    }
    return outV;
}

//void testoutput_local() {
//    double eps = 1e-12;
//
//    int nsteps = 100;
//    int ndofglobal = 40;
//    cie::otFly::ontheFly test(nsteps, ndofglobal);
//    cie::linalg::Matrix vector_Result = test.hardcodedwriteonu();
//    std::string name = "outputmain5.csv";
//
//    cie::otFly::writeU_(vector_Result, name);
//}

} // splinekernel
} // cie

void runtime_check(bool result, const char message[])
{
    if (!result)
    {
        throw std::runtime_error{ message };
    }
}