#include "catch.hpp"

#include <vector>
#include "alias.hpp"
#include "GaussLobatto.hpp" 
#include "params.hpp" 

namespace cie
{
namespace spectralkernel
{


TEST_CASE("Paramstest")
{
    double eps = 1e-12;
    //initializing
    std::array<size_t, 2> nelements = { 5, 5 };
    std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
    std::array<double, 2>center = { 1.0, 1.0 };

    double nsteps = 1000;
    size_t polynomialDegree = 2;

    double duration = 12;
    double radius = 0.4;
    double frequency = 1.0;
    double sigmaS = 0.06;
    double E = 1.0;
    double rho = 1.0;
    double domain = 1.0;
    double alpha = 1.0;
    double c = 1.0;

    Params params(nelements, lengths, center, E, rho, domain, alpha, c,
        duration, radius, frequency, sigmaS,
        nsteps, polynomialDegree);


    //std::vector<double> number = params.createLocationMap(0, 0);
    std::vector<double> result = { 0,  1,  2, 11, 12, 13, 22, 23, 24 };
    std::vector<double> number = params.createLocationMap(0, 0);

    for (size_t i = 0; i < number.size(); ++i) {
        CHECK(number[i] == Approx(result[i]).margin(eps));

    }

    std::vector<double> result2 = { 96,  97,  98, 107, 108, 109, 118, 119, 120 };
    std::vector<double> number2 = params.createLocationMap(4,4);

    for (size_t i = 0; i < number2.size(); ++i) {
        CHECK(number2[i] == Approx(result2[i]).margin(eps));

    }

    std::vector<double> result3 = { 68, 69, 70, 79, 80, 81, 90, 91, 92 };
    std::vector<double> number3 = params.createLocationMap(3,1);

    for (size_t i = 0; i < number3.size(); ++i) {
        CHECK(number3[i] == Approx(result3[i]).margin(eps));

    }
    std::vector<double> result4 = { 48, 49, 50, 59, 60, 61, 70, 71, 72 };
    std::vector<double> number4 = params.createLocationMap(2,2);

    for (size_t i = 0; i < number4.size(); ++i) {
        CHECK(number4[i] == Approx(result4[i]).margin(eps));

    }
}//Paramstest

TEST_CASE("MappingTest")
{
    double eps = 1e-12;
    //initializing
    std::array<size_t, 2> nelements = { 5, 5 };
    std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
    std::array<double, 2>center = { 1.0, 1.0 };

    double nsteps = 1000;
    size_t polynomialDegree = 2;

    double duration = 12;
    double radius = 0.4;
    double frequency = 1.0;
    double sigmaS = 0.06;
    double E = 1.0;
    double rho = 1.0;
    double domain = 1.0;
    double alpha = 1.0;
    double c = 1.0;

    Params params(nelements, lengths, center, E, rho, domain, alpha, c,
        duration, radius, frequency, sigmaS,
        nsteps, polynomialDegree);

    double ielement = 2;
    double jelement = 2;
    double s = params.gllcoords(1);
    double r = params.gllcoords(1);
    std::vector<double> result = { 1.0,1.0 };

    CHECK(params.mapX(ielement, r) == Approx(result[0]).margin(eps));
    CHECK(params.mapY(jelement, s) == Approx(result[1]).margin(eps));

    ielement = 4;
    jelement = 4;
    result = { 1.8,1.8 };
    CHECK(params.mapX(ielement, r) == Approx(result[0]).margin(eps));
    CHECK(params.mapY(jelement, s) == Approx(result[1]).margin(eps));

    ielement = 0;
    jelement = 4;
    s = params.gllcoords(2);
    r = params.gllcoords(1);
    result = { 0.2,2.0 };
    CHECK(params.mapX(ielement, r) == Approx(result[0]).margin(eps));
    CHECK(params.mapY(jelement, s) == Approx(result[1]).margin(eps));


}//MappingTest

TEST_CASE("spatialfunctiontest")
{
    double eps = 1e-12;
    //initializing
    std::array<size_t, 2> nelements = { 5, 5 };
    std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
    std::array<double, 2>center = { 1.0, 1.0 };

    double nsteps = 1000;
    size_t polynomialDegree = 2;

    double duration = 12;
    double radius = 0.4;
    double frequency = 1.0;
    double sigmaS = 0.06;
    double E = 1.0;
    double rho = 1.0;
    double domain = 1.0;
    double alpha = 1.0;
    double c = 1.0;

    Params params(nelements, lengths, center, E, rho, domain, alpha, c,
        duration, radius, frequency, sigmaS,
        nsteps, polynomialDegree);

    double ielement = 0;
    double jelement = 0;
    double s = params.gllcoords(0);
    double r = params.gllcoords(0);
    double result_x = 10;// { 1.0, 1.0 };
    // double result_t = 10;// { 1.0, 1.0 };
    double x = params.mapX(ielement, r);
    double y = params.mapY(jelement, s);

    CHECK(params.fx(x, y) == Approx(result_x).margin(eps));
    //CHECK(params.fx(x, y) == Approx(result).margin(eps));

    s = params.gllcoords(1);
    y = params.mapY(jelement, s);
    result_x = 0.038659201394728004;

    CHECK(params.fx(x, y) == Approx(result_x).margin(eps));

    ielement = 1;
    jelement = 0;
    s = params.gllcoords(0);
    r = params.gllcoords(0);
    x = params.mapX(ielement, r);
    y = params.mapY(jelement, s);
    result_x = 2.23363143620315e-09;

    CHECK(params.fx(x, y) == Approx(result_x).margin(eps));

    ielement = 4;
    jelement = 4;
    s = params.gllcoords(0);
    r = params.gllcoords(0);
    x = params.mapX(ielement, r);
    y = params.mapY(jelement, s);
    result_x = 1.473562036913071e-308;
    CHECK(params.fx(x, y) == Approx(result_x).margin(eps));


}//spatialfunctiontest
TEST_CASE("timefunctiontest")
{
    double eps = 1e-12;
    //initializing
    std::array<size_t, 2> nelements = { 5, 5 };
    std::array<double, 2> lengths = { 2.0, 2.0 }; //x, y dimension
    std::array<double, 2>center = { 1.0, 1.0 };

    double nsteps = 1000;
    size_t polynomialDegree = 2;

    double duration = 12;
    double radius = 0.4;
    double frequency = 1.0;
    double sigmaS = 0.06;
    double E = 1.0;
    double rho = 1.0;
    double domain = 1.0;
    double alpha = 1.0;
    double c = 1.0;

    Params params(nelements, lengths, center,E,rho,domain,alpha,c,
        duration, radius, frequency, sigmaS,
        nsteps, polynomialDegree);

    double t = .05;
    double result_t = 1.723356746322617e-06;
    CHECK(params.ft(t) == Approx(result_t).margin(eps));

    t = .005;
    result_t = 3.2074158468037215e-07;
    CHECK(params.ft(t) == Approx(result_t).margin(eps));

    t = .25;
    result_t = 0.0011179100567117002;
    CHECK(params.ft(t) == Approx(result_t).margin(eps));

    t = .75;
    result_t = 7.2044418165232935;
    CHECK(params.ft(t) == Approx(result_t).margin(eps));

}//timefunctiontest

}//spectralkernel
}//cie