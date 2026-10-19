#pragma once

#include <tuple>
#include <array>
#include <functional>

#include "alias.hpp"
#include "params.hpp" 
#include <time.h>

namespace cie
{
namespace spectralkernel
{
Params w_eq_2d_decreaseGrid();
Params w_eq_2d_completeGrid();

// Matrix solve_w_eq_2d_decreaseGrid(bool w, std::string name);
void solve_w_eq_2d_decreaseGrid(bool w,std::string name, std::string s);
Matrix testingPython_output();
Params w_eq_2d_decreaseGrid_4testing(size_t nx, size_t ny, size_t PolDegree,
    double duration = 12.0, size_t nsteps = 1000);
Params w_eq_2d_autostable(size_t nx, size_t ny, size_t PolDegree, size_t nsteps);

void solve_w_eq_2d_decreaseGridCat(bool w,std::string name, std::string s);
}//namespace spectralkernel

namespace performancekernel
{
class Performance {
public:
    Performance( );
    Performance(std::string s);
    ~Performance( );
    void checkpoint();

    double time_Diff(size_t i) const ;

private:
    double t0_;
    std::string title_;
    std::vector<double> times_;
};



} // namespace splinekernel
} // namespace cie
