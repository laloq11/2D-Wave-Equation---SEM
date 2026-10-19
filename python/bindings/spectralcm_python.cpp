#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "params.hpp"
#include "ontheFly.hpp"
#include "spectralcm.hpp"
#include "denseMatrixConversion.hpp"

#ifdef CUDA_ON
#include "cuontheFly.cuh"
#endif


PYBIND11_MODULE( pyspectralcm, m )
{
    m.doc( ) = "Spectral element method computation kernel"; // optional module docstring

    pybind11::class_<cie::spectralkernel::Params>patch(m, "SCM_ParametersPatch");


    //m.def( "evaluateBSplineBasis", &cie::spectralkernel::evaluateBSplineBasis, "Evaluates single b-spline basis function." );
    //m.def("evaluate2DCurve", &cie::spectralkernel::evaluate2DCurve, "Samples a B-Spline curve.");
    //m.def("evaluateSurface", &cie::spectralkernel::evaluateSurface, "Samples a B-Spline surface.");


    patch.def(pybind11::init<std::array<size_t, 2>,
        std::array<double, 2>,std::array<double, 2>,
        double,double,double,double,double,
        double,double,double,double,int,size_t>());

    patch.def("nsteps", &cie::spectralkernel::Params::nsteps);
    patch.def("nelements", &cie::spectralkernel::Params::nelements);
    patch.def("lagrangeCoords", &cie::spectralkernel::Params::gllcoords2py);
    patch.def("lengths", &cie::spectralkernel::Params::lengths);
    patch.def("ndofdirection", &cie::spectralkernel::Params::ndofdirection);
    patch.def("center", &cie::spectralkernel::Params::center);
    patch.def("radius", &cie::spectralkernel::Params::radius);
    patch.def("mapX", &cie::spectralkernel::Params::mapX);
    patch.def("mapY", &cie::spectralkernel::Params::mapY);
    //solve from Params class
    patch.def("solve", &cie::spectralkernel::Params::solve);//needs "ontheFly" string input
    // patch.def("solve and outputs on csv file", &cie::spectralkernel::Params::solve);//needs "ontheFly" string input


    m.def( "testingPython_output", &cie::spectralkernel::testingPython_output, "Initializes a ontheFly class and returns a matrix to python." );
    // m.def( "testingPython_output_omp", &cie::spectralkernel::testingPython_output_omp, "Initializes a ontheFly_omp class and returns a matrix to python." );

    // #ifdef CUDA_ON
    // m.def("cuda_w_eq_decreeseGrid_solve1", &cie::otFlyGPU::solve_w_eq_2d_decreaseGrid_cuda1<float>,"Initializes a cuda kernel and prints on csv file.");//needs "ontheFly" string input
    // #endif

}
