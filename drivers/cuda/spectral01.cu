#include <cuda.h>
#include <cuda_runtime.h>

#include <CLI/CLI.hpp>
#include <stdio.h>

#include "error_handling.hpp"
#include "gpu_helpers.hpp"
#include "test_helpers.hpp"
#include "timers.hpp"
#include "types.hpp"

#include "linalg.cuh"
#include "linalg_omp.hpp"

#include "alias.hpp"

#include "params.hpp" 

#include "ontheFly.hpp" 
#include "LagrangeBasisFunctions.hpp"
#include "GaussLobatto.hpp" 
#include "basisfunctions.hpp" 
#include "spectralcm.hpp"


#include "cubase.hpp"
#include "cuontheFly.cuh"



constexpr int default_block_size = 16; //this must be the same as in linalg.cuh
// constexpr int block_size = 128;


int main(int argc, char** argv)
{
    CLI::App app{"Operations"};
    std::string  cudaSolver = "";
    app.add_option("--solver", cudaSolver, "Options[]");
    bool debug = false;
    app.add_option("--debug", debug, "debug print statements the result");
    static int block_size = default_block_size;
    app.add_option("-b,--block_size", block_size, "Thread block size");
    CLI11_PARSE(app, argc, argv);

    using ValueType = float;
    if ( cudaSolver == ""){
        throw std::invalid_argument("[Error] No given GPU - cuda solver ");
    }
    if ( cudaSolver != "singleThread" || cudaSolver != "pDegreeThread"){
        throw std::invalid_argument("[Error] Given GPU - cuda solver not valid. ");
    }
    std::stringstream debug_buffer;
    std::string output_name = "scm_cuda_"+cudaSolver+".csv";

    std::string storage_type = "";//todo
    cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid();
    double t = 0.0;
    bool writeB = true;
    if ( cudaSolver == "singleThread"){
    TIME_FUNC(t, 
        {cie::otFlyGPU::solve_w_eq_2d_decreaseGrid_singleThread<float>
            (params,writeB,output_name,storage_type);
        });
    } else if (cudaSolver == "pDegreeThread"){
            TIME_FUNC(t, 
        {cie::otFlyGPU::solve_w_eq_2d_decreaseGrid_pDegreeThread<float>
            (params,writeB,output_name,storage_type);
        });
    }
    std::cout << "\t--\tTotal time:\t" << t << std::endl;

}
