#include <iostream>
#include "alias.hpp"

#include "params.hpp" 

#include "ontheFly.hpp" 
#include "LagrangeBasisFunctions.hpp"
#include "GaussLobatto.hpp" 
#include "basisfunctions.hpp" 
#include <spectralcm.hpp>
#include "timers.hpp"
#include <time.h>
#include <omp.h>
// omp_set_num_threads(16);

int main(int argc, char* argv[]) {
    //testing decreaseGrid
    // std::cout << "Welcome to C++" << std::endl;
    // std::string name = "test1.csv";
    // cie::linalg::Matrix a = cie::spectralkernel::solve_w_eq_2d_decreaseGrid(true, name);
    // omp_set_num_threads(4);
    // omp_set_num_threads(16);
    int num_threads = omp_get_num_procs();
    if (argc > 1) {
        num_threads = std::atoi(argv[1]);
        if (num_threads < 1) {
            std::cerr << "invalid number of threads: " << argv[1] << std::endl;
            return 1;
        }
    }
    omp_set_num_threads(num_threads);

    std::cout << "\n\t\t---Welcome to omp!" << std::endl;
    // #pragma omp parallel
    // {
    //     printf("Hello from process: %d\n", omp_get_thread_num());
    // }

    std::string name2 = "scm_omp.csv";
    // cie::linalg::Matrix a2 = cie::ompkernel::solve_w_eq_2d_decreaseGrid(true, name2);
    // std::string solver2use = "ontheFly_omp_write";
    std::string solver2use = "ontheFly_omp_write";
    double time2 = 0.0;
    TIME_FUNC(time2, 
        {cie::spectralkernel::solve_w_eq_2d_decreaseGrid(true, 
                        name2,solver2use);});
    std::cout << "\t--\tTotal time:\t" << time2 << std::endl;


}
