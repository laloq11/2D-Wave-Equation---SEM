#include <cstring>
#include "cubase.hpp"
#include "cuontheFly.cuh"
#include "cuda_tests.hpp"

#include <fstream>   // For file handling
#include <algorithm>
/* 
Main goal of challenges is to come up with the best combination
without giving up precision. The goal in this document is to 
    1.parallelize all segmentas that are possible
    2.optimize code
So far on the test_cu folder we guaranteed that the based algorithms
are now references. Which means that a simple comparison in this section 
of the code must be more than enough.
This part
*/

namespace cie {
    namespace otFlyGPU {
        template <typename ValueType>
        __host__ void alg57_challenge_12(){
            /*
            * Challenge of alg_57_kernel and alg_57_kernel_2
            */
        } 
        __host__ void StiffnessMatrix_challenge_12(){
        //WIP 
        // std::cout << "\t--\tInitializing parameters..." << std::endl;
        // double eps = 1e-12;
        // int nsteps = 100;
        // // int ndofglobal = 40;

        // //initialize params from function
        // cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(num_elems,num_elems,PolDegree);
        // if (debug) { std::cout << "\t--\tInitializing solver..." << std::endl; }
        // std::string ontheFlymode = "strg_1";//storage,strg_1

        // cie::otFly::ontheFly fKu_test(params.nsteps(), params.ndofglobal(),ontheFlymode);

        // if (debug) {std::cout << "\t--\t Setting up omp..." << std::endl;}

        // constexpr int block_size = 128;
        // const int num_blocks = (num_elems + block_size - 1) / block_size;

        // double fKu_time = 0.0;
        // int it = 1;//random value that must be repeated num_reps times to make an average
        // for (int rep = 2; rep < num_reps; ++rep) {
        //     if (device == "omp") {
        //         // std::memcpy(y.data(), o_y.data(),
        //         //             sizeof(ValueType) * num_elems);
        //         // TIME_FUNC(fKu_time, {omp_namespace::axpy(&alpha, x.data(), num_elems, y.data());});

        //         TIME_FUNC(fKu_time, {cie::otFly::compute_fKu_Cvector_OMP(fKu_test.get_fku(),fKu_test.get_u(),params,it);});
        //     }

        //     if (debug) {std::cout<<"\t \trep: "<<rep<<std::endl; }
        // }

        // std::cout << fKu_time / num_reps << std::endl;
        // // std::cout << "Success!" << std::endl;

        // if (verify) {
        //     cie::otFly::ontheFly fKu_expected(params.nsteps(), params.ndofglobal(),ontheFlymode);
        //     std::vector<ValueType> expected=fKu_expected.compute_fKu_Cvector(params,it);

        //     std::vector<ValueType> omp_result = fKu_test.get_fku();

        //     ASSERT_TRUE(compare_arrays(num_elems, omp_result.data(), expected.data()).first);
        // }
        }
        __host__ void MassMatrix_challenge_12(){
        //-------------------------     Mass     -------------------------------------------
        //WIP
            // bool debug =1;
    
            // // std::cout << "\t--\tInitializing parameters..." << std::endl;
            // double eps = 1e-12;
            // int nsteps = 100;
            // // int ndofglobal = 40;
    
            // //initialize params from function
            // cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(num_elems,num_elems,PolDegree);
            // if (debug) { std::cout << "\t--\tInitializing solver for mass test..." << std::endl; }
            // std::string ontheFlymode = "strg_1";//storage,strg_1
    
            // cie::otFly::ontheFly Mu_test(params.nsteps(), params.ndofglobal(),ontheFlymode);
    
            // if (debug) {std::cout << "\t--\t Setting up omp..." << std::endl;}
    
            // constexpr int block_size = 128;
            // const int num_blocks = (num_elems + block_size - 1) / block_size;
    
            // double fKu_time = 0.0;
            // int it = 1;//random value that must be repeated num_reps times to make an average
            // for (int rep = 2; rep < num_reps; ++rep) {
            //     if (device == "omp") {
            //         // std::memcpy(y.data(), o_y.data(),
            //         //             sizeof(ValueType) * num_elems);
            //         // TIME_FUNC(fKu_time, {omp_namespace::axpy(&alpha, x.data(), num_elems, y.data());});
    
            //         TIME_FUNC(fKu_time, {cie::otFly::compute_Mass_Cvector_OMP(Mu_test.get_mu(),params);});
            //     }
    
            //     if (debug) {std::cout<<"\t \trep: "<<rep<<std::endl; }
            // }
    
            // std::cout << fKu_time / num_reps << std::endl;
            // // std::cout << "Success!" << std::endl;
    
            // if (verify) {
            //     cie::otFly::ontheFly Mu_expected(params.nsteps(), params.ndofglobal(),ontheFlymode);
            //     // Copy data back from device
            //     std::vector<ValueType> omp_result = Mu_test.get_mu();
            //     std::vector<ValueType> expected= Mu_expected.compute_Mass_Cvector(params);
    
            //     ASSERT_TRUE(compare_arrays(num_elems, omp_result.data(), expected.data()).first);
            // }

        }

    }
}