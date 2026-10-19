#include <cstring>
#include <CLI/CLI.hpp>

//for running using the libraries
#include "alias.hpp"

#include "params.hpp" 

#include "ontheFly.hpp" 
#include "LagrangeBasisFunctions.hpp"
#include "GaussLobatto.hpp" 
#include "basisfunctions.hpp" 
#include "spectralcm.hpp"
#include "spectralcm.hpp" 

#include "timers.hpp"

#include <stdio.h>

// #include "error_handling.hpp" //this will be necessary when cuda is available
#include "test_helpers.hpp"
#include "timers.hpp" //important to keep!
#include "types.hpp" //not really relevant, we are not using different types

#include <omp.h>
// omp_set_num_threads(4);


int main(int argc, char** argv)
{
    //fKu force - Ku
    //mass matrix element level
    //mm matrix matrix sum (this might not be necessary)
    // omp_set_num_threads(4);


    CLI::App app{"Operations for roofline"};
    int num_elems = 5;
    app.add_option("-n,--num_elems", num_elems, "Number of elements - both directions");
    int num_reps = 20;
    app.add_option("--num_reps", num_reps, "Number of repetitions");
    bool verify = false;
    app.add_option("--verify", verify, "Verify the result");
    std::string device = "omp";
    app.add_option("-d, --device", device,
                   "The device to run the operation on");
    std::string op = "fKu";//fKu mass mm
    app.add_option("-o, --op", op, "The operation to run");
    int PolDegree = 4;
    app.add_option("-p,--poldegree", PolDegree, "Polynomial degree order - both directions");
    bool debug = false;
    app.add_option("--debug", debug, "true make control out enabled.");
    CLI11_PARSE(app, argc, argv);

    using ValueType = double;
    if(debug) {
        #pragma omp parallel
        {
            printf("Hello from process: %d\n", omp_get_thread_num());
        }
    }
    // std::cout<<"op is: "<<op<<std::endl;
    if (op == "fKu") {

        // std::cout << "\t--\tInitializing parameters..." << std::endl;
        double eps = 1e-12;
        int nsteps = 100;
        // int ndofglobal = 40;

        //initialize params from function
        cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(num_elems,num_elems,PolDegree);
        if (debug) { std::cout << "\t--\tInitializing solver..." << std::endl; }
        std::string ontheFlymode = "strg_1";//storage,strg_1

        cie::otFly::ontheFly fKu_test(params.nsteps(), params.ndofglobal(),ontheFlymode);

        if (debug) {std::cout << "\t--\t Setting up omp..." << std::endl;}

        constexpr int block_size = 128;
        const int num_blocks = (num_elems + block_size - 1) / block_size;

        double fKu_time = 0.0;
        int it = 1;//random value that must be repeated num_reps times to make an average
        for (int rep = 2; rep < num_reps; ++rep) {
            if (device == "omp") {
                // std::memcpy(y.data(), o_y.data(),
                //             sizeof(ValueType) * num_elems);
                // TIME_FUNC(fKu_time, {omp_namespace::axpy(&alpha, x.data(), num_elems, y.data());});

                // TIME_FUNC(fKu_time, {cie::otFly::compute_fKu_Cvector_OMP(fKu_test.get_fku(),fKu_test.get_u(),params,it);});
                TIME_FUNC(fKu_time, {cie::otFly::compute_fKu_Cvector_OMP(fKu_test.get_fku(),fKu_test.get_up(it),params,it);});
            }

            if (debug) {std::cout<<"\t \trep: "<<rep<<std::endl; }
        }

        std::cout << fKu_time / num_reps << std::endl;
        // std::cout << "Success!" << std::endl;

        if (verify) {
            cie::otFly::ontheFly fKu_expected(params.nsteps(), params.ndofglobal(),ontheFlymode);
            std::vector<ValueType> expected=fKu_expected.compute_fKu_Cvector(params,it);

            std::vector<ValueType> omp_result = fKu_test.get_fku();

            ASSERT_TRUE(compare_arrays(num_elems, omp_result.data(), expected.data()).first);
        }
    } 
    else if (op == "Mass") {
    //-------------------------     Mass     -------------------------------------------



        // std::cout << "\t--\tInitializing parameters..." << std::endl;
        double eps = 1e-12;
        int nsteps = 100;
        // int ndofglobal = 40;

        //initialize params from function
        cie::spectralkernel::Params params = cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(num_elems,num_elems,PolDegree);
        if (debug) { std::cout << "\t--\tInitializing solver for mass test..." << std::endl; }
        std::string ontheFlymode = "strg_1";//storage,strg_1

        cie::otFly::ontheFly Mu_test(params.nsteps(), params.ndofglobal(),ontheFlymode);

        if (debug) {std::cout << "\t--\t Setting up omp..." << std::endl;}

        constexpr int block_size = 128;
        const int num_blocks = (num_elems + block_size - 1) / block_size;

        double fKu_time = 0.0;
        int it = 1;//random value that must be repeated num_reps times to make an average
        for (int rep = 2; rep < num_reps; ++rep) {
            if (device == "omp") {
                // std::memcpy(y.data(), o_y.data(),
                //             sizeof(ValueType) * num_elems);
                // TIME_FUNC(fKu_time, {omp_namespace::axpy(&alpha, x.data(), num_elems, y.data());});

                TIME_FUNC(fKu_time, {cie::otFly::compute_Mass_Cvector_OMP(Mu_test.get_mu(),params);});
            }

            if (debug) {std::cout<<"\t \trep: "<<rep<<std::endl; }
        }

        std::cout << fKu_time / num_reps << std::endl;
        // std::cout << "Success!" << std::endl;

        if (verify) {
            cie::otFly::ontheFly Mu_expected(params.nsteps(), params.ndofglobal(),ontheFlymode);
            // Copy data back from device
            std::vector<ValueType> omp_result = Mu_test.get_mu();
            std::vector<ValueType> expected= Mu_expected.compute_Mass_Cvector(params);

            ASSERT_TRUE(compare_arrays(num_elems, omp_result.data(), expected.data()).first);
        }




    }
//-------------------------      END     -------------------------------------------

    else {
        throw std::runtime_error("incorrect op");
    }

    return 0;
}
