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
#include "linalg.hpp"
#include "alias.hpp"
#include "params.hpp"
#include "spectralcm.hpp"
#include "cuontheFly.cuh"
#include "cubase.hpp"
#include <random>
#include <memory>
#include <algorithm>
#include <numeric>
#include <vector>

//./ex1/roofline -n 500 -d omp -o matmul --debug 1
// constexpr int default_block_size = 32; //this must be the same as in linalg.cuh

// constexpr int block_size = 128;

// mean, median, min and max of the timed repetitions [s]
struct TimeStats { double mean, median, min, max; };

TimeStats time_stats(std::vector<double> t) {
    std::sort(t.begin(), t.end());
    const size_t n = t.size();
    const double median = (n % 2 == 1) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
    return {std::accumulate(t.begin(), t.end(), 0.0) / n, median, t.front(), t.back()};
}

// stdout of a timing run: the mean, or with --stats 1 "mean median min max" [s]
void print_times(const std::vector<double>& t, bool stats) {
    const TimeStats s = time_stats(t);
    if (stats) std::cout << s.mean << " " << s.median << " " << s.min << " " << s.max << std::endl;
    else       std::cout << s.mean << std::endl;
}

int main(int argc, char** argv)
{
    CLI::App app{"Operations for roofline"};
    int num_elems = 5;
    app.add_option("-n,--num_elems", num_elems, "Number of elements - both directions");
    int num_reps = 10;
    app.add_option("--num_reps", num_reps, "Number of repetitions");
    int num_warmup = 1;
    app.add_option("--warmup", num_warmup, "Untimed repetitions before the timed ones (CUDA start-up, first use of memory)");
    bool stats = false;
    app.add_option("--stats", stats, "Print mean, median, min and max of the timed repetitions [s] instead of the mean");
    bool verify = false;
    app.add_option("--verify", verify, "Verify the result");
    std::string device = "omp";
    app.add_option("-d, --device", device,
                   "Solver: omp, cu_singleThread or cu_pDegreeThread");
    // std::string op = "57";//fKu mass mm
    // app.add_option("-o, --op", op, "The operation to run ");
    int polynomialDegree = 2;
    app.add_option("-p,--poldegree", polynomialDegree, "Polynomial degree order - both directions");
    bool debug = false;
    app.add_option("--debug", debug, "true make control out enabled.");

    size_t num_threads = 1;
    app.add_option("-t,--threads", num_threads, "# of threads working with.");
    std::string filename = "default";
    app.add_option("--filename", filename, "Output filename.");
    double duration = 12.0;
    app.add_option("--duration", duration, "Simulated time.");
    size_t nsteps = 1000;
    app.add_option("--nsteps", nsteps, "Number of time steps (dt = duration/nsteps).");

    CLI11_PARSE(app, argc, argv);
    if (num_reps < 1 || num_warmup < 0) {
        std::cerr << "Error: --num_reps must be at least 1 and --warmup at least 0" << std::endl;
        return 1;
    }
    if (device != "omp") {
        try { cie::otFlyGPU::check_polynomialDegree_gpu(polynomialDegree); }
        catch (const std::invalid_argument& e) { std::cerr << "Error: " << e.what() << std::endl; return 1; }
    }
    const bool outputfile = (filename != "default");
    double tolerance = 1e-4;

    std::stringstream debug_buffer;
    debug_buffer << "Running " 
                 << "\ninput size num_elements: (" << num_elems
                 << ", on device: " << device
                 << "\nnsteps: " << nsteps << std::endl;


    using ValueType = float;

    debug_buffer << "\t--\tInitializing parameters..." << std::endl;
    cie::spectralkernel::Params params = 
    // cie::spectralkernel::w_eq_2d_decreaseGrid_4testing( 
    cie::spectralkernel::w_eq_2d_autostable(
                        num_elems,
                        num_elems,
                        polynomialDegree,
                        nsteps
    );
    params.changesamplingFrequency(params.nsteps());
    // dt comes from the Cp table (w_eq_2d_autostable); --duration is not used yet
    debug_buffer << "dt: " << params.dt() << ", simulated time: " << params.dt() * nsteps << std::endl;

    cie::otFlyGPU::spectral_pointers<ValueType> sp;
    int exit_code = 0;   // 1 if --verify fails

    if (device == "cu_singleThread" || device == "cu_pDegreeThread"){
        omp_set_num_threads(1); 

        ValueType dx= params.dx();
        ValueType dy= params.dy();
        ValueType sS= params.sigmaS();
        ValueType sT= params.sigmaT();
        ValueType t0= params.t0();
        cie::otFlyGPU::lambdaXY_kernel<ValueType> MapX{dx}; 
        cie::otFlyGPU::lambdaXY_kernel<ValueType> MapY{dy}; 
        cie::otFlyGPU::lambdafx_kernel<ValueType> fx{sS}; 
        cie::otFlyGPU::lambdaft_kernel<ValueType> ft_k{sT,t0};
        // fKu kernel and threads per element for this device (untimed)
        cie::otFlyGPU::solverStruct<ValueType> solver =
            cie::otFlyGPU::get_solverStruct_gpu<ValueType>(device, polynomialDegree);


        std::vector<ValueType> cuda_res(params.ndofglobal());

        std::vector<double> rep_times;   // timed repetitions [s]; the warm-up runs are not kept
        for (int tt = 0; tt < num_warmup + num_reps; tt++){
            double rep_time = 0.0;
            TIME_FUNC(rep_time, {
                    cie::otFlyGPU::preinitialize_invM_fKu_gpu<ValueType>(params, 
                        sp, 
                        debug_buffer);

                    cie::otFlyGPU::benchmark_fKu_gpu<ValueType>(params,
                        sp,
                        MapX,MapY,fx,ft_k,
                        solver);
                    ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
                    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
                    cie::otFlyGPU::finish_simulation_gpu(cuda_res,sp,debug_buffer);
            });
            if (tt >= num_warmup) rep_times.push_back(rep_time);
            if (debug){
                debug_buffer <<"end of rep "<<tt+1<<(tt < num_warmup ? " (warm-up, untimed)" : "")<<std::endl;
                std::cout << debug_buffer.str(); // Print current contents
                debug_buffer.str("");            // Clear the contents
                debug_buffer.clear();            // Reset any error flags
            }
        }
        
        debug_buffer << "Took time: " << time_stats(rep_times).mean << std::endl;
        if (outputfile){ // untimed: last u_ from the CUDA solver
            cie::otFlyGPU::writeU_inter<ValueType>(cuda_res, filename);
        }

        if (verify){
            std::string mode = "storage";
            debug_buffer << "\t--\t Computing Reference..." << std::endl;
            debug_buffer << "\t--\tInitializing solver..." << std::endl;
            cie::otFly::ontheFly solver(params.nsteps(), params.ndofglobal(),mode);
            debug_buffer << "\t--\tStarting solver..." << std::endl;
            solver.otF_Solve_OMP_write(params,filename,false,debug); // reference only, file holds CUDA result
    
            //formating data back to the standard-comparable format.
            std::vector<ValueType> res_omp(params.ndofglobal());
            for (size_t r=0;r<params.ndofglobal();r++) res_omp[r] = solver.get_u()(0,r);



            auto comparison = compare_vectors(res_omp,cuda_res,tolerance);
            if (!comparison.first) {
                std::cerr << "verify: FAILED, GPU and omp differ by more than " << tolerance
                          << " at entry " << comparison.second << std::endl;
                exit_code = 1;
            } else {
                std::cerr << "verify: passed (GPU and omp agree within " << tolerance << ")" << std::endl;
            }
        }

        if (debug){
            std::cout<<debug_buffer.str();
        }
        else {
            print_times(rep_times, stats);
        }
    }
    else if (device == "omp"){
        omp_set_num_threads(num_threads); // Sets number of threads for CPU OpenMP region
        std::string mode = "storage";
        debug_buffer << "\t--\t Computing OMP!..." << std::endl;
        debug_buffer << "\t--\tInitializing solver..." << std::endl;

        std::vector<double> rep_times;   // timed repetitions [s]; the warm-up runs are not kept
        std::unique_ptr<cie::otFly::ontheFly> solver;   // kept after the loop to write the last field
        for (int tt = 0; tt < num_warmup + num_reps; tt++){
            debug_buffer << "\t--\tStarting solver..." << std::endl;
            solver.reset();   // free the previous repetition outside the timer

            // total time to solution, as on the GPU: allocation, mass vector and time loop
            double rep_time = 0.0;
            TIME_FUNC(rep_time, {
                solver = std::make_unique<cie::otFly::ontheFly>(params.nsteps(), params.ndofglobal(), mode);
                solver->otF_Solve_OMP_write(params,filename,false,debug);
                });
                if (tt >= num_warmup) rep_times.push_back(rep_time);
                if (debug){
                    debug_buffer <<"end of repetion "<<tt+1<<(tt < num_warmup ? " (warm-up, untimed)" : "")<<std::endl;
                    std::cout << debug_buffer.str(); // Print current contents
                    debug_buffer.str("");            // Clear the contents
                    debug_buffer.clear();            // Reset any error flags
                }
        }
        if (outputfile){ // untimed: the last field of the last repetition, as on the GPU
            cie::otFly::writeU_inter(solver->get_u(), filename);
        }
        debug_buffer << "Took time: " << time_stats(rep_times).mean << std::endl;
        if (debug){
            std::cout<<debug_buffer.str();
        }
        else {
            print_times(rep_times, stats);
        }
    }
    else {
        std::cerr << "Error: unknown device \"" << device << "\"; valid: omp, cu_singleThread, cu_pDegreeThread" << std::endl;
        return 1;
    }

    return exit_code;
}
