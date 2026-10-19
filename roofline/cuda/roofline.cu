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

#include "linalg.hpp"
#include "alias.hpp"
#include "params.hpp"
#include "spectralcm.hpp"
#include "cuontheFly.cuh"


#include "cubase.hpp"
#include <cuda_runtime.h>
#include <random>

//./ex1/roofline -n 500 -d omp -o matmul --debug 1
// constexpr int default_block_size = 32; //this must be the same as in linalg.cuh

// constexpr int block_size = 128;

size_t generateRandomBit() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<size_t> dis(0, 1);
    return dis(gen);
}

int main(int argc, char** argv)
{
    CLI::App app{"Operations for roofline"};
    int num_elems = 5;
    app.add_option("-n,--num_elems", num_elems, "Number of elements - both directions");
    int num_reps = 20;
    app.add_option("--num_reps", num_reps, "Number of repetitions");
    bool verify = false;
    app.add_option("--verify", verify, "Verify the result");
    std::string device = "cuda";
    app.add_option("-d, --device", device,
                   "The device to run the operation on");
    std::string op = "57";//fKu mass mm
    app.add_option("-o, --op", op, "The operation to run alg 57.1,57.2, 82 or 84");
    int version = 1;
    app.add_option("-v, --version", version, "Depending on the version of the operation 1,2,3,etc.");
    int polynomialDegree = 2;
    app.add_option("-p,--poldegree", polynomialDegree, "Polynomial degree order - both directions");
    bool debug = false;
    app.add_option("--debug", debug, "true make control out enabled.");

    size_t num_threads = 1;
    app.add_option("-t,--threads", num_threads, "# of threads working with.");
    // as we are working only with one SM. the # of threads is the variable we are interested in
    // when solving the whole simulation. Each SM will be computing the whole element.
   
    CLI11_PARSE(app, argc, argv);

    std::stringstream debug_buffer;
    debug_buffer << "Running " << op 
                 << "\ninput size num_elements: (" << num_elems
                 << ", on device: " << device << std::endl;

    using ValueType = float;

    cie::spectralkernel::Params params = 
    cie::spectralkernel::w_eq_2d_decreaseGrid_4testing(
                        num_elems,
                        num_elems,
                        polynomialDegree
    );

    cie::otFlyGPU::spectral_pointers<ValueType> sp;
    cie::otFlyGPU::initialize_spectral_gpu<ValueType>(sp,params);
    
    if (device == "cuda"){
        //necessary as we need to change the variable type from double to ValueType
        ValueType dx= params.dx();
        ValueType dy= params.dy();
        ValueType sS= params.sigmaS();
        ValueType sT= params.sigmaT();
        ValueType t0= params.t0();
        cie::otFlyGPU::lambdaXY_kernel<ValueType> MapX{dx}; 
        cie::otFlyGPU::lambdaXY_kernel<ValueType> MapY{dy}; 
        cie::otFlyGPU::lambdafx_kernel<ValueType> fx{sS}; 
        cie::otFlyGPU::lambdaft_kernel<ValueType> ft_k{sT,t0}; 
    }

    if (op == "57") {
        debug_buffer << "\t--\t Starting  algorithm57..." << std::endl;
        debug_buffer << "\t--\t... using version "<<version << std::endl;

        if (device == "cuda"){

            debug_buffer << "ALG57: Using CUDA." << std::endl;

            double eps = 1e-12;
            double tolerance = 1e-6;

            size_t D = 2; //dimensions working with
            size_t P_size= (polynomialDegree + 1)*(polynomialDegree + 1);
            if (version == 2 && num_threads < P_size){
                num_threads = P_size;
            } 

            ValueType* d_Bd_diff;
            ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_Bd_diff, 
                sizeof(ValueType) * P_size * D) );

            //probalby should set this up directly to 1 or 0
            // at this point i want to see if there is any impact of using one
            //or the other
            size_t ii = generateRandomBit();
            ValueType* d_BB;
            ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_BB, 
                sizeof(ValueType) * P_size * D) );
            if (ii == 0) {
                std::vector<ValueType> B = params.Bvector<ValueType>();
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_BB, B.data(),
                    sizeof(ValueType) * P_size * D,
                    cudaMemcpyHostToDevice));
                debug_buffer<<" d(N(s))^" << ii <<"/ds" << std::endl;

            }
            else {
                std::vector<ValueType> Bdiff = params.Bdiffvector<ValueType>();
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_BB, Bdiff.data(),
                    sizeof(ValueType) * P_size * D,
                    cudaMemcpyHostToDevice));
                debug_buffer<<" d(N(s))^" << ii <<"/ds" << std::endl;

            }
            debug_buffer << "Generating Bd_diff." << std::endl;
            cie::otFlyGPU::test_alg_84_mergeBd<ValueType><<<1,1>>>(d_Bd_diff, 
                        d_BB, 
                        ii,
                        polynomialDegree+1,
                        polynomialDegree+1,
                        D,1);
            ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
            ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
            //we need this data only for computing the reference value
            //copy data of Bd_diff back2HOST 
            std::vector<ValueType> Bd_diff(P_size * D);
            ASSERT_NO_CUDA_ERRORS(cudaMemcpy(Bd_diff.data(),d_Bd_diff,
                sizeof(ValueType) * P_size * D,
                cudaMemcpyDeviceToHost));
                
            std::vector<ValueType> w1 (P_size,1.0);
            std::vector<ValueType> w1_ref (P_size,1.0);

            ValueType* d_w1;
            ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_w1, 
                sizeof(ValueType) * P_size ) );
            // function to test
            double alg57_time = 0.0;
            debug_buffer << "Starting alg57 "<<num_reps<<"times." << std::endl;

            for (size_t tt=0;tt<num_reps;tt++){

                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_w1, w1.data(),
                    sizeof(ValueType) * P_size,
                    cudaMemcpyHostToDevice));
                //Bd_diff is already on the device... No need to copy H2D
                TIME_FUNC(alg57_time, 
                    {
                    cie::otFlyGPU::testalg_57<ValueType><<<1,num_threads>>>(D,
                        d_Bd_diff,
                        d_w1,
                        polynomialDegree,
                        P_size,
                        version); 
                    });

                ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
                ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
            }

            debug_buffer << "Took time: " 
                            << alg57_time / num_reps
                            << std::endl;

            if (verify){
                //compute reference data
                cie::otFlyGPU::reference::alg_57_flat<ValueType>(D,
                    Bd_diff.data(), 
                    w1_ref.data(), 
                    polynomialDegree,
                    P_size
                );
                //copy back data... 
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(w1.data(), d_w1,
                    sizeof(ValueType) * P_size,
                    cudaMemcpyDeviceToHost));
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(w1.data(), d_w1,
                    sizeof(ValueType) * P_size,
                    cudaMemcpyDeviceToHost));            
                auto comparison = compare_vectors(w1_ref,w1,tolerance);
                if (!comparison.first) {
                    debug_buffer<< "Failed alg57 PRODUCT"<< std::endl;
                    debug_buffer<< "ref,cuda"<<std::endl;
                    for (int i =0;i<P_size;i++) 
                        debug_buffer<<w1_ref[i]<<" , "<<w1[i]<<std::endl;
                    debug_buffer<< "end"<<std::endl;


                } else {
                    debug_buffer << "++++++++++++ Result in alg57correct!" << std::endl;
                }
            }
            debug_buffer << "-- Freeing space on device." <<std::endl;
            cie::otFlyGPU::terminate_spectral_gpu(sp);
            cudaFree(d_w1);
            cudaFree(d_Bd_diff);
            cudaFree(d_BB);
            if (debug){
                std::cout<<debug_buffer.str();
            }
            else {
                std::cout<<alg57_time / num_reps<<std::endl;
            }
        }

    }
    else if (op == "844"){
            double eps = 1e-12;
            double tolerance = 1e-6;
            size_t D = 2; //dimensions working with
            size_t P_size= (polynomialDegree + 1)*(polynomialDegree + 1);
            if (version == 2 && num_threads < P_size){
                num_threads = P_size;
            } 

            ValueType* d_Bd_diff;
            ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_Bd_diff, 
                sizeof(ValueType) * P_size * D) );

            ValueType* d_Bd_diff_ref;
            ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_Bd_diff_ref, 
                sizeof(ValueType) * P_size * D) );

            size_t ii = generateRandomBit();
            ValueType* d_BB;
            ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_BB, 
                sizeof(ValueType) * P_size * D) );
            if (ii == 0) {
                std::vector<ValueType> B = params.Bvector<ValueType>();
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_BB, B.data(),
                    sizeof(ValueType) * P_size * D,
                    cudaMemcpyHostToDevice));
                debug_buffer<<" d(N(s))^" << ii <<"/ds" << std::endl;

            }
            else {
                std::vector<ValueType> Bdiff = params.Bdiffvector<ValueType>();
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_BB, Bdiff.data(),
                    sizeof(ValueType) * P_size * D,
                    cudaMemcpyHostToDevice));
                debug_buffer<<" d^"<<ii<<"(N(s))/ds^"<<ii << std::endl;

            }

                        // function to test
            double alg844_time = 0.0;
            debug_buffer << "Starting alg84 merge VD "<<num_reps<<"times." << std::endl;

            for (size_t tt=0;tt<num_reps;tt++){
                TIME_FUNC(alg844_time, 
                {
                    cie::otFlyGPU::test_alg_84_mergeBd<ValueType><<<1,num_threads>>>(
                            d_Bd_diff, 
                            d_BB, 
                            ii,
                            polynomialDegree+1,
                            polynomialDegree+1,
                            D,version);
                });
                ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
                ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
            }
            //we need this data only for computing the reference value
            //copy data of Bd_diff back2HOST 
            debug_buffer << "Took time: " 
                            << alg844_time / num_reps
                            << std::endl;

            if (verify){
                std::vector<ValueType> Bd_diff(P_size * D);
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(Bd_diff.data(),d_Bd_diff,
                    sizeof(ValueType) * P_size * D,
                    cudaMemcpyDeviceToHost));

                cie::otFlyGPU::test_alg_84_mergeBd<ValueType><<<1,1>>>(
                    d_Bd_diff_ref, 
                    d_BB, 
                    ii,
                    polynomialDegree+1,
                    polynomialDegree+1,
                    D,1);
                ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
                ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
                //we need this data only for computing the reference value
                //copy data of Bd_diff back2HOST 
                std::vector<ValueType> Bd_diff_ref(P_size * D);
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(Bd_diff_ref.data(),d_Bd_diff_ref,
                    sizeof(ValueType) * P_size * D,
                    cudaMemcpyDeviceToHost));


                auto comparison = compare_vectors(Bd_diff_ref,Bd_diff,tolerance);
                if (!comparison.first) {
                    debug_buffer<< "Failed alg84 merge PRODUCT"<< std::endl;
                    debug_buffer<< "ref,cuda"<<std::endl;
                    for (int i =0;i<P_size;i++) 
                        debug_buffer<<Bd_diff_ref[i]<<" , "<<Bd_diff[i]<<std::endl;
                    debug_buffer<< "end"<<std::endl;


                } else {
                    debug_buffer << "++++++++++++ Result in alg84 merge correct!" << std::endl;
                }
                cudaFree(d_Bd_diff_ref);
            }
            debug_buffer << "-- Freeing space on device." <<std::endl;
            cie::otFlyGPU::terminate_spectral_gpu(sp);
            cudaFree(d_BB);
            cudaFree(d_Bd_diff);
            if (debug){
                std::cout<<debug_buffer.str();
            }
            else {
                std::cout<<alg844_time / num_reps<<std::endl;
            }



    }
    else if (op == "84") {

        debug_buffer << "\t--\t Starting  algorithm84..." << std::endl;
        debug_buffer << "\t--\t... using version "<<version << std::endl;

        if (device == "cuda"){

            debug_buffer << "ALG84: Using CUDA." << std::endl;

            double eps = 1e-12;
            double tolerance = 1e-6;

            size_t D = 2; //dimensions working with
            size_t P_size= (polynomialDegree + 1)*(polynomialDegree + 1);
            if (version == 2 && num_threads < P_size){
                num_threads = P_size;
            } 

            std::vector<ValueType> v = { 0.04234617, 0.9941601 , 0.39384291, 0.78736183, 0.86274709,
		            0.25537733, 0.18095811, 0.12164062, 0.25661357 };
            std::vector<ValueType> r (P_size,0.0);
            ValueType* d_v;
            ValueType* d_r;

            ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_v, 
                sizeof(ValueType) * P_size) );

            ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_r, sizeof(ValueType) * P_size ));

            ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_v, v.data(),
            sizeof(ValueType) * P_size,
            cudaMemcpyHostToDevice));

            ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_r, r.data(),
            sizeof(ValueType) * P_size,
            cudaMemcpyHostToDevice));
                

            // function to test
            double alg84_time = 0.0;
            debug_buffer << "Starting alg84 "<<num_reps<<" times." << std::endl;

            for (size_t tt=0;tt<num_reps;tt++){

                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_r, r.data(),
                sizeof(ValueType) * P_size,
                cudaMemcpyHostToDevice));
                //Bd_diff is already on the device... No need to copy H2D
                TIME_FUNC(alg84_time, 
                    {
                    cie::otFlyGPU::test_alg_84<ValueType><<<1,num_threads>>>(D, 
                                sp.d_B,
                                sp.d_Bdiff,
                                sp.d_M, 
                                d_v,
                                polynomialDegree,
                                d_r,version);
                    });

                ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
                ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
            }

            debug_buffer << "Took time: " 
                            << alg84_time / num_reps
                            << std::endl;

            if (verify){
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(r.data(),d_r,
                sizeof(ValueType) * P_size ,
                cudaMemcpyDeviceToHost));

                //compute reference data
                ValueType* d_ref;
                std::vector<ValueType> ref (P_size,0.0);
                ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_ref, sizeof(ValueType) * P_size ));
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_ref, ref.data(),
                sizeof(ValueType) * P_size,
                cudaMemcpyHostToDevice));

                cie::otFlyGPU::test_alg_84<ValueType><<<1,1>>>(D, 
                                sp.d_B,
                                sp.d_Bdiff,
                                sp.d_M, 
                                d_v,
                                polynomialDegree,
                                d_ref,1);

                //copy back data... 

                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(ref.data(),d_ref,
                sizeof(ValueType) * P_size ,
                cudaMemcpyDeviceToHost));
                auto comparison = compare_vectors(ref,r,tolerance);
                if (!comparison.first) {
                    debug_buffer<< "Failed alg84 PRODUCT"<< std::endl;
                    debug_buffer<< "ref,cuda"<<std::endl;
                    for (int i =0;i<P_size;i++) 
                        debug_buffer<<ref[i]<<" , "<<r[i]<<std::endl;
                    debug_buffer<< "end"<<std::endl;


                } else {
                    debug_buffer << "++++++++++++ Result in alg84 correct!" << std::endl;
                }
                cudaFree(d_ref);

            }
            debug_buffer << "-- Freeing space on device." <<std::endl;
            cie::otFlyGPU::terminate_spectral_gpu(sp);
            cudaFree(d_v);
            cudaFree(d_r);

            if (debug){
                std::cout<<debug_buffer.str();
            }
            else {
                std::cout<<alg84_time / num_reps<<std::endl;
            }
        }





    }
    else if (op == "82") {
    
            debug_buffer << "ALG84: Using CUDA." << std::endl;

            double eps = 1e-12;
            double tolerance = 1e-6;

            size_t D = 2; //dimensions working with
            size_t P_size= (polynomialDegree + 1)*(polynomialDegree + 1);
            if (version == 2 && num_threads < P_size){
                num_threads = P_size;
            } 


            std::vector<ValueType> P (P_size,1.0);
            ValueType* d_P;


            ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_P, sizeof(ValueType) * P_size ));

            ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_P, P.data(),
            sizeof(ValueType) * P_size,
            cudaMemcpyHostToDevice));
                

            // function to test
            double alg82_time = 0.0;
            debug_buffer << "Starting alg82 "<<num_reps<<" times." << std::endl;

            for (size_t tt=0;tt<num_reps;tt++){

                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_P, P.data(),
                sizeof(ValueType) * P_size,
                cudaMemcpyHostToDevice));
                //Bd_diff is already on the device... No need to copy H2D
                TIME_FUNC(alg82_time, 
                    {
                    cie::otFlyGPU::test_alg_82<ValueType><<<1,num_threads>>>(D, 
                                sp.d_B,
                                sp.d_G,
                                d_P, 
                                polynomialDegree,
                                version);
                    });

                ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
                ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
            }

            debug_buffer << "Took time: " 
                            << alg82_time / num_reps
                            << std::endl;

            if (verify){
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(P.data(),d_P,
                sizeof(ValueType) * P_size ,
                cudaMemcpyDeviceToHost));

                //compute reference data
                ValueType* d_ref;
                std::vector<ValueType> ref (P_size,1.0);
                ASSERT_NO_CUDA_ERRORS(cudaMalloc(&d_ref, sizeof(ValueType) * P_size ));
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_ref, ref.data(),
                sizeof(ValueType) * P_size,
                cudaMemcpyHostToDevice));

                cie::otFlyGPU::test_alg_82<ValueType><<<1,1>>>(D, 
                                sp.d_B,
                                sp.d_G,
                                d_ref, 
                                polynomialDegree,
                                1);

                //copy back data... 

                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(ref.data(),d_ref,
                sizeof(ValueType) * P_size ,
                cudaMemcpyDeviceToHost));
                auto comparison = compare_vectors(ref,P,tolerance);
                if (!comparison.first) {
                    debug_buffer<< "Failed alg82 PRODUCT"<< std::endl;
                    debug_buffer<< "ref,cuda"<<std::endl;
                    for (int i =0;i<P_size;i++) 
                        debug_buffer<<ref[i]<<" , "<<P[i]<<std::endl;
                    debug_buffer<< "end"<<std::endl;


                } else {
                    debug_buffer << "++++++++++++ Result in alg82 correct!" << std::endl;
                }
                cudaFree(d_ref);

            }
            debug_buffer << "-- Freeing space on device." <<std::endl;
            cie::otFlyGPU::terminate_spectral_gpu(sp);
            cudaFree(d_P);

            if (debug){
                std::cout<<debug_buffer.str();
            }
            else {
                std::cout<<alg82_time / num_reps<<std::endl;
            }
        }
}
