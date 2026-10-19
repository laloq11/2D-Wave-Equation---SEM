#include <cuda.h>
#include <cuda_runtime.h>

#include <CLI/CLI.hpp>
#include <stdio.h>
#include <cstring>

#include "error_handling.hpp"
#include "gpu_helpers.hpp"
#include "test_helpers.hpp"
#include "timers.hpp"
#include "types.hpp"


#include "linalg.cuh"
#include "linalg_omp.hpp"


//./ex1/roofline -n 500 -d omp -o matmul --debug 1
constexpr int default_block_size = 32; //this must be the same as in linalg.cuh

// constexpr int block_size = 128;

int main(int argc, char** argv)
{
    CLI::App app{"Operations"};
    int num_elems = 1024;
    app.add_option("-n,--num_elems", num_elems, "Number of elements");
    int num_reps = 10;
    app.add_option("--num_reps", num_reps, "Number of repetitions");
    bool verify = false;
    app.add_option("--verify", verify, "Verify the result");
    std::string device = "cuda"; //cuda , omp
    app.add_option("-d, --device", device,
                   "The device to run the operation on. ");
    std::string op = "axpy";//axpy, dot, matvec, matmul
    app.add_option("-o, --op", op, "The operation to run");
    bool debug = false;
    app.add_option("--debug", debug, "debug print statements the result");
    static int block_size = default_block_size;
    app.add_option("-b,--block_size", block_size, "Thread block size");
    CLI11_PARSE(app, argc, argv);

    std::stringstream debug_buffer;
    debug_buffer << "Running " << op 
                 << "\ninput size num_elements: (" << num_elems
                 << ", on device: " << device << std::endl;


    using ValueType = float;

    if (op == "axpy") {
        std::vector<ValueType> x(num_elems, 2.0);
        std::vector<ValueType> y(num_elems, 3.0);
        std::vector<ValueType> o_y(num_elems, 3.0);
        ValueType alpha = 2.5;
        ValueType* d_x;
        ValueType* d_y;
        ValueType* d_alpha;
        // constexpr int block_size = 128;
        const int num_blocks = (num_elems + block_size - 1) / block_size;

        if (device == "cuda") {
            // Allocate memory on device
            ASSERT_NO_CUDA_ERRORS(
                cudaMalloc(&d_x, sizeof(ValueType) * (num_elems)));
            ASSERT_NO_CUDA_ERRORS(
                cudaMalloc(&d_y, sizeof(ValueType) * (num_elems)));
            ASSERT_NO_CUDA_ERRORS(
                cudaMalloc(&d_alpha, sizeof(ValueType) * (1)));

            // Copy data to device
            ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_x, x.data(),
                                             sizeof(ValueType) * num_elems,
                                             cudaMemcpyHostToDevice));
            ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_y, y.data(),
                                             sizeof(ValueType) * num_elems,
                                             cudaMemcpyHostToDevice));
            ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_alpha, &alpha,
                                             sizeof(ValueType) * 1,
                                             cudaMemcpyHostToDevice));
        }

        double axpy_time = 0.0;
        for (int rep = 0; rep < num_reps; ++rep) {
            if (device == "cuda") {
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_y, y.data(),
                                                 sizeof(ValueType) * num_elems,
                                                 cudaMemcpyHostToDevice));
                TIME_FUNC(axpy_time, {
                    cuda::axpy<ValueType><<<num_blocks, block_size>>>(
                        d_alpha, d_x, num_elems, d_y);
                    ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
                    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
                });
            } else if (device == "omp") {
                std::memcpy(y.data(), o_y.data(),
                            sizeof(ValueType) * num_elems);
                TIME_FUNC(axpy_time, {
                    omp::axpy(&alpha, x.data(), num_elems, y.data());
                });
            }
        }
        if (debug) {std::cout << "axpy" <<device<< std::endl;}
        std::cout << axpy_time / num_reps << std::endl;

        if (verify) {
            // Copy data back from device
            if (device == "cuda") {
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(y.data(), d_y,
                                                 sizeof(ValueType) * num_elems,
                                                 cudaMemcpyDeviceToHost));
            }

            std::vector<ValueType> expected(num_elems, 8.0);
            ASSERT_TRUE(
                compare_arrays(num_elems, y.data(), expected.data()).first);
            std::cout << "Success!" << std::endl;
        }
    } 
    else if (op == "dot") {
        // TODO
        std::vector<ValueType> x(num_elems, 2.0);
        std::vector<ValueType> y(num_elems, 2.0);
        // ValueType res = 0.0;
        std::vector<ValueType> res(1, 0.0);
        // ValueType* d_x,d_y,d_alpha;
        ValueType* d_x;
        ValueType* d_y;
        ValueType* d_alpha;
        const int num_blocks = (num_elems + block_size - 1) / block_size;

        if (device == "cuda") {
            // Allocate memory on device
            // auto sizeof(ValueType) * (num_elems) = sizeof(ValueType) * (num_elems);
            ASSERT_NO_CUDA_ERRORS(
                cudaMalloc(&d_x, sizeof(ValueType) * (num_elems)));
                // [ x1,x2,...xn ]

            ASSERT_NO_CUDA_ERRORS(
                cudaMalloc(&d_y, sizeof(ValueType) * (num_elems)));
                // [ y1,y2,...yn ]
            ASSERT_NO_CUDA_ERRORS(
                cudaMalloc(&d_alpha, sizeof(ValueType) * (1)));
                //alpha only oen value

            // Copy data to device
            ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_x, x.data(),
                                             sizeof(ValueType) * (num_elems),
                                             cudaMemcpyHostToDevice));
            ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_y, y.data(),
                                             sizeof(ValueType) * (num_elems),
                                             cudaMemcpyHostToDevice));
            // ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_alpha, res.data(),
            //                                  sizeof(ValueType),
            //                                  cudaMemcpyHostToDevice));
            //no copy of alpha!
            debug_buffer << "Copied " << 2*sizeof(ValueType) * (num_elems) + sizeof(ValueType)
            << " bytes from CPU to the GPU main memory" << std::endl;
            // std::cout << debug_buffer.str();

        }

        double dot_time = 0.0;
        for (int rep = 0; rep < num_reps; ++rep) {
            if (device == "omp") {
                TIME_FUNC(dot_time, {omp::dot( x.data(), y.data(), num_elems, res.data());});
            }
            else if (device == "cuda") {

                TIME_FUNC(dot_time, {
                    cuda::dot<ValueType><<<num_blocks, block_size>>>(
                        d_x, d_y, num_elems, d_alpha);
                    ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
                    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());


                // Copy data back from device
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(res.data(), d_alpha, sizeof(ValueType) * 1,
                cudaMemcpyDeviceToHost));
                });
            }
        }

        if (device == "cuda") {
            if (debug) {
                if (verify) {
                    std::vector<ValueType> expected(1, 0.0);
                    omp::dot( x.data(), y.data(), num_elems, expected.data());;
                    bool comparison = expected[0] == res[0];
                    if (!comparison) {
                        debug_buffer<< "Failed DOT PRODUCT"<< std::endl;
                        // ASSERT_TRUE(false);
                    } else {
                        debug_buffer << "++++++++++++Result correct!" << std::endl;
                    }
                    debug_buffer<< "expected: "<<expected[0]<<", cuda:" 
                    <<res[0]<<"," <<comparison<<std::endl;
                    debug_buffer << "Took time: " << dot_time / num_reps << "s"
                                << std::endl;
                }
                std::cout << debug_buffer.str();
                } 
            else {
                std::cout << dot_time / num_reps << std::endl;
            }
        }
    } 
    else if (op == "matvec") {
        std::vector<ValueType> b(num_elems, 0.0);
        std::vector<ValueType> A(num_elems*num_elems,2.0);
        std::vector<ValueType> x(num_elems, 2.0);
        ValueType* d_x;
        ValueType* d_A;
        ValueType* d_b;
        const int num_blocks = (num_elems + block_size - 1) / block_size;
        double tolerance = 1e-6;

        if (device =="cuda"){
            ASSERT_NO_CUDA_ERRORS(
                cudaMalloc(&d_x, sizeof(ValueType) * (num_elems)));
                // [ x1,x2,...xn ]

            ASSERT_NO_CUDA_ERRORS(
                cudaMalloc(&d_A, sizeof(ValueType) * (num_elems*num_elems)));
                // [ y1,y2,...yn ]
            ASSERT_NO_CUDA_ERRORS(
                cudaMalloc(&d_b, sizeof(ValueType) * (num_elems)));

            // Copy data to device
            ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_x, x.data(),
                                             sizeof(ValueType) * (num_elems),
                                             cudaMemcpyHostToDevice));
            ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_A, A.data(),
                                             sizeof(ValueType) * (num_elems*num_elems),
                                             cudaMemcpyHostToDevice));
            ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_b, b.data(),
                                             sizeof(ValueType)* (num_elems),
                                             cudaMemcpyHostToDevice));
            debug_buffer << "Copied " << 2*sizeof(ValueType) * (num_elems) + 
            sizeof(ValueType)*(num_elems*num_elems)
            << " bytes from CPU to the GPU main memory" << std::endl;
            // std::cout << debug_buffer.str();

        }


        double matvec_time = 0.0;
        for (int rep = 0; rep < num_reps; ++rep) {
            if (device == "omp") {
                TIME_FUNC(matvec_time, {omp::matvec(num_elems, num_elems,A.data(), x.data(),b.data());});
            }
            else if (device =="cuda"){
                TIME_FUNC(matvec_time, {
                    cuda::matvec<ValueType><<<num_blocks, block_size>>>(num_elems,num_elems,
                        d_A, d_x, d_b);
                    ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
                    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());


                // Copy data back from device
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(b.data(), d_b, sizeof(ValueType) * num_elems,
                cudaMemcpyDeviceToHost));

            });
            }
        }
        std::cout << matvec_time / num_reps << std::endl;

        if (device == "cuda") {
            if (debug) {
                if (verify) {
                    std::vector<ValueType> expected(num_elems, 0.0);
                    omp::matvec(num_elems,num_elems,A.data(), x.data(), 
                    expected.data());
                    auto comparison =compare_vectors(expected,
                        b, tolerance);

                    if (!comparison.first) {
                        debug_buffer<< "Failed MATVEC PRODUCT"<< std::endl;
                        // ASSERT_TRUE(false);
                    } else {
                        debug_buffer << "++++++++++++ Result correct!" << std::endl;
                    }
                    debug_buffer<< "expected: "<<expected[0]<<", cuda:" 
                    <<b[0]<<",failed at " <<comparison.second<<std::endl;
                    debug_buffer << "Took time: " << matvec_time / num_reps << "s"
                                << std::endl;
                }
                std::cout << debug_buffer.str();
            } 
        }
    } 


    else if (op == "matmul") {
        std::vector<ValueType> A(num_elems*num_elems,2.0);
        std::vector<ValueType> B(num_elems*num_elems,2.0);
        std::vector<ValueType> C(num_elems*num_elems,0.0);
        ValueType* d_B;
        ValueType* d_A;
        ValueType* d_C;
        const int num_blocks = (num_elems + block_size - 1) / block_size;
        double tolerance = 1e-6;
        size_t shared_mem_size = 2 * block_size * block_size * sizeof(ValueType);

        if (device == "cuda") {
                    ASSERT_NO_CUDA_ERRORS(
                    cudaMalloc(&d_B, sizeof(ValueType) * (num_elems*num_elems)));
                    // [ x1,x2,...xn ]
    
                ASSERT_NO_CUDA_ERRORS(
                    cudaMalloc(&d_A, sizeof(ValueType) * (num_elems*num_elems)));
                    // [ y1,y2,...yn ]
                ASSERT_NO_CUDA_ERRORS(
                    cudaMalloc(&d_C, sizeof(ValueType) * (num_elems*num_elems)));
    
                // Copy data to device
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_B, B.data(),
                                                sizeof(ValueType) * (num_elems*num_elems),
                                                cudaMemcpyHostToDevice));
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_A, A.data(),
                                                 sizeof(ValueType) * (num_elems*num_elems),
                                                 cudaMemcpyHostToDevice));
                // ASSERT_NO_CUDA_ERRORS(cudaMemcpy(d_C, C.data(),
                //                                 sizeof(ValueType) * (num_elems*num_elems),
                //                                 cudaMemcpyHostToDevice));

                debug_buffer << "Copied " << 2*sizeof(ValueType) *(num_elems*num_elems)
                << " bytes from CPU to the GPU main memory" << std::endl;
                // std::cout << debug_buffer.str();
                  // Setup execution parameters


        }
        cudaError_t err;

        double matmul_time = 0.0;
        for (int rep = 0; rep < num_reps; ++rep) {
            if (device == "omp") {
                TIME_FUNC(matmul_time, {omp::matmul(num_elems, num_elems,A.data(), B.data(),C.data());});
            }
            else if (device == "cuda") {
                TIME_FUNC(matmul_time, {
                    cuda::matmul<ValueType><<<num_blocks, block_size,shared_mem_size>>>(num_elems,num_elems,block_size,
                        d_A, d_B, d_C);
                        err = cudaGetLastError();
                    ASSERT_NO_CUDA_ERRORS(cudaPeekAtLastError());
                    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());

                // Copy data back from device
                ASSERT_NO_CUDA_ERRORS(cudaMemcpy(C.data(), d_C, sizeof(ValueType) * num_elems*num_elems,
                cudaMemcpyDeviceToHost));
                });
                if (err != cudaSuccess) {
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                }
                cudaDeviceSynchronize();  // Ensures error detection
                            }
        }
        if (device == "cuda") {
            if (debug) {
                if (verify) {
                    std::vector<ValueType> expected(num_elems*num_elems, 0.0);
                    omp::matmul( num_elems, num_elems,
                        A.data(), B.data(), expected.data());

                    auto comparison =compare_matrices(expected,C, num_elems,tolerance);

                    if (!comparison.first) {
                        debug_buffer<< "Failed MATMUL PRODUCT"
                        <<"\n at "<<comparison.second<< std::endl;
                        // ASSERT_TRUE(false);
                    } else {
                        debug_buffer << "++++++++++++ Result correct!" << std::endl;
                    }
                    debug_buffer << "Took time: " << matmul_time / num_reps << "s"
                                << std::endl;
                }
                std::cout << debug_buffer.str();
                } 
            }
        std::cout << matmul_time / num_reps << std::endl;
    } 
    else {
        throw std::runtime_error("incorrect op");
    }

    return 0;
}
