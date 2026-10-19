#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "types.hpp"

// #define BLOCK_SIZE 16  // 16x16 threads per block

namespace cuda {


template <typename ValueType>
__global__ void axpy(const ValueType* alpha, const ValueType* x,
                     const size_type length, ValueType* y)
{
    // constexpr int block_size = 32;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        y[tid] += alpha[0] * x[tid];
    }
}


template <typename ValueType>
__global__ void dot(const ValueType* a, const ValueType* b,
                    const int num_elems, 
                    ValueType* out)
{
/*
    tid is used for shared memory indexing (temp[tid]) 
            because shared memory is per block.
    idx is used for global memory indexing (a[idx] * b[idx])
            because a and b exist in global memory across 
            all blocks.
*\

/*
    Global Memory (a, b):
    a =  [1, 2, 3, 4, | 5, 6, 7, 8]
    b =  [8, 7, 6, 5, | 4, 3, 2, 1]

    Threads:
    Block 0        |   Block 1
    -------------- |  --------------
    t0 -> 1*8 = 8  |  t0 -> 5*4 = 20
    t1 -> 2*7 = 14 |  t1 -> 6*3 = 18
    t2 -> 3*6 = 18 |  t2 -> 7*2 = 14
    t3 -> 4*5 = 20 |  t3 -> 8*1 = 8

    Shared Memory:
    Block 0 -> [8, 14, 18, 20]  --> Summed to 60
    Block 1 -> [20, 18, 14, 8]  --> Summed to 60

    Atomic Addition:
    Final out = 60 + 60 = 120
*/
    constexpr int block_size = 32;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ ValueType temp[block_size];
    temp[tid] = a[idx] * b[idx];
    __syncthreads();

    if (tid == 0) {
        ValueType res = 0.0;
        for (int i = 0; i < block_size; ++i) {
            res += temp[i];
        }
        atomicAdd(out, (ValueType)res);
    }

}

template <typename ValueType>
__global__ void dot__dyn(const ValueType* a, const ValueType* b,
                    const int num_elems, 
                    ValueType* out)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ char local_shmem[];
    ValueType* temp =
reinterpret_cast<ValueType*>(local_shmem);

    temp[tid] = a[idx] * b[idx];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i = i / 2) {
        if (tid < i) {
            temp[tid] += temp[tid + i];
            }
            __syncthreads();
        }
    if (tid == 0) {
        atomicAdd(out, temp[tid]);
    }
}


template <typename ValueType>
__global__ void matvec(const int num_rows, const int num_cols,
                       const ValueType* A, const ValueType* x, ValueType* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // constexpr int block_size = 256;
    constexpr int block_size = 32;

    __shared__ ValueType x_tile[block_size];

    if (idx < num_rows) {
        ValueType sum = 0.0;
        for (int j = 0; j < num_cols; j += block_size) {
            int tile_index = threadIdx.x;
            int global_index = j + tile_index;

            if (global_index < num_cols) {
                x_tile[tile_index] = x[global_index];
            }
            __syncthreads();

            for (int k = 0; k < block_size; ++k) {
                if(j + k < num_cols){
                    sum += A[idx * num_cols + (j+k)] * x_tile[k];
                }
            }
            __syncthreads();
        }
        b[idx] = sum;
    }
}

template <typename ValueType>
__global__ void matmul(const int num_rows, const int num_cols, const int block_size1,
                       const ValueType* A, const ValueType* B, ValueType* C)
{
    size_type row = blockIdx.y * blockDim.y + threadIdx.y;
    size_type col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ ValueType sharedMemory[];

    ValueType* Asub = sharedMemory;
    ValueType* Bsub = Asub + block_size1 * block_size1;


    ValueType sum = 0.0;  // Accumulator for the dot product

    for (int t = 0; t < (num_cols + block_size1 - 1) / block_size1; ++t) {
        // Load tiles into shared memory
        if (row < num_rows && (t * block_size1 + threadIdx.x) < num_cols) 
            Asub[threadIdx.y*block_size1+threadIdx.x] = A[row * num_cols + t * block_size1 + threadIdx.x];
        else 
            Asub[threadIdx.y*block_size1+threadIdx.x] = 0.0;

        if (col < num_rows && (t * block_size1 + threadIdx.y) < num_cols) 
            Bsub[threadIdx.y*block_size1+threadIdx.x] = B[(t * block_size1 + threadIdx.y) * num_rows + col];
        else 
            Bsub[threadIdx.y*block_size1+threadIdx.x] = 0.0;

        __syncthreads(); // Ensure all threads have loaded their tile

        // Compute partial sum for this tile
        for (int k = 0; k < block_size1; ++k)
            sum += Asub[threadIdx.y*block_size1+k] * Bsub[k*block_size1+threadIdx.x];

        __syncthreads(); // Ensure all threads finish computing before loading next tile
    }
    if (row < num_rows && col < num_rows) C[row * num_rows + col] = sum;
}

}  // namespace cuda
