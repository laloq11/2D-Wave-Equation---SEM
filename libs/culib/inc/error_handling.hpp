#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


#define ASSERT_NO_CUDA_ERRORS(function)             \
    {                                               \
        cuda_check((function), __FILE__, __LINE__); \
    }


inline void cuda_check(cudaError_t code, const char* file, int line,
                       bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU error: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort) exit(code);
    }
}
