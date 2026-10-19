#pragma once


#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#define ATTRIBUTES __host__ __device__
#define INLINE __forceinline__
#define RESTRICT __restrict__
#else
#define ATTRIBUTES
#define INLINE inline
#define RESTRICT
#endif  // defined(__CUDACC__)
