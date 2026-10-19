#pragma once

#include <cuda.h>
// #include <cuda_runtime.h>


// void print_gpu_specs(){
//     int nDevices;
//     cudaGetDeviceCount(&nDevices);
//     for (int i = 0; i < nDevices; i++) {
//         cudaDeviceProp prop;
//         cudaGetDeviceProperties(&prop, i);
//         printf("Device Number: %d\n", i);
//         printf("  Device name: %s\n", prop.name);
//         printf("  Total Global memory (GB): %f\n",
//                (prop.totalGlobalMem) / 1.0e9);
//         printf("  Peak Memory Bandwidth (GB/s): %f\n",
//                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
//         printf("  L2 cache size (MB): %f\n", (prop.l2CacheSize) / 1.0e6);
//         printf("  Max Shared Memory Per thread block (KB): %f\n",
//                (prop.sharedMemPerBlock) / 1.0e3);
//         printf("  Total Const Memory on device (KB): %f\n",
//                (prop.totalConstMem) / 1.0e3);
//         printf("  Number of 32-bit registers per thread block: %d\n",
//                prop.regsPerBlock);
//         printf("  Max Shared Memory Per Multiprocessor (KB): %f\n",
//                (prop.sharedMemPerMultiprocessor) / 1.0e3);
//         printf("  Number of 32-bit registers per Multiprocessor: %d\n",
//                prop.regsPerMultiprocessor);
//         printf("  Number of SMs: %d\n", prop.multiProcessorCount);
//         printf("  Max Number of threads per thread block: %d\n\n",
//                prop.maxThreadsPerBlock);
//     }
// }
