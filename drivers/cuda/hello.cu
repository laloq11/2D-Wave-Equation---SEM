// hello_cuda_gpu_info_fixed.cu
#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromGPU(int deviceId, int major, int minor, size_t globalMem, int multiProcessors, char* deviceName) {
    printf("Hello from GPU thread %d, block %d on device %d (%s)\n",
           threadIdx.x, blockIdx.x, deviceId, deviceName);
    printf("  Compute Capability: %d.%d\n", major, minor);
    printf("  Global Memory: %.2f GB\n", globalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors: %d\n", multiProcessors);
}

int main() {
    printf("STARTING TEST...\n");
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    int numBlocks = 1;
    int threadsPerBlock = 2;

    // Allocate memory on the device for the device name.
    char* deviceNameDevice;
    size_t deviceNameSize = strlen(props.name) + 1; // +1 for null terminator
    cudaMalloc((void**)&deviceNameDevice, deviceNameSize);
    cudaMemcpy(deviceNameDevice, props.name, deviceNameSize, cudaMemcpyHostToDevice);

    helloFromGPU<<<numBlocks, threadsPerBlock>>>(
        deviceId, props.major, props.minor, props.totalGlobalMem, props.multiProcessorCount, deviceNameDevice);

    cudaDeviceSynchronize();
    cudaFree(deviceNameDevice); // Free the device memory.
    return 0;
}