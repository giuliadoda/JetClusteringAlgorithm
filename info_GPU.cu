#include <cstdio>
#include <cuda_runtime.h>

int main() {

    int device = 0; // GPU idx (0 = first GPU, only one here)

    cudaDeviceProp prop;
    
    cudaGetDeviceProperties(&prop, device);

    printf("\nGPU name: %s\n\n", prop.name);
    // printf("Compute capability: %d.%d\n", prop.major, prop.minor); // version, 7.5 means Turing architecture
    printf("Number of 32-bit registers per block: %d\n", prop.regsPerBlock);
    printf("Number of 32-bit registers per SMP: %d\n", prop.regsPerMultiprocessor);
    printf("Shared memory per block (byte): %zu\n", prop.sharedMemPerBlock);
    printf("Shared memory per SMP (byte): %zu\n", prop.sharedMemPerMultiprocessor);
    printf("Max #threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max #threads per SMP: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp size: %d\n\n", prop.warpSize);

    printf("#SMPs (multiProcessorCount): %d\n", prop.multiProcessorCount);
    int totalCudaCores = prop.multiProcessorCount * 64; // Turing: 64 CUDA core per SM
    int totalRegisters  = prop.multiProcessorCount * prop.regsPerMultiprocessor;
    int totalMaxThreads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;

    printf("CUDA cores (in total): %d\n", totalCudaCores);
    printf("Registers (in total): %d\n", totalRegisters);
    printf("Max #threads (together): %d\n\n", totalMaxThreads);

    printf("Global memory size (byte): %zu\n", prop.totalGlobalMem);
    printf("Constant memory size (byte): %zu\n", prop.totalConstMem);
    printf("L2 cache size (byte): %d\n", prop.l2CacheSize);
    printf("Persisting L2 Cache Max Size (byte): %d\n\n", prop.persistingL2CacheMaxSize);

    printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("Memory Bus Width (bit): %d\n", prop.memoryBusWidth);
    // Theoretical BW (GB/s) = 2 * memoryClockRate(KHz) * (memoryBusWidth/8) / 1e6, factor 2 because GDDR = double data rate (data transfer twice per clock cycle, rising and falling edges)
    double bw = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    printf("Theoretical BW (GB/s): %.2f\n", bw);

    printf("GPU Clock Rate (KHz): %d\n\n", prop.clockRate);

    printf("Block max dim (x,y,z): %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Grid max dim (x,y,z): %d, %d, %d\n\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");


    return 0;
}