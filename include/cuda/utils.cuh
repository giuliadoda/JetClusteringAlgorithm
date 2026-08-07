#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA error checking
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr,                                                \
                    "CUDA error at %s:%d: %s\n",                           \
                    __FILE__,                                              \
                    __LINE__,                                              \
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                            \
        }                                                                 \
    } while (0)

#endif 