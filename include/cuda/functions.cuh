#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include "utils.h"

__global__ void processEvent(
    const double* __restrict__ data,
    int* __restrict__ clusters_trace,
    float* __restrict__ times
);

#endif