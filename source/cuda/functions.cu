#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h> 

#include "constants.h" 
#include "utils.h" 
#include "functions.cuh" 


// to avoid overhead
__device__ __forceinline__ double deltaPhi(double phi_i, double phi_j) {
    double dphi = fabs(phi_i - phi_j);

    if (dphi > M_PI) {
        
        dphi = 2.0*M_PI - dphi;

    }   

    return dphi;
}


// main kernel (each event for each block)
__global__ void processEvent(double *data) {

    // get block index
    int ev = blockIdx.x;

    if (ev > N_EVENTS)
    {
        return;
    }

    int thr_id = threadIdx.x;
    int block_dim = blockDim.x;
    int offset = ev * MAX_P; // check it

    // initialize shared data
    // TO DO

    // loop to read the event
    // check iterator
    for (int p = thr_id; p < MAX_P; p += block_dim)
    {
        double pt; // TO DO

        // fill event struct
        // TO DO

        if (pt == 0) // check
        {
            continue;
        }
        
    }
    
    __shared__ int free_particle_counter;

    if (thr_id == 0)
    {
        free_particle_counter = MAX_P; // CHANGE with actual number of particles
    }

    // wait for all the threads to finish
    __syncthreads;
    
    // loop over free particles
    while (free_particle_counter > 0)
    {
        // only one particle left: jet
        if (free_particle_counter == 1)
        {
            if (thr_id == 0)
            {
                // TO DO

                free_particle_counter--;
            }
            __syncthreads;
            continue;
            
        }

        // define minima
        // TO DO

        // loop over computing distances
        // TO DO

        // reduction to find minima
        // TO DO

        // merging
        
    }
    
}