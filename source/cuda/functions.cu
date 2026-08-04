#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h> 

#include "constants.h" 
#include "utils.h" 
#include "functions.cuh" 


// to ocmpute difference between phi angle when computing distance
// __forceinline__ to avoid overhead basically
__device__ __forceinline__ double deltaPhi(double phi_i, double phi_j) {
    
    double dphi = fabs(phi_i - phi_j);

    if (dphi > M_PI) {
        
        dphi = 2.0*M_PI - dphi;

    }   

    return dphi;
}


// main kernel (each event for each block)
// cluster_trace for each event is something like: [0, 0, 5, 3, 3, 7, ...] meaning that particles 1 has been merged into particle 0, particle 2 into particle 5, particle 4 into 3, 5 into 7
__global__ void processEvent(
    const double* __restrict__ data,
    int* __restrict__ clusters_trace // ID of final particle each particle (<--> array position) has been merged to 
) {

    // get block index
    int ev = blockIdx.x;

    if (ev > N_EVENTS)
    {
        return;
    }

    // get thread index etc
    int thr_id = threadIdx.x;
    int block_dim = blockDim.x; // number of threads

    int event_offset = ev * N_COLS;
    int particle_offset = ev * MAX_P; // check it, maybe define it after getting n_part is better

    // initialize shared data (among all threads, limited to block)
    // to store kinematics
    __shared__ double pt_s[MAX_P];
    __shared__ double eta_s[MAX_P];
    __shared__ double phi_s[MAX_P];
    __shared__ double dB_s[MAX_P];

    // to keep track of clusterized particles
    __shared__ int free_p_s[MAX_P]; // maybe initialize it after getting n_part
    __shared__ int free_particle_counter;

    if (thr_id == 0)
    {
        free_particle_counter = 0;
    }
    __syncthreads;


    // loop to read the event
    // check iterator
    for (int p = thr_id; p < MAX_P; p += block_dim)
    {

        // at the beginning each particle belongs to itself
        clusters_trace[particle_offset+p] = p;

        // get momentum
        // + 0 just for clarity (momentum is the first kinematic value)
        double p_t = data[event_offset + N_FEAT*p + 0]; 

        if (p_t != 0.0)
        {
            pt_s[p] = p_t;
            eta_s[p] = data[event_offset + N_FEAT*p + 1];
            phi_s[p] = data[event_offset + N_FEAT*p + 2];

            dB_s[p] = 1.0/(p_t*p_t);

            // avoiding data race to update the number of particle of the event
            int free_particle_id = atomicAdd(&free_particle_counter, 1);

            // fill array to keep track of the particles
            free_p_s[free_particle_id] = p;

         }
        
    }

    // wait for all the threads to finish
    __syncthreads;

    // to handle minima 
    // shared memory to keep results of all threads working on different particles over the event
    // a kernel can have only one dynamically allocated shared array
    // so we have to manually partition a big array
    extern __shared__ char dyn[]; // extern specifier to declare a variable that will be allocated at kernel launch --> WHY CHAR??
    double *s_d_ij = (double*) dyn;
    int *s_idx_i = (int*)(s_d_ij + block_dim);
    int *s_idx_j = (int*)(s_idx_i + block_dim); 

    double *s_d_iB = (double*)(s_idx_j + block_dim);
    int *s_idx_iB = (int*)(s_d_iB + block_dim);
    
    // loop over free particles
    while (free_particle_counter > 0)
    {
        // only one particle left: jet
        if (free_particle_counter == 1)
        {
            if (thr_id == 0)
            {
                // TO DO (?)

                free_particle_counter--;
            }
            __syncthreads;
            continue;
            
        }

        // define minima for this iteration
        double min_d_ij = D;
        int min_idx_i = -1;
        int min_ixd_j = -1;

        double min_d_iB = D;
        int min_idx_iB = -1;

        // loop over computing distances and updating minima
        // thread stride
        for (int i = thr_id; i < free_particle_counter; i+=block_dim)
        {
            int free_particle_i = free_p_s[i];

            // beam distance
            double dB = s_d_iB[free_particle_i];
            if (dB < min_d_iB)
            {
                min_d_iB = dB;
                min_idx_iB = i;
            }

            // particle distance
            for (int j = i+1; i < free_particle_counter; ++j)
            {
                int free_particle_j = free_p_s[j];

                double phi_diff = deltaPhi(phi_s[free_particle_i], phi_s[free_particle_j]);
                double eta_diff = eta_s[free_particle_i] - eta_s[free_particle_j];

                double deltaR2 = eta_diff*eta_diff + phi_diff*phi_diff;
                double factR = deltaR2/(R*R);

                double p2 = fmin(dB, s_d_iB[free_particle_j]);

                double current_dist_ij = p2*factR;

                if (current_dist_ij < min_d_ij)
                {
                    min_d_ij = current_dist_ij;
                    min_idx_i = i;
                    min_ixd_j = j;
                }
                
            }
            
        }
        
        // track each minima found by each thread
        s_d_ij[thr_id] = min_d_ij;
        s_idx_i[thr_id] = min_idx_i;
        s_idx_j[thr_id] = min_ixd_j;

        s_d_iB[thr_id] = min_d_iB;
        s_idx_iB[thr_id] = min_idx_iB;

        __syncthreads();

        // reduction to find minima
        // k >>= 1 means shift bits to the right (divide by 2)
        for (int k = block_dim/2; k > 0; k >>= 1)
        {
            if (thr_id < k)
            {
                if (s_d_ij[thr_id+k] < s_d_ij[thr_id])
                {
                    s_d_ij[thr_id] = s_d_ij[thr_id+k];

                    s_idx_i[thr_id] = s_idx_i[thr_id+k];
                    s_idx_j[thr_id] = s_idx_j[thr_id+k];
                }

                if (s_d_iB[thr_id+k] < s_d_iB[thr_id])
                {
                    s_d_iB[thr_id] = s_d_iB[thr_id+k];

                    s_idx_iB[thr_id] = s_idx_iB[thr_id+k];
                }
                
            }
            
        }

        __syncthreads();
        
        // merging
        
    }
    
}