#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <hdf5.h>  

#include "constants.h" 
#include "utils.h" 
#include "functions.cuh" 

#define N_BLOCKS 32 // maybe not ok
#define N_THREADS_PER_BLOCK 64 // maybe not ok


__device__ void readEvent(double *raw_data, Event *ev, int id, int n_part) {

    // loop over particles only

    // get thread idx
    int thr_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (thr_idx < n_part)
    {
        int idx = N_FEAT*thr_idx;

        double p_t = raw_data[idx];

        ev->particles[p].id = thr_idx;
        ev->free_particles[p] = thr_idx;

        double dB = 1./(p_t*p_t);

        ev->particles[p].p_t = p_t;
        ev->particles[p].d_B = dB;
        ev->particles[p].eta = raw_data[idx+1];
        ev->particles[p].phi = raw_data[idx+2];

        ev->particles[p].components = malloc(n_part*sizeof(int));
        ev->particles[p].components[0] = thr_idx;

        ev->particles[p].n_components = 1;

        ev->particles[p].isJet = false;
    }

}


__device__ double computeDistance(Event *ev, int i, int j) {

    Particle *particle_i = &ev->particles[i]; 
    Particle *particle_j = &ev->particles[j];

    double eta_i = particle_i->eta;
    double phi_i = particle_i->phi;

    double eta_j = particle_j->eta;
    double phi_j = particle_j->phi;

    double diff_eta = eta_i - eta_j;
    double diff_phi = fabs(phi_i - phi_j); 
    
    // recall that phi is periodic
    if (diff_phi > M_PI)
    {
        diff_phi = 2.*M_PI - diff_phi;
    }

    double deltaR2 = diff_eta*diff_eta + diff_phi*diff_phi;

    double factR = deltaR2/(R*R);

    double p2;

    double p_i2 = particle_i->d_B; // maybe it's better to compute them again on the fly instead of access the memory (?)
    double p_j2 = particle_j->d_B;

    if ( p_i2 > p_j2)
    {
        p2 = p_j2;
    } else {
        p2 = p_i2;
    }
    
    double distance = p2 * factR;

    return distance;

}


__device__ void findMinimumDistance(int iterator, int free_particle_i, int particle_counter, Event event, int min_d_ij, int id_i, int id_j, int idx_i, int idx_j) {

    // get thread idx 
    int thr_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (thr_idx >= iterator+1 && thr_idx < particle_counter)
    {
        // get particle-j ID
        int free_particle_j = event.free_particles[thr_idx];
        
        double current_d_ij = computeDistance(&event, free_particle_i, free_particle_j);

        // update finding minimum distance ij
        if (current_d_ij < min_d_ij)
        {
            min_d_ij = current_d_ij;

            id_i = free_particle_i;
            id_j = free_particle_j;

            idx_i = iterator;
            idx_j = thr_idx;
        }
    }

}


__device__ void findMinimum(Event event, int particle_counter, double min_d_ij, int id_i, int id_j, int idx_i, int idx_j, double min_d_iB, int id_iB, int idx_iB) {

    // get thread idx
    int thr_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (thr_idx < particle_counter)
    {
        // get particle-i ID (that is the idx for particles array)
        int free_particle_i = event.free_particles[thr_idx];

        // compute distances and update minimum
        // possible race condition here
        // maybe change N_BLOCKS and N_THREAD_PER_BLOCKS here
        findMinimumDistance<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(thr_idx, free_particle_i, particle_counter, event, min_d_ij, id_i, id_j, idx_i, idx_j);

        // wait for all child threads to finish
        cudaDeviceSynchronize();

        // beam distance
        double current_d_iB = event.particles[free_particle_i].d_B;

        // update finding minimum distance iB
        if (current_d_iB < min_d_iB)
        {
            min_d_iB = current_d_iB;

            id_iB = free_particle_i;
            idx_iB = thr_idx;
        }
    }

}


__global__ void eventProcess(double *data, int ev, double *times, hid_t fout) {

    // get thread idx 
    int thr_idx = threadIdx.x + blockIdx.x * blockDim.x;

    // compute event time
    cudaEvent_t start_t_ev, end_t_ev, t_event; 

    start_t_ev = omp_get_wtime();

    Event event;

    int n_part = 0;

    // first find number of particle sequentially
    while (n_part < MAX_P && raw_data[N_FEAT*n_part] != 0.0f)
    {
        n_part++;
    }

    event.n_clusters = 0;
    event.id = ev;
    event.n_particles = n_part;

    event.free_particles = malloc(n_part*sizeof(int)); // check
    event.particles = malloc(n_part*sizeof(Particle)); // check

    // check memory allocation
    if (event.particles == NULL || event.free_particles == NULL) {
        fprintf(stderr, "Memory allocation failure, event %d\n", ev);
        exit(1);
    }
    
    readEvent<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(&data[ev*N_COLS], &event, ev, n_part);

    // free particles counter
    int particle_counter = n_part;

    // loop over particles
    while (particle_counter > 0)
    {
        // only one particle left --> jet
        if (particle_counter == 1)
        {
            int idx_1 = event.free_particles[0];
            event.particles[idx_1].isJet = true;
            event.n_clusters++;
            break;
        }

        // compute distances for every i different from j and beam, and find minimum distances
        double min_d_ij = D; 
        int id_i = -1;
        int id_j = -1;     // particles IDs
        int idx_i = -1;
        int idx_j = -1;   // free particles indexes

        double min_d_iB = D;
        int id_iB = -1;
        int idx_iB = -1;

        // race condition here
        // maybe change blocks and threads per blocks here
        findMinimum<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(min_d_ij, id_i, id_j, idx_i, idx_j, min_d_iB, id_iB, idx_iB);

        // wait for all the threads to finish
        cudaDeviceSynchronize();

        // merge: if d_iB_min < d_ij_min, jet; else new "particle"
        if (min_d_iB < min_d_ij)
        {
            // printf("Minimum distance from the beam: %.10e\n", min_d_iB);
            event.free_particles[idx_iB] = event.free_particles[particle_counter-1];
            event.n_clusters++;
            event.particles[id_iB].isJet = true;

        }
        else 
        {
            // printf("Minimum distance between particles: %.10e\n", min_d_ij);
            particle_update(&event, id_i, id_j);
            event.free_particles[idx_j] = event.free_particles[particle_counter-1];
        }

        particle_counter--;

    }
    


}