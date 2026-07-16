#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <hdf5.h>           // TO BE INSTALLED

#include "include/constants.h" 
#include "include/openmp/utils.h"  

// function to read each event 
void read_event(double *raw_data, Event *ev, int id) {

    int n_part = 0;

    // first find number of particle sequentially
    while (n_part < MAX_P && raw_data[N_FEAT*n_part] != 0.0f)
    {
        n_part++;
    }

    ev->n_clusters = 0;
    ev->id = id;
    ev->n_particles = n_part;

    ev->free_particles = malloc(n_part*sizeof(int));
    ev->particles = malloc(n_part*sizeof(Particle));

    // check memory allocation
    if (ev->particles == NULL || ev->free_particles == NULL) {
        fprintf(stderr, "Memory allocation failure, event %d\n", id);
        exit(1);
    }
    
    // maybe this is not worth parallelizing since operations are not heavy (per thread)
    #pragma omp parallel for
    // index p is private to each thread by default
    // local copy of n_part, then reduced to a single value and combined with the original global value
    for (int p = 0; p < n_part; ++p) {

        int idx = N_FEAT*p;

        double p_t = raw_data[idx];

        ev->particles[p].id = p;
        ev->free_particles[p] = p;

        double dB = 1./(p_t*p_t);

        ev->particles[p].p_t = p_t;
        ev->particles[p].d_B = dB;
        ev->particles[p].eta = raw_data[idx+1];
        ev->particles[p].phi = raw_data[idx+2];

        ev->particles[p].components = malloc(n_part*sizeof(int));
        ev->particles[p].components[0] = p;

        ev->particles[p].n_components = 1;

        ev->particles[p].isJet = false;

    }

}

// function to compute distance between particles 
double compute_distance_ij(Event *ev, int i, int j) {

    Particle *particle_i = &ev->particles[i]; 
    Particle *particle_j = &ev->particles[j];

    double p_i = particle_i->p_t;
    double p_j = particle_j->p_t;

    double eta_i = particle_i->eta;
    double phi_i = particle_i->phi;

    double eta_j = particle_j->eta;
    double phi_j = particle_j->phi;

    double diff_eta = eta_i - eta_j;
    double diff_phi = fabs(phi_i - phi_j); 
    
    // recall that phi is periodic
    if (diff_phi > M_PI)
    {
        diff_phi = fabs(diff_phi-2.*M_PI);
    }

    if (diff_phi < -M_PI)
    {
        diff_phi = fabs(diff_phi+2.*M_PI);
    }

    double deltaR2 = diff_eta*diff_eta + diff_phi*diff_phi;

    double factR = deltaR2/(R*R);

    double p2;

    double p_i2 = particle_i->d_B; // maybe it's better to compute it again on the fly instead of access the memory
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

// function to compute new quantities for a pseudocluster
void particle_update(Event *ev, int i, int j) {

    Particle *part_i = &ev->particles[i];
    Particle *part_j = &ev->particles[j];

    int n_comp_i = part_i->n_components;
    int n_comp_j = part_j->n_components;

    int n_comp_new = n_comp_i + n_comp_j;
    
    part_i->n_components = n_comp_new;

    double p_t_i = part_i->p_t;
    double eta_i = part_i->eta;
    double phi_i = part_i->phi;

    double p_t_j = part_j->p_t;
    double eta_j = part_j->eta;
    double phi_j = part_j->phi;

    double p_t_new = p_t_i + p_t_j;

    double dB_new = 1./(p_t_new*p_t_new);

    double eta_new = (p_t_i*eta_i + p_t_j*eta_j)/p_t_new;
    double phi_new = (p_t_i*phi_i + p_t_j*phi_j)/p_t_new;

    // recal that phi is periodic
    if (phi_new > M_PI)
    {
        phi_new = fabs(phi_new-2.*M_PI);
    }

    if (phi_new < -M_PI)
    {
        phi_new = fabs(phi_new+2.*M_PI);
    }

    part_i->p_t = p_t_new;
    part_i->eta = eta_new;
    part_i->phi = phi_new;
    part_i->d_B = dB_new;

    for (int k = 0; k < n_comp_j; ++k)
    {
        part_i->components[n_comp_i+k] = part_j->components[k];
    }

    // free memory
    free(part_j->components);
    part_j->components = NULL;

}

// free event memory 
void event_free(Event *ev) {
    for (int p = 0; p < ev->n_particles; ++p) {
        if (ev->particles[p].components != NULL) {
            free(ev->particles[p].components);
            ev->particles[p].components = NULL;
        }
    }
    free(ev->particles);
    free(ev->free_particles);
}

// function to loop over events --> output file management
void process_event() {


}

#endif