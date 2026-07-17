#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <hdf5.h>  
#include <omp.h>         

#include "constants.h" 
#include "utils.h" 
#include "functions.h"  

// function to read each event 
static void read_event(double *raw_data, Event *ev, int id) {

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
    
    // loop over particles
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
static double compute_distance_ij(Event *ev, int i, int j) {

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
        diff_phi = 2.*M_1_PI - diff_phi;
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

// function to compute new quantities for a pseudocluster
static void particle_update(Event *ev, int i, int j) {

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

    // recall that phi is periodic --> CHECK HERE MAYBE
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
static void event_free(Event *ev) {

    if (ev == NULL)
        return;

    if (ev->particles != NULL)
    {
        for (int p = 0; p < ev->n_particles; ++p)
        {
            free(ev->particles[p].components);
            ev->particles[p].components = NULL;
        }

        free(ev->particles);
        ev->particles = NULL;
    }

    free(ev->free_particles);
    ev->free_particles = NULL;

    ev->n_particles = 0;
    ev->n_clusters = 0;
    
}

// write cluster to file (return type: error return type, int, 0 if success, 1 if failure)
static herr_t write_cluster(hid_t event_group, Particle *cluster, int cluster_ID) {

    char cluster_name[64];
    
    snprintf(cluster_name, sizeof(cluster_name), "cluster_%d", cluster_ID); 

    // creating gluster group
    hid_t cluster_group = H5Gcreate2(
        event_group,
        cluster_name,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT
    );

    if (cluster_group < 0)
    {
        fprintf(stderr, "Cannot create cluster group %s\n", cluster_name);
        return -1;
    }

    // id to check writing to file status 
    herr_t status = 0;

    // saving particles ID (components)
    hsize_t dims[1] = { (hsize_t) cluster->n_components };

    hid_t space = H5Screate_simple(
        1, 
        dims, 
        NULL
    );

    hid_t dset = H5Dcreate2(
        cluster_group,
        "components",
        H5T_NATIVE_INT,
        space,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT
    );

    if (dset < 0 || space < 0)
    {
        status = -1;
    }
    else
    {
        status |= H5Dwrite(     // status = 1 only if all we are writing to cluster_group is ok
            dset,
            H5T_NATIVE_INT,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            cluster->components
        );
    }

    if (dset >= 0)
    {
        H5Dclose(dset);
    }

    if (space >= 0)
    {
        H5Sclose(space);
    }
    
    // saving cluster ID
    hid_t scalar = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(
        cluster_group,
        "cluster_id",
        H5T_NATIVE_INT,
        scalar,
        H5P_DEFAULT,
        H5P_DEFAULT
    );

    if (scalar < 0 || attr < 0)
    {
        status = -1;
    }
    else
    {
        status |= H5Awrite(
            attr,
            H5T_NATIVE_INT,
            &cluster_ID
        );
    }

    if (attr >= 0)
    {
        H5Aclose(attr);
    }

    if (scalar >= 0)
    {
        H5Sclose(scalar);
    }

    // saving jet kinematics
    double kin[3] = { cluster->p_t, cluster->eta, cluster->phi};
    hsize_t kdims[1] = {3};
    hid_t kspace = H5Screate_simple(1, kdims, NULL);
    hid_t kdset = H5Dcreate2(
        cluster_group,
        "kinematics",
        H5T_NATIVE_DOUBLE,
        kspace,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT
    );

    if (kspace < 0 || kdset < 0)
    {
        status = -1;
    }
    else
    {
        status |= H5Dwrite(
            kdset,
            H5T_NATIVE_DOUBLE,
            H5S_ALL, 
            H5S_ALL,
            H5P_DEFAULT,
            kin
        );
    }
    
    if (kdset >= 0)
    {
        H5Dclose(kdset);
    }
    
    if (kspace >= 0)
    {
        H5Sclose(kspace);
    }

    status |= H5Gclose(cluster_group);
    
    return status;
    
}

// save event to file
static herr_t save_event(hid_t fout, Event *event) {

    char event_name[64];

    snprintf(event_name, sizeof(event_name), "/event_%d", event->id);

    // creating event group
    hid_t event_group = H5Gcreate2(
        fout,
        event_name,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT
    );

    if (event_group < 0)
    {
        fprintf(stderr, "Cannot create event group for %s\n", event_name);
        return -1;
    }

    int cluster_counter = 0;
    herr_t event_status = 0;

    for (int p = 0; p < event->n_particles; ++p)
    {
        Particle *cluster = &event->particles[p];
        if (cluster->isJet)
        {
            event_status |= write_cluster(event_group, cluster, cluster_counter);
            cluster_counter++;

        }
        
    }
    
    event_status |= H5Gclose(event_group);

    return event_status;
    
}

// function to loop over events 
void process_single_event(double *data, int ev, double *times, hid_t fout) {

    Event event;

    // compute event time
    double start_t_ev, end_t_ev, t_event; 

    start_t_ev = omp_get_wtime();

    read_event(&data[ev*N_COLS], &event, ev); 

    int n_part = event.n_particles;

    // free particles counter
    int particle_counter = n_part;

    // loop over event particles
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
        
        for (int i = 0; i < particle_counter; ++i)
        {
            // get particle-i ID (that is the idx for particles array)
            int free_particle_i = event.free_particles[i];

            for (int j = i+1; j < particle_counter; ++j) 
            {
                // get particle-j ID
                int free_particle_j = event.free_particles[j];
                
                double current_d_ij = compute_distance_ij(&event, free_particle_i, free_particle_j);

                // update finding minimum distance ij
                if (current_d_ij < min_d_ij)
                {
                    min_d_ij = current_d_ij;

                    id_i = free_particle_i;
                    id_j = free_particle_j;

                    idx_i = i;
                    idx_j = j;
                }
        
            }

            // beam distance
            double current_d_iB = event.particles[free_particle_i].d_B;

            // update finding minimum distance iB
            if (current_d_iB < min_d_iB)
            {
                min_d_iB = current_d_iB;

                id_iB = free_particle_i;
                idx_iB = i;
            }
            
        } 

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

    // save event clusters
    save_event(fout, &event);

    // free event memory
    event_free(&event);

    end_t_ev = omp_get_wtime();

    t_event = end_t_ev - start_t_ev;

    times[ev] = t_event;

}