#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <hdf5.h>        // TO BE INSTALLED

#define MAX_P 700
#define N_FEAT 3
#define N_COLS (MAX_P * N_FEAT)
#define DIM 2

#define R 0.4

#define FILE_PATH "datafile.h5"
#define DATA_PATH "/dataframe/data"     // substitute with output of h5dump -H datafile.h5

// particle struct
typedef struct Particle
{
    int id;

    bool inCluster;

    float p_t, eta, phi;

};

// cluster struct
typedef struct Cluster
{
    int id;

    int n_particles;

    int part_idxs[MAX_P];

    float p_t, eta, phi;

};

// event struct
typedef struct Event 
{
    int id;
    int n_particles;

    Particle particles[MAX_P];

    Cluster clusters[MAX_P]; 

};

// function to read each event (to start with)
void read_event(float *raw_data, Event *ev, int id) {

    ev->n_particles = 0;

    for (int p = 0; p < MAX_P; ++p) {

        int idx = N_FEAT*p;

        float p_t = raw_data[idx];

        if (p_t == 0.0) break;

        ev->id = id;

        ev->n_particles++;

        ev->particles[p].id = p;

        ev->particles[p].p_t = p_t;
        ev->particles[p].eta = raw_data[idx+1];
        ev->particles[p].phi = raw_data[idx+2];

        ev->particles[p].inCluster = false;

    }

}

// function to compute distance between particles 
float compute_distances(Event *ev, int i, int j) {

    Particle particle_i = ev->particles[i];
    Particle particle_j = ev->particles[j];

    float p_i = particle_i.p_t;
    float p_j = particle_j.p_t;

    float eta_i = particle_i.eta;
    float phi_i = particle_i.phi;

    float eta_j = particle_j.eta;
    float phi_j = particle_j.phi;

    float diff_eta = eta_i - eta_j;
    float diff_phi = phi_i - phi_j; // ask how phi is defined

    float deltaR2 = diff_eta*diff_eta + diff_phi*diff_phi;

    float factR = deltaR2/(R*R);

    float p2;

    float p_i2 = 1./(p_i*p_i);
    float p_j2 = 1./(p_j*p_j);

    if ( p_i2 > p_j2)
    {
        p2 = p_j2;
    } else {
        p2 = p_i2;
    }
    
    float distance = p2 * factR;

    return distance;
}

// ----------- MAIN -------------

int main() {

    // get file identifier first 
    // H5F_ACC_RDONLY -> read only  
    // H5P_DEFAULT -> use the default behavior of the library
    hid_t file_id = H5Fopen(FILE_PATH, H5F_ACC_RDONLY, H5P_DEFAULT);

    // get dataset identifier
    hid_t dset_id = H5Dopen2(file_id, DATA_PATH, H5P_DEFAULT);

    // get identifier for a copy of the dataspace for a dataset 
    hid_t space_id = H5Dget_space(dset_id); 

    // read only part of the dataset (to start with and maybe it won't fit in memory)
    int start_row = 0;
    int start_col = 0;
    hsize_t n_read = 100;
    hsize_t offset[DIM] = {start_row, start_col};         // starting row and column
    hsize_t count[DIM]  = {n_read, N_COLS};    // ending point

    // get the corresponding portion of the dataset
    H5Sselect_hyperslab(
        space_id,           // dataspace identifier
        H5S_SELECT_SET,     // operation to perform on current selection
        offset,             // offset of start
        NULL,               // stride
        count,              // number of blocks included in hyperslab
        NULL                // size of block in hyperslab
    );

    // create dataspace before reading (it creates a new simple dataspace and opens it for access)
    hid_t memspace = H5Screate_simple(
        DIM,            // number of dimensions of dataspace
        count,          // array specifying the size of each dimension
        NULL            // maximum size of each dimension
    );

    // allocate memory
    float *data = malloc(n_read * N_COLS * sizeof(float));

    // read data -> maybe read in chunks
    H5Dread(
        dset_id,                // dataset identifier
        H5T_NATIVE_FLOAT,       // memory datatype identifier
        memspace,               // memory dataspace identifier
        space_id,               // file dataspace containing the selected hyperslab
        H5P_DEFAULT,            // identifier of a transfer property list
        data                    // buffer to receive data read from file
    );

    // loop over events
    for (int ev = 0; ev < n_read; ++ev) {

        Event event;

        read_event(&data[ev*N_COLS], &event, ev);

        printf("\nEvent number %d\n", event.id);

        int n_part = event.n_particles;
        printf("Number of particles %d\n", n_part);

        // compute particle clusters

        int part_counter = n_part;

        // stop merging when there are no more particles
        while (part_counter != 0)
        {
            for (int i = 0; i < n_part; ++i) {

                float min_d_i;
                int min_idx = 0;

                // i <-> j symmetry, i and j must be different 
                for (int j = i+1; j < n_part; ++j) { // check if ++i and i++ make a difference
                    
                    float d = compute_distances(&event, i, j);

                    if (j == 1)
                    {
                        min_d_i = d;
                        min_idx = j;
                    }
                    else if (min_d_i > d)
                    {
                        min_d_i = d;
                        min_idx = j;
                    }
                    
                }

                // if d_iB < min d_ij --> jet
                // update cluster quantities

                float p_ti = event->particles[i].p_t;
                // distance from beam
                float d_iB = 1./(p_ti*p_ti);

                if (d_iB >= min_d_i) {

                    // update particle quantities
                    float p_tj = event->particles[min_idx].p_t;

                    float eta_i = event->particles[i].eta;
                    float eta_j = event->particles[min_idx].eta;

                    float phi_i = event->particles[i].phi;
                    float phi_j = event->particles[min_idx].phi;

                    float new_p = p_ti + p_tj;
                    event.particles[i]->p_t = new_p;
                    event.particles[i]->eta = (p_ti*eta_i+p_tj*eta_j)/(new_p);
                    event.particles[i]->phi = (p_ti*phi_i+p_tj*eta_j)/(new_p);

                    event.particles[i]->id = i;

                    part_counter = part_counter - 1;

                }
                else // the particle is a jet on its own
                {

                    part_counter = part_counter - 1;

                    event.particles[i]->id = i;

                }
            
        }
        }

        // compute how many particles in each cluster

        // free event memory (?)

    }

    // free memory and close file
    free(data);

    H5Sclose(memspace); // maybe here not in the right order (?)
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);

    return 0;
}