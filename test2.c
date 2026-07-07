#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <hdf5.h>        // TO BE INSTALLED

#define MAX_P 700
#define N_FEAT 3
#define N_COLS (MAX_P * N_FEAT)
#define DIM 2

#define R 0.4
#define D 1000. // maybe change it after inspecting distance actual values

#define FILE_PATH "datafile.h5"
#define DATA_PATH "/dataframe/data"     // substitute with output of h5dump -H datafile.h5

// particle-pseudocluster struct
typedef struct Particle
{
    int id;

    int components[MAX_P]; // to keep track of the particles belonging here

    int n_components; // how many particles in the cluster

    bool isCluster;

    float p_t, eta, phi;

};

// event struct
typedef struct Event 
{
    int id;
    int n_particles;

    int n_clusters;

    int free_particles[MAX_P]; // to keep track of the particles that are not yet assigned to a cluster

    Particle particles[MAX_P]; // will become clusters in the end

};

// function to read each event (to start with)
void read_event(float *raw_data, Event *ev, int id) {

    ev->n_particles = 0;
    ev->n_clusters = 0;

    for (int p = 0; p < MAX_P; ++p) {

        int idx = N_FEAT*p;

        float p_t = raw_data[idx];

        if (p_t == 0.0) break;

        ev->id = id;

        ev->n_particles++;

        ev->particles[p].id = p;
        ev->free_particles[p] = p;

        ev->particles[p].p_t = p_t;
        ev->particles[p].eta = raw_data[idx+1];
        ev->particles[p].phi = raw_data[idx+2];

        ev->particles[p].components[0] = p;

        ev->particles[p].n_components = 1;

        ev->particles[p].isCluster = false;

    }

}

// function to compute distance between particles 
float compute_distance_ij(Event *ev, int i, int j) {

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

// function to compute distance between particles and the beam
float compute_distance_iB(Event *ev, int i) {

    Particle particle_i = ev->particles[i];

    float p_i = particle_i.p_t;

    float d_iB = 1./(p_i*p_i);

    return d_iB;
}

// function to compute new quantities for a pseudocluster
void particle_update(Event *ev, int i, int j) {

    Particle *part_i = &ev->particles[i];
    Particle *part_j = &ev->particles[j];

    int n_comp_i = part_i->n_components;
    int n_comp_j = part_j->n_components;

    int n_comp_new = n_comp_i + n_comp_j;

    for (int k = 0; k < n_comp_j; ++k)
    {
        part_i->components[n_comp_i+k] = part_j->components[k];
    }
    
    part_i->n_components = n_comp_new;

    float p_t_i = part_i->p_t;
    float eta_i = part_i->eta;
    float phi_i = part_i->phi;

    float p_t_j = part_j->p_t;
    float eta_j = part_j->eta;
    float phi_j = part_j->phi;

    float p_t_new = p_t_i + p_t_j;
    float eta_new = (p_t_i*eta_i + p_t_j*eta_j)/p_t_new;
    float phi_new = (p_t_i*phi_i + p_t_j*phi_j)/p_t_new;

    part_i->p_t = p_t_new;
    part_i->eta = eta_new;
    part_i->phi = phi_new;

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

        int particle_counter = n_part;

        float *distances_ij = malloc(n_part * n_part * sizeof(float));

        float *distances_iB = malloc(n_part * sizeof(float));

        while (particle_counter > 0)
        {    
            // 0. Only one particle left --> jet
            if (particle_counter == 1)
            {
                int idx_1 = event.free_particles[0];
                event.particles[idx_1].isCluster = true;
                event.n_clusters++;
                break;
            }
            

            // I. compute distances for every i different from j
            for (int i = 0; i < particle_counter; ++i)
            {
                int free_particle_i = event.free_particles[i];

                for (int j = i+1; j < particle_counter; ++j) 
                {
                    int free_particle_j = event.free_particles[j];
                    
                    float d = compute_distance_ij(&event, free_particle_i, free_particle_j);

                    distances_ij[free_particle_i*n_part+free_particle_j] = d;
                    distances_ij[free_particle_j*n_part+free_particle_i] = d;
                }

                // II. compute distances for every i and the beam
                float d_iB = compute_distance_iB(&event, free_particle_i);

                distances_iB[free_particle_i*n_part] = d_iB;

            }

            // III. find minimum distance between i and j
            float min_d_ij = D; 
            int idx_i, idx_j;
            int pos_i, pos_j;
            for (int i = 0; i < particle_counter; ++i)
            {
                int free_particle_i = event.free_particles[i];
                for (int j = i+1; j < particle_counter; ++j)
                {
                    int free_particle_j = event.free_particles[j];
                    float current_d_ij = distances_ij[free_particle_i*n_part+free_particle_j];
                    if (current_d_ij < min_d_ij)
                    {
                        min_d_ij = current_d_ij;

                        idx_i = free_particle_i;
                        idx_j = free_particle_j;

                        pos_i = i;
                        pos_j = j;
                    }
                    
                }
                
            }
            
            // IV. find minimum distance from beam
            float min_d_iB = D;
            int idx_iB;
            int pos_iB;
            for (int i = 0; i < particle_counter; ++i)
            {
                int free_particle_i = event.free_particles[i];
                float current_d_iB = distances_iB[free_particle_i*n_part];
                if (current_d_iB < min_d_iB)
                {
                    min_d_iB = current_d_iB;
                    idx_iB = free_particle_i;
                    pos_iB = i;
                }
                
            }
            
            /// IV. merge: if d_iB_min < d_ij_min, jet; else new "particle"
            if (min_d_iB < min_d_ij)
            {
                event.free_particles[pos_iB] = event.free_particles[particle_counter-1];
                event.n_clusters++;
                event.particles[idx_iB].isCluster = true;
            }
            else 
            {
                particle_update(&event, idx_i, idx_j);
                event.free_particles[pos_j] = event.free_particles[particle_counter-1];
            }

            particle_counter--;
            
        }
        
        // free memory
        free(distances_ij);
        free(distances_iB);

        // save clusters
        

    }

    // free memory and close file
    free(data);

    H5Sclose(memspace); // maybe here not in the right order (?)
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);

    return 0;
}