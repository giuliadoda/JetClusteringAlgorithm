#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <hdf5.h>           // TO BE INSTALLED
#include <omp.h>            // TO BE INSTALLED

#define MAX_P 700
#define N_FEAT 3
#define N_COLS (MAX_P * N_FEAT)
#define DIM 2

#define N_EVENTS 10         // actually 8192

#define R 0.4
#define D 1000.             // maybe change it after inspecting distance actual values

#define FILE_PATH "datafile.h5"
#define DATA_PATH "df"     





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
    hsize_t n_read = N_EVENTS;
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
    double *data = malloc(n_read * N_COLS * sizeof(double));

    // check if the allocation happened properly
    if (data == NULL) {
        fprintf(stderr, "Failed to allocate data buffer\n");
        exit(1);
    }

    // read data -> maybe read in chunks
    H5Dread(
        dset_id,                // dataset identifier
        H5T_NATIVE_DOUBLE,      // memory datatype identifier
        memspace,               // memory dataspace identifier
        space_id,               // file dataspace containing the selected hyperslab
        H5P_DEFAULT,            // identifier of a transfer property list
        data                    // buffer to receive data read from file
    );

    // output file
    hid_t fout = H5Fcreate(
        "clusters.h5",
        H5F_ACC_TRUNC,  // file access flag: if the file already exists, erase all data previously stored
        H5P_DEFAULT,    // file creation property list identifier
        H5P_DEFAULT     // file access property list identifier
    );

    // array to store elapsed time for each event
    double times[N_EVENTS];

    // loop over events
    #pragma omp parallel for // each event per thread
    for (int ev = 0; ev < n_read; ++ev) {

        Event event; 

        // compute wall time
        double t_event = omp_get_wtime();

        read_event(&data[ev*N_COLS], &event, ev); // contains a for cycle

        printf("\nEvent ID %d\n", event.id);

        int n_part = event.n_particles;
        printf("Number of particles for this event: %d\n", n_part);

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

            // to handle minimum distance from beam
            // shared between threads
            Minimum beam_min;

            beam_min.distance = D;
            beam_min.id_i = -1;
            beam_min.id_j = -1;
            beam_min.idx_i = -1;
            beam_min.idx_j = -1;

            // to handle minimum distance between all the particles
            // shared between threads
            Minimum global_min;

            global_min.distance = D;
            global_min.id_i = -1;
            global_min.id_j = -1;
            global_min.idx_i = -1;
            global_min.idx_j = -1;

            // >>> starting parallel region to compute distances <<<
            #pragma omp parallel
            {
            // to handle minimun distance between particles (shared between threads)
            // defining it there means that is private to each thread
            Minimum local_min;

            local_min.distance = D;
            local_min.id_i = -1;
            local_min.id_j = -1;
            local_min.idx_i = -1;
            local_min.idx_j = -1;

            #pragma omp for 
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
                    if (current_d_ij < local_min.distance)
                    {
                        local_min.distance = current_d_ij;

                        local_min.idx_i = i;
                        local_min.idx_j = j;
                        local_min.id_i = free_particle_i;
                        local_min.id_j = free_particle_j;
                    }
            
                }

                // beam distance
                // maybe false sharing because event is shared
                double current_d_iB = event.particles[free_particle_i].d_B;

                // update finding minimum distance iB
                #pragma omp critical // since beam_min in shared among threads and we have to avoid race conditions (serial)
                if (current_d_iB < beam_min.distance)
                {
                    beam_min.distance = current_d_iB;

                    beam_min.id_i = free_particle_i;
                    beam_min.idx_i = i;
                }
                
            } // implicit barrier here (end of parallel for)

            // update minimum correctly between threads
            // each thread enters this zone safely without race (serial)
            #pragma omp critical
            if (local_min.distance < global_min.distance)
            {
                global_min.distance = local_min.distance;

                global_min.id_i = local_min.id_i;
                global_min.id_j = local_min.id_j;

                global_min.idx_i = local_min.idx_i;
                global_min.idx_j = local_min.idx_j;
            }

            }
            // >>> ending parallel region to compute distances <<<

            // safely merge outside parallel region (we don't need threads here)
            // merge: if d_iB_min < d_ij_min, jet; else new "particle"
            if (beam_min.distance < global_min.distance)
            {
                event.free_particles[beam_min.idx_i] = event.free_particles[particle_counter-1];
                event.n_clusters++;
                event.particles[beam_min.id_i].isJet = true;
            }
            else 
            {
                particle_update(&event, global_min.id_i, global_min.id_j);
                event.free_particles[global_min.idx_j] = event.free_particles[particle_counter-1];
            }

            particle_counter--;
            
        }

        // save clusters
        char event_name[64];
        sprintf(event_name, "/event_%d", event.id);

        hid_t event_group = H5Gcreate2(
            fout,
            event_name,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT
        );

        for (int p = 0; p < n_part; ++p)
        {
            Particle *cluster = &event.particles[p];

            if (!cluster->isJet) continue;

            char cluster_name[64];
            sprintf(cluster_name, "cluster_%d", p);

            hid_t cluster_group = H5Gcreate2(
                event_group,
                cluster_name,
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT
            );

            // save particle ids
            hsize_t dims[1] = {cluster->n_components};

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

            H5Dwrite(
                dset,
                H5T_NATIVE_INT,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                cluster->components
            );

            H5Dclose(dset);
            H5Sclose(space);

            H5Gclose(cluster_group);
            
        }

        H5Gclose(event_group); 

        // free event memory
        event_free(&event);

        t_event = omp_get_wtime() - t_event;

        times[ev] = t_event;

    }

    // just read elapsed processing time for each event
    for (int id = 0; id < N_EVENTS; ++id)
    {
        printf("Event ID %d elapsed time (s) %f", id, times[id]);
    }
    

    // free memory and close file
    free(data);

    H5Sclose(memspace); // maybe here not in the right order (?)
    H5Sclose(space_id);
    H5Dclose(dset_id);

    H5Fclose(fout);
    H5Fclose(file_id);

    return 0;
}