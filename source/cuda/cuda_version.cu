// FIRST VERSION: serial loop over events, parallelizing over minimum distance calculation

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <hdf5.h>           

#include "constants.h" 
#include "utils.h"  
#include "functions.h"


int main() {

    // get file identifier first 
    // H5F_ACC_RDONLY -> read only  
    // H5P_DEFAULT -> use the default behavior of the library
    hid_t file_id = H5Fopen(FILE_PATH, H5F_ACC_RDONLY, H5P_DEFAULT);

    // safety check
    if (file_id < 0) {
        fprintf(stderr, "Cannot open file\n");
        return EXIT_FAILURE;
    }   

    // get dataset identifier
    hid_t dset_id = H5Dopen2(file_id, DATA_PATH, H5P_DEFAULT);

    if (dset_id < 0)
    {
        fprintf(stderr, "Cannot get dataset\n");
        return EXIT_FAILURE;
    }
    
    // get identifier for a copy of the dataspace for a dataset 
    hid_t space_id = H5Dget_space(dset_id); 

    if (space_id < 0)
    {
        fprintf(stderr, "Cannot get dataspace\n");
        return EXIT_FAILURE;
    }

    // check dataset dimensions

    hsize_t dataset_dims[2];

    H5Sget_simple_extent_dims(space_id, dataset_dims, NULL);

    printf("Dataset dimensions: %lu x %lu\n", dataset_dims[0], dataset_dims[1]);
    
    // read dataset (only the selected number of events)
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

    if (memspace < 0)
    {
        fprintf(stderr, "Cannot create dataspace\n");
        return EXIT_FAILURE;
    }

    // allocate memory
    double *data = malloc(n_read * N_COLS * sizeof(double));

    // check if the allocation happened properly
    if (data == NULL) {
        fprintf(stderr, "Failed to allocate data buffer\n");
        exit(1);
    }

    // read data 
    hid_t read_id = H5Dread(
        dset_id,                // dataset identifier
        H5T_NATIVE_DOUBLE,      // memory datatype identifier
        memspace,               // memory dataspace identifier
        space_id,               // file dataspace containing the selected hyperslab
        H5P_DEFAULT,            // identifier of a transfer property list
        data                    // buffer to receive data read from file
    );

    if (read_id < 0)
    {
        fprintf(stderr, "Cannot read data\n");
        return EXIT_FAILURE;
    }
    
    // output file
    hid_t fout = H5Fcreate(
        "mnt/POD/MCP_GD/JetClusteringAlgorithm/data/results/serial/clusters.h5",
        H5F_ACC_TRUNC,  // file access flag: if the file already exists, erase all data previously stored
        H5P_DEFAULT,    // file creation property list identifier
        H5P_DEFAULT     // file access property list identifier
    );

    if (fout < 0)
    {
        fprintf(stderr, "Cannot create output file\n");
        return EXIT_FAILURE;
    }

    // array to store elapsed time for each event, to be passed to GPU
    double *times = malloc(N_EVENTS * sizeof(double));

    if (times == NULL) {
        fprintf(stderr, "Failed to allocate times buffer\n");
        free(data);
        return EXIT_FAILURE;
    }

    // allocate memory on GPU
    double *dev_t; // pointer to GPU memory for times array
    int times_size = N_EVENTS * sizeof(double);
    cudaMalloc((void **)&dev_t, times_size);

    // compute number of blocks and threads per blocks
    int N_blocks;
    int N_thr_bl;

    // loop over events
    for (int ev = 0; ev < N_EVENTS; ++ev)
    {
        Event event;

        // read event

        int n_part = event.n_particles;

        // free particles counter
        int particle_counter = n_part;

        // loop over free particles
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

            // compute distance (parallelizing)

            int event_size = sizeof(Event);
            cudaMalloc((void **)&event, event_size);

            // allocate memory on GPU for data
            int ev_data_size = ev * N_COLS * sizeof(double);
            double event_data = data[ev * N_COLS];
            double *dev_data;
            cudaMalloc((void **)&dev_data, ev_data_size);

            // copy data from host to device
            cudaMemcpy(dev_data, event_data, ev_data_size, cudaMemcpyHostToDevice);

            // launch kernel for distance computation

            // merge

            particle_counter--;

            // free GPU memory
            cudaFree(event);
            cudaFree(dev_data);

        }

        // save event clusters

        // free event memory




    }
    


}