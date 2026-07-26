// strategy: each event per block 

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <hdf5.h>           

#include "constants.h" 
#include "utils.h"  
#include "functions.cuh"


int main() {

    // execution time --> am I getting this right?
    clock_t start_t, end_t;
    double exec_time;

    start_t = clock();

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

    // also for data
    // TO DO

    // define number of blocks and threads per blocks
    int N_blocks = N_EVENTS;
    int N_thr_bl = 128; // maybe change later

    // calculate amount of shared memory needed --> CHECK IT
    size_t shared_mem_size = N_thr_bl * (2*sizeof(double) + 3*sizeof(int));

    processEvent<<<N_blocks, N_thr_bl, shared_mem_size>>>();

    // get data from GPU

    // save clustering results

    // free memory and close file
    free(data);

    H5Sclose(memspace); 
    H5Sclose(space_id);
    H5Dclose(dset_id);

    H5Fclose(fout);
    H5Fclose(file_id);

    end_t = clock();

    exec_time = (double) (end_t - start_t)/CLOCKS_PER_SEC; // also save it

    // save event execution times

    printf("\nExecution time (total, %d events): %f (sec)\n\n", N_EVENTS, exec_time);

    return 0;

}