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

// CUDA error checking
#define CUDA_CHECK(call)                                                     
    do {                                                                     
        cudaError_t err = (call);                                            
        if (err != cudaSuccess) {                                            
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     
                    __FILE__, __LINE__, cudaGetErrorString(err));            
            exit(EXIT_FAILURE);                                              
        }                                                                    
    } while (0)

int main() {

    // execution time (CPU)
    clock_t start_t, end_t;
    double exec_time;

    start_t = clock();

    // CUDA timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }
    
    // get identifier for a copy of the dataspace for a dataset 
    hid_t space_id = H5Dget_space(dset_id); 

    if (space_id < 0)
    {
        fprintf(stderr, "Cannot get dataspace\n");
        H5Dclose(dset_id);
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }

    // check dataset dimensions

    hsize_t dataset_dims[DIM];

    H5Sget_simple_extent_dims(space_id, dataset_dims, NULL);
    
    // read dataset (only the selected number of events)
    hsize_t start_row = 0;
    hsize_t start_col = 0;
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
        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }

    // allocate memory
    double *data = malloc(n_read * N_COLS * sizeof(double));

    // check if the allocation happened properly
    if (data == NULL) {
        fprintf(stderr, "Failed to allocate data buffer\n");
        H5Sclose(memspace);
        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(file_id);
        return EXIT_FAILURE;
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
        free(data);
        H5Sclose(memspace);
        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }
    
    // output file
    hid_t fout = H5Fcreate(
        "/mnt/POD/MCP_GD/JetClusteringAlgorithm/data/results/cuda/clusters.h5",
        H5F_ACC_TRUNC,  // file access flag: if the file already exists, erase all data previously stored
        H5P_DEFAULT,    // file creation property list identifier
        H5P_DEFAULT     // file access property list identifier
    );

    if (fout < 0)
    {
        fprintf(stderr, "Cannot create output file\n");
        free(data);
        H5Sclose(memspace);
        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }

    hsize_t out_dims[DIM] = {n_read, MAX_P};

    hid_t space_out = H5Screate_simple(
        DIM,
        out_dims,
        NULL
    );

    if (space_out < 0)
    {
        fprintf(stderr, "Cannot create output dataspace\n");
        free(data);
        H5Sclose(memspace);
        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(fout);
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }

    hid_t dset_out = H5Dcreate2(
        fout,
        "/cluster_trace",
        H5T_NATIVE_INT,
        space_out,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT
    );
    
    if (dset_out < 0)
    {
        fprintf(stderr, "Cannot create output dataset\n");
        free(data);
        H5Sclose(space_out);
        H5Sclose(memspace);
        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(fout);
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }

    hsize_t time_dims[1] = {n_read};

    hid_t time_space = H5Screate_simple(
        1,
        time_dims,
        NULL
    );

    if (time_space < 0)
    {
        fprintf(stderr, "Cannot create output time space\n");
        free(data);
        H5Dclose(dset_out);
        H5Sclose(space_out);
        H5Sclose(memspace);
        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(fout);
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }

    hid_t time_dset = H5Dcreate2(
        fout,
        "/event_times",
        H5T_NATIVE_FLOAT,
        time_space, 
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT
    );

    if (time_dset < 0)
    {
        fprintf(stderr, "Cannot create output time dataset\n");
        free(data);
        H5Sclose(time_space);
        H5Dclose(dset_out);
        H5Sclose(space_out);
        H5Sclose(memspace);
        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(fout);
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }

    // array to store elapsed time for each event, to be passed to GPU
    float *times = malloc(N_EVENTS * sizeof(float));

    // array to store cluster trace
    int *cluster_trace = malloc(N_EVENTS*MAX_P*sizeof(int));

    // allocate memory on GPU
    double *dev_data;
    size_t data_size = n_read * N_COLS * sizeof(double);
    CUDA_CHECK(cudaMalloc((void**)&dev_data, data_size));

    int *dev_cluster_trace;
    size_t cluster_trace_size = N_EVENTS * MAX_P * sizeof(int);
    CUDA_CHECK(cudaMalloc((void**)&dev_cluster_trace, cluster_trace_size));

    float *dev_times; // pointer to GPU memory for times array
    size_t times_size = N_EVENTS * sizeof(float);
    CUDA_CHECK(cudaMalloc((void **)&dev_times, times_size));

    // copy data from host to device
    cudaMemcpy(dev_data, data, data_size, cudaMemcpyHostToDevice);

    // define number of blocks and threads per blocks
    int N_blocks = N_EVENTS;
    int N_thr_bl = 512;

    // calculate amount of dynamic shared memory needed 
    size_t shared_mem_size = (size_t)N_thr_bl * (2*sizeof(double) + 3*sizeof(int));

    // check if shared memory is enough
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, processEvent);
    size_t total_shared = attr.sharedSizeBytes + shared_mem_size; // including also dynamic memory

    if (total_shared > prop.sharedMemPerBlock) {
        fprintf(stderr, "Requested shared memory %zu bytes exceeds device max of %zu bytes\n",
                total_shared, prop.sharedMemPerBlock);
        exit(EXIT_FAILURE);
    }

    // kernel launch
    cudaEventRecord(start);

    processEvent<<<N_blocks, N_thr_bl, shared_mem_size>>>(dev_data, dev_cluster_trace, dev_times);

    cudaGetLastError();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    cudaMemcpy(cluster_trace, dev_cluster_trace, cluster_trace_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(times, dev_times, times_size, cudaMemcpyDeviceToHost);

    // save clustering results
    H5Dwrite(
        dset_out,
        H5T_NATIVE_INT,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        cluster_trace
    );

    // save times per event
    H5Dwrite(
        time_dset,
        H5T_NATIVE_FLOAT,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        times
    );

    // free memory and close file
    free(data);
    free(times);
    free(cluster_trace);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_data);
    cudaFree(dev_cluster_trace);
    cudaFree(dev_times);

    H5Dclose(time_dset);
    H5Sclose(time_space);

    H5Dclose(dset_out);
    H5Sclose(space_out);

    H5Sclose(memspace); 
    H5Sclose(space_id);
    H5Dclose(dset_id);

    H5Fclose(fout);
    H5Fclose(file_id);

    end_t = clock();

    exec_time = (double) (end_t - start_t)/CLOCKS_PER_SEC; 

    // save event execution times

    printf("\nExecution time (total, %d events): %f (sec)\n\n", N_EVENTS, exec_time);

    printf("\nKernel execution time: %.3f ms\n", kernel_ms);

    double avg = 0.0;

    for(int i=0;i<N_EVENTS;i++)
        avg += times[i];

    avg /= N_EVENTS;

    printf("\nAverage event time: %.6f ms\n", avg);

    return 0;

}