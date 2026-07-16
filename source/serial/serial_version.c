#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <hdf5.h>           // TO BE INSTALLED

#include "include/constants.h" 
#include "include/serial/utils.h"  
#include "include/serial/functions.h"


// ----------- MAIN -------------

int main() {

    // execution time
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

    // read data -> maybe read in chunks
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
        "clusters.h5",
        H5F_ACC_TRUNC,  // file access flag: if the file already exists, erase all data previously stored
        H5P_DEFAULT,    // file creation property list identifier
        H5P_DEFAULT     // file access property list identifier
    );

    if (fout < 0)
    {
        fprintf(stderr, "Cannot create output file\n");
        return EXIT_FAILURE;
    }
    

    // array to store elapsed time for each event
    double times[N_EVENTS];

    // declare Event
    static Event event;

    // loop over events --> ADD OUTPUT FILE MANAGEMENT!
    process_event(&data, &event, &times, fout);

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

    end_t = clock();

    exec_time = (double) (end_t - start_t)/CLOCKS_PER_SEC; // save it

    return 0;
}