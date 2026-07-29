#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <hdf5.h>
#include <cuda_runtime.h>

#include "constants.h"     // FILE_PATH, DATA_PATH, N_EVENTS, N_COLS, N_FEAT, MAX_P, DIM
#include "antikt_gpu.h"     // dichiarazione di antikt_kernel

// ------------------------------------------------------------------
// macro di comodo per controllare gli errori CUDA
// ------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err__));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// ------------------------------------------------------------------
// numero di particelle reali in un evento (stesso criterio di padding
// di read_event): serve solo per sapere fino a dove leggere quando
// scriviamo l'output, non per preparare dati per la GPU.
// ------------------------------------------------------------------
static int getNPart(const double* data, int ev)
{
    long base_col = (long)ev * N_COLS;
    int n_part = 0;
    while (n_part < MAX_P && data[base_col + N_FEAT*n_part] != 0.0) {
        n_part++;
    }
    return n_part;
}

// ------------------------------------------------------------------
// union-find: risale la catena di parentOf fino alla radice
// (equivalente a leggere Particle.components, ma ricostruito a posteriori)
// ------------------------------------------------------------------
static int findRoot(const int* parentOf, int slot)
{
    while (parentOf[slot] != slot) slot = parentOf[slot];
    return slot;
}

// ------------------------------------------------------------------
// scrive un evento in formato piatto: un dataset (n_part x 3) con
// eta/phi/pt e uno (n_part) con il jetID, pronti per l'istogramma 2D
// ------------------------------------------------------------------
static herr_t save_event_flat(hid_t fout, int ev, int n_part,
                               const double* data,   // buffer grezzo completo
                               const int* jetID)
{
    char event_name[64];
    snprintf(event_name, sizeof(event_name), "/event_%d", ev);

    hid_t event_group = H5Gcreate2(fout, event_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (event_group < 0) {
        fprintf(stderr, "Cannot create event group for %s\n", event_name);
        return -1;
    }

    herr_t status = 0;
    long base_col = (long)ev * N_COLS;

    // ---- kinematics: (n_part, 3) = [eta, phi, pt] letti direttamente da data ----
    double* kin = (double*)malloc((size_t)n_part * 3 * sizeof(double));
    for (int p = 0; p < n_part; ++p) {
        long idx = base_col + N_FEAT*p;
        kin[3*p+0] = data[idx+1];   // eta
        kin[3*p+1] = data[idx+2];   // phi
        kin[3*p+2] = data[idx+0];   // pt
    }

    hsize_t kdims[2] = { (hsize_t)n_part, 3 };
    hid_t kspace = H5Screate_simple(2, kdims, NULL);
    hid_t kdset  = H5Dcreate2(event_group, "kinematics", H5T_NATIVE_DOUBLE,
                              kspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (kspace < 0 || kdset < 0) {
        status = -1;
    } else {
        status |= H5Dwrite(kdset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, kin);
    }
    if (kdset  >= 0) H5Dclose(kdset);
    if (kspace >= 0) H5Sclose(kspace);
    free(kin);

    // ---- jetID: (n_part) ----
    hsize_t jdims[1] = { (hsize_t)n_part };
    hid_t jspace = H5Screate_simple(1, jdims, NULL);
    hid_t jdset  = H5Dcreate2(event_group, "jetID", H5T_NATIVE_INT,
                              jspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (jspace < 0 || jdset < 0) {
        status = -1;
    } else {
        status |= H5Dwrite(jdset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, jetID);
    }
    if (jdset  >= 0) H5Dclose(jdset);
    if (jspace >= 0) H5Sclose(jspace);

    status |= H5Gclose(event_group);
    return status;
}

// ------------------------------------------------------------------
// MAIN
// ------------------------------------------------------------------
int main()
{
    clock_t start_t, end_t;
    start_t = clock();

    printf("Getting data ...\n");

    // ---------------- lettura HDF5: identica al seriale ----------------
    hid_t file_id = H5Fopen(FILE_PATH, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) { fprintf(stderr, "Cannot open file\n"); return EXIT_FAILURE; }

    hid_t dset_id = H5Dopen2(file_id, DATA_PATH, H5P_DEFAULT);
    if (dset_id < 0) { fprintf(stderr, "Cannot get dataset\n"); return EXIT_FAILURE; }

    hid_t space_id = H5Dget_space(dset_id);
    if (space_id < 0) { fprintf(stderr, "Cannot get dataspace\n"); return EXIT_FAILURE; }

    hsize_t dataset_dims[2];
    H5Sget_simple_extent_dims(space_id, dataset_dims, NULL);
    printf("Dataset dimensions: %lu x %lu\n", dataset_dims[0], dataset_dims[1]);

    hsize_t n_read = N_EVENTS;
    hsize_t offset[DIM] = {0, 0};
    hsize_t count[DIM]  = {n_read, N_COLS};

    H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    hid_t memspace = H5Screate_simple(DIM, count, NULL);
    if (memspace < 0) { fprintf(stderr, "Cannot create dataspace\n"); return EXIT_FAILURE; }

    printf("Allocating memory for data ...\n");
    double* data = (double*)malloc((size_t)n_read * N_COLS * sizeof(double));
    if (data == NULL) { fprintf(stderr, "Failed to allocate data buffer\n"); return EXIT_FAILURE; }

    herr_t read_id = H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, space_id, H5P_DEFAULT, data);
    if (read_id < 0) { fprintf(stderr, "Cannot read data\n"); return EXIT_FAILURE; }

    printf("Creating output file ...\n");
    hid_t fout = H5Fcreate(
        "mnt/POD/MCP_GD/JetClusteringAlgorithm/data/results/gpu/clusters.h5",
        H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fout < 0) { fprintf(stderr, "Cannot create output file\n"); return EXIT_FAILURE; }

    // ---------------- allocazione e copia su device: buffer grezzo 1:1 ----------------
    // Nessuna conversione SoA sull'host: data ha gia' il layout che serve
    // (data[ev*N_COLS + N_FEAT*p + feat]), lo copiamo cosi' com'e'.
    printf("Copying raw data to device ...\n");

    double *d_data;
    int    *d_parentOf;

    size_t dataBytes = (size_t)N_EVENTS * N_COLS * sizeof(double);
    size_t intBytes  = (size_t)N_EVENTS * MAX_P * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_data, dataBytes));
    CUDA_CHECK(cudaMalloc(&d_parentOf, intBytes));

    CUDA_CHECK(cudaMemcpy(d_data, data, dataBytes, cudaMemcpyHostToDevice));

    // ---------------- lancio kernel: un blocco per evento ----------------
    printf("Launching kernel ...\n");

    int threadsPerBlock = 128;   // valore di partenza, non e' ancora il momento di ottimizzarlo
    size_t dynShared = threadsPerBlock * (2*sizeof(double) + 3*sizeof(int));

    antikt_kernel<<<N_EVENTS, threadsPerBlock, dynShared>>>(
        d_data, N_COLS, d_parentOf, N_EVENTS);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------- copia indietro SOLO parentOf ----------------
    int* parentOf_h = (int*)malloc(intBytes);
    if (!parentOf_h) {
        fprintf(stderr, "Failed to allocate output buffers\n");
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaMemcpy(parentOf_h, d_parentOf, intBytes, cudaMemcpyDeviceToHost));

    // ---------------- ricostruzione jetID + scrittura output ----------------
    printf("Reconstructing jet IDs and saving events ...\n");

    int* jetID_buf = (int*)malloc(MAX_P * sizeof(int));

    for (int ev = 0; ev < N_EVENTS; ++ev) {
        long base = (long)ev * MAX_P;
        int n_part = getNPart(data, ev);   // stesso criterio di padding, letto da data

        for (int p = 0; p < n_part; ++p) {
            jetID_buf[p] = findRoot(parentOf_h + base, p);
        }

        save_event_flat(fout, ev, n_part, data, jetID_buf);
    }

    // ---------------- pulizia ----------------
    free(jetID_buf);
    free(parentOf_h);
    free(data);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_parentOf));

    H5Sclose(memspace);
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(fout);
    H5Fclose(file_id);

    end_t = clock();
    double exec_time = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("\nExecution time (total, %d events): %f (sec)\n\n", N_EVENTS, exec_time);

    return 0;
}

// ------------------------------------------------------------------
// Compilazione (kernel e main in file separati -> serve compilazione
// separabile, -rdc=true, per poter lanciare da main.cu un kernel
// definito in antikt_gpu.cu):
//
//   nvcc -rdc=true -O2 main.cu antikt_gpu.cu -lhdf5 -o antikt_gpu
//
// In alternativa, se preferisci evitare -rdc=true, incolla il
// contenuto di antikt_gpu.cu direttamente sopra main() in questo
// stesso file: compila normalmente senza flag aggiuntivi.
// ------------------------------------------------------------------
