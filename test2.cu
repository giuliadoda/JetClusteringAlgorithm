// antikt_gpu.cu
// -----------------------------------------------------------------------
// Port GPU della pipeline seriale (read_event / compute_distance_ij /
// particle_update / process_event). Un blocco CUDA per evento.
//
// Corrispondenza con il codice seriale:
//   - free_particles[]  ->  s_free[]      (stesso swap-remove)
//   - Particle.components[] -> parentOf[] (union-find, ricostruito sull'host)
//   - compute_distance_ij    -> stesso calcolo, parallelizzato sulle coppie
//   - particle_update        -> stessa media pesata da pt
//
// NIENTE benchmark/ottimizzazioni qui: l'obiettivo e' l'organizzazione
// corretta di memoria e flusso dati, 1:1 con la logica seriale.
// -----------------------------------------------------------------------

#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

#ifndef MAX_P
#define MAX_P 700     // deve combaciare con MAX_P di constants.h
#endif
#ifndef N_FEAT
#define N_FEAT 3      // p_t, eta, phi -- deve combaciare con N_FEAT di constants.h
#endif

#define R_PARAM   0.4
#define R2_PARAM  (R_PARAM*R_PARAM)

__device__ __forceinline__ double deltaPhi(double phi_i, double phi_j) {
    double dphi = fabs(phi_i - phi_j);
    if (dphi > M_PI) dphi = 2.0*M_PI - dphi;   // stessa logica di compute_distance_ij
    return dphi;
}

// Un blocco = un evento.
// Input: data_in e' il buffer GREZZO, stesso layout di quello letto da HDF5:
//   data_in[ev*N_COLS + N_FEAT*p + 0/1/2] = p_t/eta/phi della particella p
// Nessuna conversione SoA preventiva: il kernel scopre da solo quali slot
// sono particelle reali (p_t != 0) e quante sono (nActive), esattamente
// come faceva read_event sull'host.
// Output:
//   parentOf[base+p] = union-find: risalendo la catena si trova la radice.
//   Uno slot p e' un jet finale se e solo se, dopo il kernel, la sua
//   radice e' se stesso (nessuno lo ha mai assorbito) -- non serve un
//   array isJet separato, e' ridondante con questa proprieta'.
__global__ void antikt_kernel(
        const double* __restrict__ data_in,
        int N_COLS,
        int* __restrict__ parentOf,
        int nEvents)
{
    int ev = blockIdx.x;
    if (ev >= nEvents) return;
    int tid       = threadIdx.x;
    int nThreads  = blockDim.x;
    long base_col = (long)ev * N_COLS;   // offset nel buffer grezzo
    int  base      = ev * MAX_P;         // offset nell'output (parentOf)

    __shared__ double s_pt[MAX_P], s_eta[MAX_P], s_phi[MAX_P], s_dB[MAX_P];
    __shared__ int    s_free[MAX_P];   // equivalente di event->free_particles
    __shared__ int    nActive; // diventato free particle counter per consistenza

    // shared memory dinamica: due reduction parallele (d_ij e d_iB)
    extern __shared__ char dyn[];
    double* r_valIJ = (double*)dyn;
    int*    r_idxI  = (int*)(r_valIJ + nThreads);
    int*    r_idxJ  = (int*)(r_idxI  + nThreads);
    double* r_valB  = (double*)(r_idxJ + nThreads);
    int*    r_idxB  = (int*)(r_valB + nThreads);

    if (tid == 0) nActive = 0;
    __syncthreads();

    // ---- equivalente di read_event: scopre le particelle reali e le carica ----
    for (int p = tid; p < MAX_P; p += nThreads) {
        parentOf[base+p] = p;   // ogni slot e' inizialmente radice di se stesso

        double pt = data_in[base_col + N_FEAT*p + 0];
        if (pt != 0.0) {                 // stesso criterio di padding di read_event
            s_pt[p]  = pt;
            s_eta[p] = data_in[base_col + N_FEAT*p + 1];
            s_phi[p] = data_in[base_col + N_FEAT*p + 2];
            s_dB[p]  = 1.0/(pt*pt);
            int pos  = atomicAdd(&nActive, 1);   // posizione compattata, in ordine di scoperta
            s_free[pos] = p;
        }
    }
    __syncthreads();

    if (nActive == 0) return;

    // ---- equivalente di: while (particle_counter > 0) ----
    while (nActive > 0) {

        if (nActive == 1) {
            if (tid == 0) {
                nActive = 0;   // l'unico slot rimasto in s_free e' gia' radice di se stesso
            }
            __syncthreads();
            continue;
        }

        double bestIJ = DBL_MAX; 
        int bestI = -1, bestJ = -1;

        double bestB  = DBL_MAX; 
        int bestBi = -1;

        // equivalente del doppio for (i,j) + calcolo d_iB, ma con stride sui thread
        for (int i = tid; i < nActive; i += nThreads) {
            int slotI = s_free[i];

            double dB = s_dB[slotI];
            if (dB < bestB) { bestB = dB; bestBi = i; }

            for (int j = i+1; j < nActive; ++j) {
                int slotJ = s_free[j];
                double dphi = deltaPhi(s_phi[slotI], s_phi[slotJ]);
                double deta = s_eta[slotI] - s_eta[slotJ];
                double dR2  = deta*deta + dphi*dphi;
                double p2   = fmin(s_dB[slotI], s_dB[slotJ]);
                double dij  = p2 * dR2 / R2_PARAM;
                if (dij < bestIJ) { bestIJ = dij; bestI = i; bestJ = j; }
            }
        }

        r_valIJ[tid]=bestIJ; 
        r_idxI[tid]=bestI; 
        r_idxJ[tid]=bestJ;

        r_valB[tid]=bestB;   
        r_idxB[tid]=bestBi;

        __syncthreads();

        // reduction ad albero per entrambe le coppie di risultati insieme
        for (int s = nThreads/2; s > 0; s >>= 1) {
            if (tid < s) {
                if (r_valIJ[tid+s] < r_valIJ[tid]) {
                    r_valIJ[tid]=r_valIJ[tid+s]; 
                    r_idxI[tid]=r_idxI[tid+s]; 
                    r_idxJ[tid]=r_idxJ[tid+s];
                }
                if (r_valB[tid+s] < r_valB[tid]) {
                    r_valB[tid]=r_valB[tid+s]; 
                    r_idxB[tid]=r_idxB[tid+s];
                }
            }
            __syncthreads();
        }

        // ---- equivalente di: if (min_d_iB < min_d_ij) { jet } else { merge } ----
        if (tid == 0) {
            double min_d_ij = r_valIJ[0];
            double min_d_iB = r_valB[0];

            if (min_d_iB < min_d_ij) {
                int idxB  = r_idxB[0];
                s_free[idxB] = s_free[nActive-1];   // swap-remove, come nel seriale
                nActive--;
            } else {
                int idxI = r_idxI[0], idxJ = r_idxJ[0];
                int slotI = s_free[idxI], slotJ = s_free[idxJ];

                double pt_i = s_pt[slotI], pt_j = s_pt[slotJ];
                double pt_new = pt_i + pt_j;

                // stessa media pesata di particle_update
                s_eta[slotI] = (pt_i*s_eta[slotI] + pt_j*s_eta[slotJ]) / pt_new;
                s_phi[slotI] = (pt_i*s_phi[slotI] + pt_j*s_phi[slotJ]) / pt_new;
                s_pt[slotI]  = pt_new;
                s_dB[slotI]  = 1.0/(pt_new*pt_new);

                parentOf[base+slotJ] = slotI;       // j "assorbito" da i
                s_free[idxJ] = s_free[nActive-1];   // swap-remove
                nActive--;
            }
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------
// Lancio (nell'host code, al posto della chiamata a process_event):
//
//   int threadsPerBlock = 128;   // valore di partenza, non e' il momento
//                                // di ottimizzarlo ora
//   size_t dynShared = threadsPerBlock * (2*sizeof(double) + 3*sizeof(int));
//   antikt_kernel<<<N_EVENTS, threadsPerBlock, dynShared>>>(
//       d_data, N_COLS, d_parentOf, N_EVENTS);
//
// d_data e' semplicemente una copia 1:1 del buffer "data" letto da HDF5
// (cudaMemcpy diretto, nessuna conversione SoA sull'host).
//
// Dopo il lancio: cudaMemcpy indietro SOLO parentOf (un array di int,
// size N_EVENTS*MAX_P). pt/eta/phi originali li hai gia' sull'host in
// data[] e non servono ricopiati: per l'istogramma ti serve la
// cinematica ORIGINALE di ogni particella, non quella del pseudo-jet.
//
// Ricostruzione jetID sull'host (equivalente a leggere Particle.components,
// ma con un semplice union-find). Uno slot p e' un jet finale se e solo
// se findRoot(p) == p:
//
//   static int findRoot(const int* parentOf, int slot) {
//       while (parentOf[slot] != slot) slot = parentOf[slot];
//       return slot;
//   }
//
//   for (int ev = 0; ev < N_EVENTS; ++ev) {
//       int base = ev*MAX_P;
//       for (int p = 0; p < nPart_h[ev]; ++p) {
//           int jetID = findRoot(&parentOf_h[base], p);
//           // scrivi (eta_h[base+p], phi_h[base+p], pt_h[base+p], jetID)
//       }
//   }
// -----------------------------------------------------------------------
