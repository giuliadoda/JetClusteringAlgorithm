#ifndef OPENMP_FUNCTIONS_H
#define OPENMP_FUNCTIONS_H

#include "utils.h"  

// function to loop over events 
void process_single_event(double *data, int ev, double *times, hid_t fout);

#endif