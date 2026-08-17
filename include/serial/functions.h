#ifndef SERIAL_FUNCTIONS_H
#define SERIAL_FUNCTIONS_H

#include "utils.h"  

// function to loop over events 
void process_event(double *data, Event *ev, double *times, int *nparticles, hid_t fout);

#endif