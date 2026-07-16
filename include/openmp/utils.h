#ifndef UTILS_H
#define UTILS_H

// particle-pseudocluster struct
typedef struct 
{
    int id;                 // initial particle ID
    int n_components;       // how many particles in the cluster

    int *components;        // to keep track of the particles belonging here (particle IDs), to be dynamically allocated
    // 

    // kinematics
    double p_t, eta, phi;
    
    // distance from beam
    double d_B;

    bool isJet;

} Particle;

// event struct
typedef struct  
{
    int id;
    int n_particles;

    int n_clusters;

    int *free_particles; // to keep track of the particles that are not yet assigned to a cluster, to be dynamically allocated

    Particle *particles; // will become clusters in the end, to be dynamically allocated
    // an array of struct can be problematich in computing the distance because of non-sequential access of kinematic values (would be better a structure of array like)

} Event;

// struct to handle minimum distance
typedef struct
{
    double distance;

    // particle IDs
    int id_i, id_j;

    // particle indexes
    int idx_i, idx_j;

} Minimum;

#endif