import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  

# event ID to visualize
EVENT_ID = 3

# code version
VERSION = "serial"

# PATHS

# data
RAW_FILE_PATH = "/mnt/POD/MCP_GD/JetClusteringAlgorithm/data/events_anomalydetection_Z_XY_qqq.h5"
# dataset inside H5 file
RAW_DATASET_PATH = "/df/block0_values"

# results
CLUSTERS_FILE_PATH = f"/mnt/POD/MCP_GD/JetClusteringAlgorithm/data/results/{VERSION}/clusters.h5"

# p_t, eta, phi (kinematics)
N_FEAT = 3
MAX_P = 700

# histogram settings
BAR_WIDTH_ETA = 0.1
BAR_WIDTH_PHI = 0.1

# plot directory
SAVE_DIR = f"/mnt/POD/MCP_GD/JetClusteringAlgorithm/plots/figures/{VERSION}" 
os.makedirs(SAVE_DIR, exist_ok=True)

# read event from raw data
# returns event row and number of particles
def read_raw_particles(event_id=EVENT_ID, raw_file_path=RAW_FILE_PATH, dataset_path=RAW_DATASET_PATH, n_feat=N_FEAT):

    with h5py.File(raw_file_path, "r") as f:
        dset = f[dataset_path]
        row = dset[event_id]  

    particles = np.trim_zeros(row)

    n = particles.shape[0]/n_feat

    return particles, n

# read clustering results 
# result file structure:
# event i group
#   --> cluster i group
#       --> cluster ID
#       --> cluster components (particle IDs)    
#       --> cluster kinematics
# returns a list of clusters (sorted by ID)
def read_clusters(clusters_file_path=CLUSTERS_FILE_PATH, event_id=EVENT_ID):

    clusters = []
    event_name = f"/event_{event_id}"

    with h5py.File(clusters_file_path, "r") as f:

        event_group = f[event_name]

        for key in event_group.keys():
            if not key.startswith("cluster_"):
                continue

            cluster_group = event_group[key]

            components = cluster_group["components"][()]
            cluster_id = int(cluster_group["components"].attrs.get("cluster_id",
                              cluster_group.attrs.get("cluster_id", -1))) \
                if "cluster_id" in cluster_group.attrs else int(cluster_group["cluster_id"][()]) \
                if "cluster_id" in cluster_group else -1

            # cluster ID
            if "cluster_id" in cluster_group.attrs:
                cluster_id = int(cluster_group.attrs["cluster_id"])

            kinematics = cluster_group["kinematics"][()] if "kinematics" in cluster_group else None

            clusters.append({
                "cluster_id": cluster_id,
                "components": np.asarray(components, dtype=int),
                "kinematics": kinematics,
            })

    # sorting by cluster ID just because it's fancy
    clusters.sort(key=lambda c: c["cluster_id"])

    return clusters

# map each particle ID to the corresponding Jet
# take the list of clusters with clusters info
# returns a dict like particle ID : cluster ID
def build_particle_to_cluster_map(clusters):

    mapping = {}

    for cluster in clusters:

        for particle_idx in cluster["components"]:

            mapping[int(particle_idx)] = cluster["cluster_id"]

    return mapping

# plot 2D histogram
# take raw data (particles), mapping (particle_to_cluster) and #clusters
def plot_event(particles, particle_to_cluster, n_part_raw, n_clusters, event_id=EVENT_ID):

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = cm.get_cmap("tab20", max(n_clusters, 1))

    plotted_clusters = set()

    for p_idx in range(particles.shape[0]):

        p_t, eta, phi = particles[p_idx, 0], particles[p_idx, 1], particles[p_idx, 2]

        cluster_id = particle_to_cluster.get(p_idx, -1)
        color = cmap(cluster_id % cmap.N) if cluster_id >= 0 else "gray"

        plotted_clusters.add(cluster_id)

        ax.bar3d(
            eta - BAR_WIDTH_ETA / 2,
            phi - BAR_WIDTH_PHI / 2,
            0,
            BAR_WIDTH_ETA,
            BAR_WIDTH_PHI,
            p_t,
            color=color,
            shade=True
        )

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_zlabel(r"$p_T$ (GeV)")
    ax.set_title(f"Event ID: {event_id} - Particles: {n_part_raw}, Cluster found: {n_clusters}") 

    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, f"jets_3d_event_{event_id}.png")
    fig.savefig(save_path, dpi=300)

    plt.show()

def main():

    particles, n_part = read_raw_particles()

    clusters = read_clusters(CLUSTERS_FILE_PATH, EVENT_ID)
    particle_to_cluster = build_particle_to_cluster_map(clusters)

    print(f"Event ID {EVENT_ID}: {particles.shape[0]} particles, {len(clusters)} jets found")

    plot_event(particles, particle_to_cluster, n_part, n_clusters=len(clusters))


if __name__ == "__main__":
    main()