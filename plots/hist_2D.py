import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  

# event ID to visualize
EVENT_ID = 111

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
BAR_WIDTH_ETA = 0.08
BAR_WIDTH_PHI = 0.08

# heatmap settings
HEATMAP_BINS_ETA = 100
HEATMAP_BINS_PHI = 100
ETA_RANGE = (-8.0, 8.0)    # should be automatic
PHI_RANGE = (-np.pi, np.pi)  

# plot directory
SAVE_DIR = f"figures/{VERSION}" # make sure it exists
os.makedirs(SAVE_DIR, exist_ok=True)

# read event from raw data
def read_raw_particles(raw_file_path, dataset_path, event_id, max_p, n_feat):

    n_cols = max_p * n_feat

    with h5py.File(raw_file_path, "r") as f:
        dset = f[dataset_path]
        row = dset[event_id, :]  

    row = np.asarray(row, dtype=np.float64)

    label = row[-1]
    row = row[:n_cols]
    row = row.reshape(max_p, n_feat)

    # change here when the c code will save also this **
    n_part = 0
    while n_part < max_p and row[n_part, 0] != 0.0:
        n_part += 1

    particles = row[:n_part, :] 
    return particles

# read clustering results --> change also here after updating the c code **
def read_clusters(clusters_file_path, event_id):

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
def build_particle_to_cluster_map(clusters):
    mapping = {}
    for cluster in clusters:
        for particle_idx in cluster["components"]:
            mapping[int(particle_idx)] = cluster["cluster_id"]
    return mapping

# plot 2D histogram
def plot_event(particles, particle_to_cluster, event_id, n_clusters, isLegend=False):

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = cm.get_cmap("tab20", max(n_clusters, 1))

    plotted_clusters = set()

    for p_idx in range(particles.shape[0]):

        p_t, eta, phi = particles[p_idx, 0], particles[p_idx, 1], particles[p_idx, 2]

        cluster_id = particle_to_cluster.get(p_idx, -1)
        color = cmap(cluster_id % cmap.N) if cluster_id >= 0 else "gray"

        if isLegend:
            label = f"Jet {cluster_id}" if cluster_id not in plotted_clusters and cluster_id >= 0 else None

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
    ax.set_title(f"Event ID: {event_id}") # add how many cluster founds, how many initial particles

    if isLegend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=8)

    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, f"jets_3d_event_{event_id}.png")
    fig.savefig(save_path, dpi=200)

    plt.show()

# plot heatmap
def plot_event_heatmap(particles, particle_to_cluster, event_id, n_clusters):

    eta = particles[:, 1]
    phi = particles[:, 2]
    p_t = particles[:, 0]

    # for each bin sum momenta of particles inside the bin
    heat, eta_edges, phi_edges = np.histogram2d(
        eta, phi,
        bins=[HEATMAP_BINS_ETA, HEATMAP_BINS_PHI],
        range=[ETA_RANGE, PHI_RANGE],
        weights=p_t,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    mesh = ax.pcolormesh(
        eta_edges, phi_edges, heat.T,
        cmap="plasma",
        shading="auto",
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$p_T$ (GeV)")

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(f"Event ID {event_id}")

    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, f"jets_heatmap_event_{event_id}.png")
    fig.savefig(save_path, dpi=200)

    plt.show()


def main():

    particles = read_raw_particles(RAW_FILE_PATH, RAW_DATASET_PATH, EVENT_ID, MAX_P, N_FEAT)
    clusters = read_clusters(CLUSTERS_FILE_PATH, EVENT_ID)
    particle_to_cluster = build_particle_to_cluster_map(clusters)

    print(f"Event ID {EVENT_ID}: {particles.shape[0]} particles, {len(clusters)} jets found")

    plot_event(particles, particle_to_cluster, EVENT_ID, n_clusters=len(clusters))
    plot_event_heatmap(particles, particle_to_cluster, EVENT_ID, n_clusters=len(clusters))


if __name__ == "__main__":
    main()