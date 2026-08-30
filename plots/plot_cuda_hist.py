import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 

# total events number
N_EVENTS = 100000

# event ID to visualize
# EVENT_ID = np.random.randint(N_EVENTS)
EVENT_ID = 12623

N_FEAT = 3
MAX_P = 700

# raw data
RAW_DATA = "/mnt/POD/MCP_GD/JetClusteringAlgorithm/data/events_anomalydetection_Z_XY_qqq.h5"

# results
RESULTS = "/mnt/POD/MCP_GD/JetClusteringAlgorithm/data/results/cuda/clusters.h5"
DEFAULT_INPUT_DATASET = "/df/block0_values"
DEFAULT_CLUSTER_DATASET = "/cluster_trace"

OUTPUT = f"/mnt/POD/MCP_GD/JetClusteringAlgorithm/plots/figures/cuda/cluster_event_{EVENT_ID}.png"

# find root particle
def find_root(idx, trace):

    while trace[idx] != idx:

        idx = trace[idx]

    return idx

# get cluster
def resolve_clusters(trace, valid_mask):
    
    n = len(trace)
    cluster_id = np.full(n, -1, dtype=np.int64)

    for p in range(n):
        if not valid_mask[p]:
            continue
        cluster_id[p] = find_root(p, trace)

    return cluster_id

# get event data
def load_event(input_path, cluster_path, event_idx, max_p, n_feat,
                input_dataset, cluster_dataset):
    
    with h5py.File(input_path, "r") as f_in:
        dset = f_in[input_dataset]
        row = dset[event_idx]

    row = np.asarray(row[:-1], dtype=np.float64).reshape(max_p, n_feat)
    pt = row[:, 0]
    eta = row[:, 1]
    phi = row[:, 2]

    with h5py.File(cluster_path, "r") as f_cl:
        trace = f_cl[cluster_dataset][event_idx]

    trace = np.asarray(trace, dtype=np.int64)

    valid_mask = pt != 0.0

    return pt, eta, phi, trace, valid_mask


def plot_event(pt, eta, phi, trace, valid_mask, event_idx, output=None):

    cluster_id = resolve_clusters(trace, valid_mask)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        print(f"Event {event_idx}: no particle found.",
              file=sys.stderr)
        return

    unique_clusters = np.unique(cluster_id[valid_idx])
    n_clusters = len(unique_clusters)

    cmap = cm.get_cmap("tab20", max(n_clusters, 1))
    color_map = {c: cmap(i % 20) for i, c in enumerate(unique_clusters)}

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    dx = dy = 0.1

    for c in unique_clusters:
        mask = cluster_id[valid_idx] == c
        idx = valid_idx[mask]

        xs = eta[idx]
        ys = phi[idx]
        zs = np.zeros_like(xs)
        heights = pt[idx]

        ax.bar3d(
            xs - dx / 2, 
            ys - dy / 2, 
            zs,
            dx, 
            dy, 
            heights,
            color=color_map[c],
            shade=True
        )

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_zlabel(r"$p_T$ (GeV)")
    ax.set_title(f"Event ID {event_idx} - # Particles: {len(valid_idx)}, # Clusters found: {n_clusters} ")

    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=300)
        print(f"Plot saved.")
    else:
        plt.show()



if __name__ == "__main__":

    pt, eta, phi, trace, valid_mask = load_event(
            RAW_DATA, RESULTS, EVENT_ID, MAX_P, N_FEAT,
            DEFAULT_INPUT_DATASET, DEFAULT_CLUSTER_DATASET
        )
    
    plot_event(pt, eta, phi, trace, valid_mask, EVENT_ID,
                   output=OUTPUT)