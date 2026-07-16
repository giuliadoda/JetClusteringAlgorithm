import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D 


# file HDF5 con i dati grezzi (lo stesso letto dal programma C, FILE_PATH)
RAW_FILE_PATH = "/mnt/POD/MCP_GD/JetClusteringAlgorithm/data/events_anomalydetection_Z_XY_qqq.h5"
# path del dataset dentro il file raw (corrisponde a DATA_PATH in constants.h)
RAW_DATASET_PATH = "/df/block0_values"

# file prodotto dal clustering (corrisponde a quello creato con H5Fcreate nel main)
CLUSTERS_FILE_PATH = "data/results/serial/clusters.h5"

# numero di feature per particella nel file raw: p_t, eta, phi (N_FEAT in constants.h)
N_FEAT = 3
# numero massimo di particelle per evento (MAX_P in constants.h)
MAX_P = 700

# >>> ID DELL'EVENTO DA VISUALIZZARE <<<
EVENT_ID = 0

# larghezza delle barre nel piano eta-phi (per il plot 3D a barre)
BAR_WIDTH_ETA = 0.08
BAR_WIDTH_PHI = 0.08

# --- parametri per la heatmap 2D ---
HEATMAP_BINS_ETA = 100
HEATMAP_BINS_PHI = 100
ETA_RANGE = (-5.0, 5.0)     # intervallo tipico di eta ai collider LHC
PHI_RANGE = (-np.pi, np.pi)  # phi e' periodico in [-pi, pi]

# =====================================================================


def read_raw_particles(raw_file_path, dataset_path, event_id, max_p, n_feat):
    """
    Legge la riga corrispondente a event_id dal dataset grezzo e la
    ricostruisce come array (n_particelle, n_feat), fermandosi alla
    prima particella con p_t == 0 (stesso criterio usato in read_event
    nel codice C).
    """
    with h5py.File(raw_file_path, "r") as f:
        dset = f[dataset_path]
        row = dset[event_id, :]  # shape: (MAX_P * N_FEAT,)

    row = np.asarray(row, dtype=np.float64).reshape(max_p, n_feat)

    n_part = 0
    while n_part < max_p and row[n_part, 0] != 0.0:
        n_part += 1

    particles = row[:n_part, :]  # colonne: p_t, eta, phi, ...
    return particles


def read_clusters(clusters_file_path, event_id):
    """
    Legge, per un dato evento, tutti i gruppi cluster_* dal file di output
    e restituisce una lista di dict:
        {"cluster_id": int, "components": np.array(int), "kinematics": (p_t, eta, phi)}
    """
    clusters = []
    event_name = f"/event_{event_id}"

    with h5py.File(clusters_file_path, "r") as f:
        if event_name not in f:
            raise KeyError(f"Gruppo '{event_name}' non trovato in {clusters_file_path}")

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

            # l'attributo cluster_id e' salvato sul gruppo del cluster
            if "cluster_id" in cluster_group.attrs:
                cluster_id = int(cluster_group.attrs["cluster_id"])

            kinematics = cluster_group["kinematics"][()] if "kinematics" in cluster_group else None

            clusters.append({
                "cluster_id": cluster_id,
                "components": np.asarray(components, dtype=int),
                "kinematics": kinematics,
            })

    # ordina per cluster_id cosi' il colore e' sempre coerente
    clusters.sort(key=lambda c: c["cluster_id"])
    return clusters


def build_particle_to_cluster_map(clusters):
    """Associa ad ogni indice di particella l'indice del cluster (jet) a cui appartiene."""
    mapping = {}
    for cluster in clusters:
        for particle_idx in cluster["components"]:
            mapping[int(particle_idx)] = cluster["cluster_id"]
    return mapping


def plot_event(particles, particle_to_cluster, event_id, n_clusters):
    """Disegna il lego plot 3D: eta-phi sul piano, altezza = p_t, colore = cluster."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = cm.get_cmap("tab20", max(n_clusters, 1))

    plotted_clusters = set()

    for p_idx in range(particles.shape[0]):
        p_t, eta, phi = particles[p_idx, 0], particles[p_idx, 1], particles[p_idx, 2]

        cluster_id = particle_to_cluster.get(p_idx, -1)
        color = cmap(cluster_id % cmap.N) if cluster_id >= 0 else "gray"

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
            shade=True,
            label=label,
        )

    ax.set_xlabel("eta")
    ax.set_ylabel("phi")
    ax.set_zlabel("p_t (momento)")
    ax.set_title(f"Jet clustering - evento {event_id}")

    # legenda senza duplicati
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_event_heatmap(particles, particle_to_cluster, event_id, n_clusters):
    """
    Heatmap 2D nel piano eta-phi: il colore di sfondo rappresenta il momento
    (p_t) depositato in ciascuna cella della griglia (come un calorimetro).
    Sopra vengono sovrapposti i marker delle singole particelle colorati per
    cluster (jet), cosi' da mantenere anche la distinzione tra jet diversi.
    """
    eta = particles[:, 1]
    phi = particles[:, 2]
    p_t = particles[:, 0]

    # istogramma 2D pesato: in ogni bin (eta, phi) si somma il p_t delle
    # particelle che ci cadono dentro
    heat, eta_edges, phi_edges = np.histogram2d(
        eta, phi,
        bins=[HEATMAP_BINS_ETA, HEATMAP_BINS_PHI],
        range=[ETA_RANGE, PHI_RANGE],
        weights=p_t,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    mesh = ax.pcolormesh(
        eta_edges, phi_edges, heat.T,
        cmap="viridis",
        shading="auto",
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("p_t (momento) per bin")

    # marker delle singole particelle, colorati per cluster
    cmap_clusters = cm.get_cmap("tab20", max(n_clusters, 1))
    plotted_clusters = set()

    for p_idx in range(particles.shape[0]):
        cluster_id = particle_to_cluster.get(p_idx, -1)
        color = cmap_clusters(cluster_id % cmap_clusters.N) if cluster_id >= 0 else "white"
        label = f"Jet {cluster_id}" if cluster_id not in plotted_clusters and cluster_id >= 0 else None
        plotted_clusters.add(cluster_id)

        ax.scatter(
            eta[p_idx], phi[p_idx],
            s=30 + 200 * (p_t[p_idx] / max(p_t.max(), 1e-9)),  # dimensione ~ p_t
            color=color,
            edgecolors="black",
            linewidths=0.6,
            label=label,
        )

    ax.set_xlabel("eta")
    ax.set_ylabel("phi")
    ax.set_title(f"Heatmap jet clustering - evento {event_id}")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def main():

    particles = read_raw_particles(RAW_FILE_PATH, RAW_DATASET_PATH, EVENT_ID, MAX_P, N_FEAT)
    clusters = read_clusters(CLUSTERS_FILE_PATH, EVENT_ID)
    particle_to_cluster = build_particle_to_cluster_map(clusters)

    print(f"Evento {EVENT_ID}: {particles.shape[0]} particelle, {len(clusters)} jet trovati")

    plot_event(particles, particle_to_cluster, EVENT_ID, n_clusters=len(clusters))
    plot_event_heatmap(particles, particle_to_cluster, EVENT_ID, n_clusters=len(clusters))


if __name__ == "__main__":
    main()