"""
plot_clusters.py

Visualizza il risultato del clustering per un singolo evento come
istogramma 2D nel piano (eta, phi), con altezza delle barre pari a p_t
e colore diverso per ogni cluster (jet) finale.

Il file cluster_trace non fa path-compression durante il merging: una
particella j che viene assorbita in i punta a i, ma se successivamente
i viene assorbito in k, clusters_trace[j] resta i (non k). Per questo
motivo qui si risolve la catena fino alla radice (find con path
compression) per ottenere l'ID di cluster finale corretto di ogni
particella.

USO:
    python plot_clusters.py \
        --input /path/to/events_anomalydetection_Z_XY_qqq.h5 \
        --clusters /path/to/clusters.h5 \
        --event 0 \
        --max-p 700 --n-feat 3 \
        --output cluster_event0.png

Se --output non è specificato, la figura viene mostrata a schermo
(plt.show()) invece di essere salvata.
"""

import argparse
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (necessario per projection='3d')


# dataset path di default nel file di input, coerenti con constants.h
DEFAULT_INPUT_DATASET = "/df/block0_values"
DEFAULT_CLUSTER_DATASET = "/cluster_trace"


def find_root(idx, trace):
    """
    Risolve la catena di parent (union-find 'find') per un singolo
    indice, seguendo trace[idx] finché non si raggiunge un punto fisso
    (trace[root] == root). Ritorna la radice.
    """
    while trace[idx] != idx:
        idx = trace[idx]
    return idx


def resolve_clusters(trace, valid_mask):
    """
    Per ogni particella valida (p_t != 0) risolve il cluster finale
    (radice della catena). Ritorna un array di interi con lo stesso
    ordine dell'array 'trace': cluster_id[p] = radice finale di p.
    Le particelle non valide vengono ignorate (restano -1).
    """
    n = len(trace)
    cluster_id = np.full(n, -1, dtype=np.int64)

    for p in range(n):
        if not valid_mask[p]:
            continue
        cluster_id[p] = find_root(p, trace)

    return cluster_id


def load_event(input_path, cluster_path, event_idx, max_p, n_feat,
                input_dataset, cluster_dataset):
    """
    Carica pt/eta/phi e cluster_trace per un singolo evento.
    """
    with h5py.File(input_path, "r") as f_in:
        dset = f_in[input_dataset]
        row = dset[event_idx]  # shape (max_p * n_feat,)

    row = np.asarray(row, dtype=np.float64).reshape(max_p, n_feat)
    pt = row[:, 0]
    eta = row[:, 1]
    phi = row[:, 2]

    with h5py.File(cluster_path, "r") as f_cl:
        trace = f_cl[cluster_dataset][event_idx]  # shape (max_p,)

    trace = np.asarray(trace, dtype=np.int64)

    # una particella e' valida se pt != 0 (stessa convenzione del kernel CUDA,
    # che interrompe il riempimento non appena trova p_t == 0)
    valid_mask = pt != 0.0

    return pt, eta, phi, trace, valid_mask


def plot_event(pt, eta, phi, trace, valid_mask, event_idx, output=None,
               elev=35, azim=-60):

    cluster_id = resolve_clusters(trace, valid_mask)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        print(f"Evento {event_idx}: nessuna particella valida, niente da plottare.",
              file=sys.stderr)
        return

    unique_clusters = np.unique(cluster_id[valid_idx])
    n_clusters = len(unique_clusters)

    # mappa cluster -> colore
    cmap = cm.get_cmap("tab20", max(n_clusters, 1))
    color_map = {c: cmap(i % 20) for i, c in enumerate(unique_clusters)}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # larghezza delle barre nel piano eta-phi (piccola, solo per visibilita')
    dx = dy = 0.05

    for c in unique_clusters:
        mask = cluster_id[valid_idx] == c
        idx = valid_idx[mask]

        xs = eta[idx]
        ys = phi[idx]
        zs = np.zeros_like(xs)
        heights = pt[idx]

        ax.bar3d(
            xs - dx / 2, ys - dy / 2, zs,
            dx, dy, heights,
            color=color_map[c],
            shade=True,
            alpha=0.9,
            label=f"cluster {c}",
        )

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_zlabel(r"$p_T$")
    ax.set_title(f"Clustering risultato - evento {event_idx} "
                 f"({n_clusters} cluster, {len(valid_idx)} particelle)")

    ax.view_init(elev=elev, azim=azim)

    # legenda: se ci sono molti cluster diventa illeggibile, la mostriamo
    # solo se sono pochi
    if n_clusters <= 20:
        # bar3d non supporta bene le proxy artist per la legenda, quindi
        # costruiamo manualmente delle patch fittizie
        from matplotlib.patches import Patch
        handles = [Patch(color=color_map[c], label=f"cluster {c}")
                   for c in unique_clusters]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1.0),
                  fontsize=8, ncol=1)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Figura salvata in: {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualizza il risultato del clustering (eta-phi-pt) per un evento."
    )
    parser.add_argument("--input", required=True,
                        help="Path al file HDF5 originale con i dati degli eventi.")
    parser.add_argument("--clusters", required=True,
                        help="Path al file clusters.h5 prodotto dal programma CUDA.")
    parser.add_argument("--event", type=int, default=0,
                        help="Indice dell'evento da visualizzare (default: 0).")
    parser.add_argument("--max-p", type=int, default=700,
                        help="Numero massimo di particelle per evento (MAX_P, default 700).")
    parser.add_argument("--n-feat", type=int, default=3,
                        help="Numero di feature per particella (N_FEAT, default 3: pt,eta,phi).")
    parser.add_argument("--input-dataset", default=DEFAULT_INPUT_DATASET,
                        help=f"Path del dataset nel file di input (default {DEFAULT_INPUT_DATASET}).")
    parser.add_argument("--cluster-dataset", default=DEFAULT_CLUSTER_DATASET,
                        help=f"Path del dataset cluster_trace (default {DEFAULT_CLUSTER_DATASET}).")
    parser.add_argument("--output", default=None,
                        help="Path del file immagine di output. Se omesso, mostra la figura a schermo.")
    parser.add_argument("--elev", type=float, default=35,
                        help="Elevazione della vista 3D (default 35).")
    parser.add_argument("--azim", type=float, default=-60,
                        help="Azimuth della vista 3D (default -60).")

    args = parser.parse_args()

    pt, eta, phi, trace, valid_mask = load_event(
        args.input, args.clusters, args.event, args.max_p, args.n_feat,
        args.input_dataset, args.cluster_dataset,
    )

    plot_event(pt, eta, phi, trace, valid_mask, args.event,
               output=args.output, elev=args.elev, azim=args.azim)


if __name__ == "__main__":
    main()