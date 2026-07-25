#!/usr/bin/env python3
"""
Plot: tempo di esecuzione per evento vs numero di particelle nell'evento,
per la versione seriale del clustering.

Legge il CSV prodotto da functions.c (colonne: event_id,n_particles,time_sec)
e produce:
  - uno scatter plot (tempo vs n_particelle)
  - un plot con la media del tempo raggruppata per numero di particelle,
    utile per vedere l'andamento medio (es. se e' O(n^2) o O(n^3))

USO:
    python3 plot_time_vs_particles.py results/serial_time_vs_particles.csv
"""

import sys
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def load_data(csv_path):
    n_particles = []
    times = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_particles.append(int(row["n_particles"]))
            times.append(float(row["time_sec"]))

    return n_particles, times


def group_mean(n_particles, times):
    """Raggruppa i tempi per numero di particelle e calcola la media."""
    buckets = defaultdict(list)
    for n, t in zip(n_particles, times):
        buckets[n].append(t)

    n_sorted = sorted(buckets.keys())
    mean_times = [sum(buckets[n]) / len(buckets[n]) for n in n_sorted]

    return n_sorted, mean_times


def main():
    if len(sys.argv) != 2:
        print(f"Uso: {sys.argv[0]} <path_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    n_particles, times = load_data(csv_path)

    if not n_particles:
        print("Nessun dato trovato nel CSV.")
        sys.exit(1)

    n_grouped, t_mean = group_mean(n_particles, times)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # scatter: un punto per evento
    axes[0].scatter(n_particles, times, s=10, alpha=0.5)
    axes[0].set_xlabel("Numero di particelle nell'evento")
    axes[0].set_ylabel("Tempo di esecuzione (s)")
    axes[0].set_title("Tempo per evento vs n. particelle")
    axes[0].grid(True, alpha=0.3)

    # media per numero di particelle
    axes[1].plot(n_grouped, t_mean, marker="o", linestyle="-", markersize=4)
    axes[1].set_xlabel("Numero di particelle nell'evento")
    axes[1].set_ylabel("Tempo medio di esecuzione (s)")
    axes[1].set_title("Tempo medio vs n. particelle")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = csv_path.rsplit(".", 1)[0] + "_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"Grafico salvato in: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()