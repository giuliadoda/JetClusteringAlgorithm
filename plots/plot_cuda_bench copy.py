"""
plot_benchmarks.py

Genera grafici a partire dai CSV prodotti da benchmark_cuda.sh:
  - results_summary.csv (media + std per combinazione thr_block/n_events)
  - results_raw.csv      (opzionale, una riga per run - usato solo se
                           passato esplicitamente con --raw)

Grafici prodotti (salvati in --outdir):
  1. kernel_ms_mean vs n_events, una linea per thr_block
  2. kernel_ms_mean vs thr_block, una linea per n_events
  3. wall_time_sec_mean vs n_events, una linea per thr_block
  4. breakdown H2D / kernel / D2H (stacked bar) vs thr_block, per un
     n_events scelto (--breakdown-nevents, default: il massimo presente)
  5. avg_event_ms_mean vs thr_block, una linea per n_events

Tutti i grafici usano le colonne *_std come barre di errore quando
disponibili.

USO:
    python plot_benchmarks.py --summary results_summary.csv --outdir plots/
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_vs_x(df, x_col, group_col, y_mean_col, y_std_col,
                     xlabel, ylabel, title, outpath, logx=False, logy=False):
    fig, ax = plt.subplots(figsize=(8, 6))

    for group_val, sub in df.groupby(group_col):
        sub = sub.sort_values(x_col)
        ax.errorbar(
            sub[x_col], sub[y_mean_col], yerr=sub[y_std_col],
            marker="o", capsize=3, label=f"{group_col}={group_val}",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.legend(title=group_col, fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Salvato: {outpath}")


def plot_breakdown(df, n_events_value, outpath):
    sub = df[df["n_events"] == n_events_value].sort_values("thr_block")

    if sub.empty:
        print(f"Nessun dato per n_events={n_events_value}, breakdown non generato.")
        return

    thr_labels = sub["thr_block"].astype(str).tolist()
    h2d = sub["h2d_ms_mean"].values
    kernel = sub["kernel_ms_mean"].values
    d2h = sub["d2h_ms_mean"].values

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(thr_labels, h2d, label="H2D")
    ax.bar(thr_labels, kernel, bottom=h2d, label="Kernel")
    ax.bar(thr_labels, d2h, bottom=h2d + kernel, label="D2H")

    ax.set_xlabel("THR_BLOCK")
    ax.set_ylabel("Tempo (ms)")
    ax.set_title(f"Breakdown tempi per THR_BLOCK - N_EVENTS={n_events_value}")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Salvato: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici dai CSV dei benchmark del programma CUDA."
    )
    parser.add_argument("--summary", required=True,
                        help="Path a results_summary.csv")
    parser.add_argument("--raw", default=None,
                        help="Path a results_raw.csv (opzionale, non usato per ora "
                             "ma disponibile per estensioni future / analisi custom).")
    parser.add_argument("--outdir", default="plots",
                        help="Cartella di output per le figure (default: plots/)")
    parser.add_argument("--breakdown-nevents", type=int, default=None,
                        help="Valore di n_events da usare per il grafico breakdown "
                             "(default: il massimo presente nel csv).")
    parser.add_argument("--logx", action="store_true",
                        help="Usa scala log sull'asse x per i grafici vs n_events.")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.summary)

    required_cols = {
        "thr_block", "n_events",
        "wall_time_sec_mean", "wall_time_sec_std",
        "h2d_ms_mean", "h2d_ms_std",
        "kernel_ms_mean", "kernel_ms_std",
        "d2h_ms_mean", "d2h_ms_std",
        "avg_event_ms_mean", "avg_event_ms_std",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {missing}")

    # 1. kernel time vs n_events, una linea per thr_block
    plot_metric_vs_x(
        df, x_col="n_events", group_col="thr_block",
        y_mean_col="kernel_ms_mean", y_std_col="kernel_ms_std",
        xlabel="N_EVENTS", ylabel="Kernel time (ms)",
        title="Tempo kernel vs numero di eventi",
        outpath=os.path.join(args.outdir, "kernel_vs_nevents.png"),
        logx=args.logx,
    )

    # 2. kernel time vs thr_block, una linea per n_events
    plot_metric_vs_x(
        df, x_col="thr_block", group_col="n_events",
        y_mean_col="kernel_ms_mean", y_std_col="kernel_ms_std",
        xlabel="THR_BLOCK", ylabel="Kernel time (ms)",
        title="Tempo kernel vs threads per block",
        outpath=os.path.join(args.outdir, "kernel_vs_thrblock.png"),
    )

    # 3. wall time vs n_events, una linea per thr_block
    plot_metric_vs_x(
        df, x_col="n_events", group_col="thr_block",
        y_mean_col="wall_time_sec_mean", y_std_col="wall_time_sec_std",
        xlabel="N_EVENTS", ylabel="Wall time (sec)",
        title="Tempo totale (wall-clock) vs numero di eventi",
        outpath=os.path.join(args.outdir, "walltime_vs_nevents.png"),
        logx=args.logx,
    )

    # 4. breakdown H2D/kernel/D2H per un n_events specifico
    n_events_value = args.breakdown_nevents
    if n_events_value is None:
        n_events_value = int(df["n_events"].max())
    plot_breakdown(
        df, n_events_value,
        outpath=os.path.join(args.outdir, f"breakdown_nevents_{n_events_value}.png"),
    )

    # 5. avg event time vs thr_block, una linea per n_events
    plot_metric_vs_x(
        df, x_col="thr_block", group_col="n_events",
        y_mean_col="avg_event_ms_mean", y_std_col="avg_event_ms_std",
        xlabel="THR_BLOCK", ylabel="Tempo medio per evento (ms)",
        title="Tempo medio per evento vs threads per block",
        outpath=os.path.join(args.outdir, "avgevent_vs_thrblock.png"),
    )

    print("\nTutti i grafici sono stati generati.")


if __name__ == "__main__":
    main()