"""

CHECH THIS SCRIPT

plot_benchmarks.py

Reads the CSV produced by run_schedule.sh (columns: schedule,chunk,threads,run,time_sec)
and produces comparison plots:

  1. Execution time vs number of threads, one line per (schedule, chunk) combo.
  2. Speedup vs number of threads (relative to threads=1), one line per (schedule, chunk).
  3. A bar chart comparing the best time achieved by each schedule type.

Usage:
    python plot_benchmarks.py path/to/benchmark_results.csv [output_dir]

If output_dir is omitted, plots are saved next to the CSV in a "plots" subfolder.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # combine schedule + chunk into a single label for grouping/legend
    df["chunk"] = df["chunk"].astype(str)
    df["label"] = df["schedule"] + " (" + df["chunk"] + ")"

    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # average over repeated runs for the same (schedule, chunk, threads)
    grouped = (
        df.groupby(["schedule", "chunk", "label", "threads"], as_index=False)
        .agg(time_sec_mean=("time_sec", "mean"), time_sec_std=("time_sec", "std"))
    )
    return grouped


def plot_time_vs_threads(grouped: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, sub in grouped.groupby("label"):
        sub = sub.sort_values("threads")
        ax.errorbar(
            sub["threads"],
            sub["time_sec_mean"],
            yerr=sub["time_sec_std"],
            marker="o",
            markersize=4,
            capsize=3,
            label=label,
        )

    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Execution time (s)")
    ax.set_title("Execution time vs number of threads, by schedule")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "time_vs_threads.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_speedup_vs_threads(grouped: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    max_threads = grouped["threads"].max()

    for label, sub in grouped.groupby("label"):
        sub = sub.sort_values("threads")

        # baseline: time at threads == 1 for this label, if available
        baseline_rows = sub[sub["threads"] == 1]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows["time_sec_mean"].values[0]

        speedup = baseline / sub["time_sec_mean"]

        ax.plot(sub["threads"], speedup, marker="o", markersize=4, label=label)

    # ideal linear speedup reference line
    ax.plot(
        [1, max_threads],
        [1, max_threads],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Ideal linear speedup",
    )

    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Speedup (T1 / Tn)")
    ax.set_title("Speedup vs number of threads, by schedule")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "speedup_vs_threads.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_best_time_per_schedule(grouped: pd.DataFrame, out_dir: str):
    # for each schedule type (ignoring chunk), find its overall best (min) time
    # across all chunks and thread counts, and which config achieved it
    best_rows = []
    for schedule, sub in grouped.groupby("schedule"):
        best = sub.loc[sub["time_sec_mean"].idxmin()]
        best_rows.append(best)

    best_df = pd.DataFrame(best_rows).sort_values("time_sec_mean")

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(best_df["schedule"], best_df["time_sec_mean"])

    # annotate bars with the winning (chunk, threads) configuration
    for bar, (_, row) in zip(bars, best_df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"chunk={row['chunk']}\nthreads={int(row['threads'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Best execution time (s)")
    ax.set_title("Best execution time achieved per schedule type")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "best_time_per_schedule.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]

    if len(sys.argv) >= 3:
        out_dir = sys.argv[2]
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "plots")

    os.makedirs(out_dir, exist_ok=True)

    df = load_data(csv_path)
    grouped = aggregate(df)

    plot_time_vs_threads(grouped, out_dir)
    plot_speedup_vs_threads(grouped, out_dir)
    plot_best_time_per_schedule(grouped, out_dir)

    # also dump the aggregated table as CSV, handy for a quick look / report table
    summary_path = os.path.join(out_dir, "summary_aggregated.csv")
    grouped.sort_values(["schedule", "chunk", "threads"]).to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()