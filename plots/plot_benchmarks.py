"""
plot_benchmarks.py

Reads the benchmark CSV produced by run_schedule.sh
(columns: schedule,chunk,threads,run,time_sec,h5_file) and produces two
families of plots, one figure per OpenMP schedule type (static/dynamic/guided):

  1) execution time  vs  number of chunks
       - one line per thread count
       - saved as: exectime_vs_nchunks_<schedule>.png

  2) speedup  vs  number of threads
       - one line per chunk size
       - saved as: speedup_vs_threads_<schedule>.png

Usage:
    python3 plot_benchmarks.py \
        --csv /mnt/POD/MCP_GD/JetClusteringAlgorithm/benchmarks/benchmark_results_openmp_schedules_threads.csv \
        --n-events 100000 \
        --outdir ./figures

Notes / assumptions (read before trusting the numbers):

  * "chunk" in the CSV is the OMP chunk SIZE (events per chunk), not the
    number of chunks. To get "number of chunks" (as requested) we compute
        n_chunks = ceil(n_events / chunk_size)
    which requires knowing n_events (N_EVENTS in the C code). Pass it with
    --n-events. If you don't pass it, the script falls back to plotting
    against the raw chunk size instead of number of chunks (a warning is
    printed).

  * Rows where chunk == "default" (i.e. OMP_SCHEDULE was set without an
    explicit chunk, e.g. "static" or "dynamic" with no number) have no
    well-defined chunk size from the CSV alone -- OpenMP picks it internally
    (for static, default chunk ~= n_events/threads; for dynamic/guided,
    default chunk = 1). These rows are EXCLUDED from plot (1), since they
    would need a threads-dependent x position. They ARE still usable as a
    reference in plot (2) if you want (currently also excluded there for
    consistency -- see INCLUDE_DEFAULT_CHUNK below).

  * Speedup for plot (2) is computed per (schedule, chunk) group as
        speedup(threads) = mean_time(threads=1) / mean_time(threads)
    i.e. relative to that same schedule+chunk configuration run serially
    within the sweep (NOT relative to a separately-timed single-threaded
    baseline binary). If you have a true serial baseline you'd rather use,
    set BASELINE_THREADS / adapt get_speedup() below.

  * When --n-runs-per-config > 1 in your sweep, this script averages
    time_sec across the 'run' column (mean) before computing anything else.
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# If True, rows with chunk == "default" are kept and plotted using OpenMP's
# implicit default chunk size approximation instead of being dropped.
# Left False by default because the "default" chunk size for static
# schedules depends on the thread count, which complicates comparisons.
INCLUDE_DEFAULT_CHUNK = False

SCHEDULES_ORDER = ["static", "dynamic", "guided"]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"schedule", "chunk", "threads", "run", "time_sec"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"Error: CSV is missing expected columns: {missing}")

    return df


def add_chunk_size_column(df: pd.DataFrame, n_events: int | None) -> pd.DataFrame:
    """Adds a numeric 'chunk_size' column (NaN for 'default' rows unless
    INCLUDE_DEFAULT_CHUNK handling is added)."""
    df = df.copy()

    def to_numeric_chunk(v):
        if v == "default":
            return math.nan
        try:
            return float(v)
        except (TypeError, ValueError):
            return math.nan

    df["chunk_size"] = df["chunk"].apply(to_numeric_chunk)

    if not INCLUDE_DEFAULT_CHUNK:
        n_dropped = df["chunk_size"].isna().sum()
        if n_dropped:
            print(f"Note: dropping {n_dropped} row(s) with chunk == 'default' "
                  f"(set INCLUDE_DEFAULT_CHUNK = True to keep them).")
        df = df.dropna(subset=["chunk_size"])

    if n_events is not None:
        df["n_chunks"] = (n_events / df["chunk_size"]).apply(math.ceil)
    else:
        df["n_chunks"] = df["chunk_size"]  # fallback: plot raw chunk size

    return df


def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Averages time_sec over repeated runs for each (schedule, chunk, threads)."""
    grouped = (
        df.groupby(["schedule", "chunk_size", "n_chunks", "threads"])["time_sec"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "time_mean", "std": "time_std", "count": "n_runs"})
    )
    return grouped


def plot_exectime_vs_nchunks(agg: pd.DataFrame, outdir: Path, using_n_chunks: bool):
    xlabel = "Number of chunks" if using_n_chunks else "Chunk size (fallback, --n-events not given)"

    for sched in SCHEDULES_ORDER:
        sub = agg[agg["schedule"] == sched]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for threads, group in sub.groupby("threads"):
            group = group.sort_values("n_chunks")
            ax.plot(
                group["n_chunks"], group["time_mean"],
                marker="o", label=f"{threads} threads",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Execution time (s)")
        ax.set_title(f"Execution time vs {xlabel.lower()} — schedule = {sched}")
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(title="Threads")
        fig.tight_layout()

        out_path = outdir / f"exectime_vs_nchunks_{sched}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def compute_speedup(agg: pd.DataFrame, baseline_threads: int = 1) -> pd.DataFrame:
    agg = agg.copy()
    speedup_rows = []

    for (sched, chunk_size), group in agg.groupby(["schedule", "chunk_size"]):
        baseline_rows = group[group["threads"] == baseline_threads]
        if baseline_rows.empty:
            print(f"Warning: no threads={baseline_threads} baseline for "
                  f"schedule={sched}, chunk={chunk_size}; skipping speedup for this group.")
            continue

        t_baseline = baseline_rows["time_mean"].iloc[0]

        group = group.copy()
        group["speedup"] = t_baseline / group["time_mean"]
        speedup_rows.append(group)

    if not speedup_rows:
        return pd.DataFrame(columns=list(agg.columns) + ["speedup"])

    return pd.concat(speedup_rows, ignore_index=True)


def plot_speedup_vs_threads(speedup_df: pd.DataFrame, outdir: Path):
    for sched in SCHEDULES_ORDER:
        sub = speedup_df[speedup_df["schedule"] == sched]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for chunk_size, group in sub.groupby("chunk_size"):
            group = group.sort_values("threads")
            ax.plot(
                group["threads"], group["speedup"],
                marker="o", label=f"chunk = {int(chunk_size)}",
            )

        # ideal linear speedup reference line
        max_threads = sub["threads"].max()
        ax.plot([1, max_threads], [1, max_threads], linestyle="--", color="gray",
                alpha=0.6, label="ideal (linear)")

        ax.set_xlabel("Number of threads")
        ax.set_ylabel("Speedup (T1 / Tn, same schedule+chunk)")
        ax.set_title(f"Speedup vs number of threads — schedule = {sched}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Chunk size")
        fig.tight_layout()

        out_path = outdir / f"speedup_vs_threads_{sched}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, type=Path,
                         help="Path to benchmark_results_openmp_schedules_threads.csv")
    parser.add_argument("--n-events", type=int, default=None,
                         help="N_EVENTS used in the C benchmark (needed to convert "
                              "chunk size -> number of chunks). If omitted, plot (1) "
                              "falls back to raw chunk size on the x-axis.")
    parser.add_argument("--baseline-threads", type=int, default=1,
                         help="Thread count used as the serial baseline for speedup (default: 1)")
    parser.add_argument("--outdir", type=Path, default=Path("./plots"),
                         help="Directory where PNG plots are saved (default: ./plots)")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"Error: CSV not found at {args.csv}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv)
    df = add_chunk_size_column(df, args.n_events)

    agg = aggregate_runs(df)

    plot_exectime_vs_nchunks(agg, args.outdir, using_n_chunks=args.n_events is not None)

    speedup_df = compute_speedup(agg, baseline_threads=args.baseline_threads)
    plot_speedup_vs_threads(speedup_df, args.outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()