#!/bin/bash
#
# Runs the OpenMP binary with different OMP_SCHEDULE settings and thread
# counts, and collects the total execution time into a CSV. Each run also
# gets its own HDF5 output file (via argv[1] of the binary) so the
# per-event `times` array saved by the C program is never overwritten by
# the next run.
#
# NOTE: this script writes one row per run (raw data). It does NOT average
# anything itself -- averaging/std-dev across the `run` column is left to
# post-processing (see the pandas snippet at the bottom of this file).
#
# Usage:
#   ./run_schedule.sh [path_to_binary] [num_runs_per_config]
#
# Example:
#   ./run_schedule.sh /mnt/POD/MCP_GD/JetClusteringAlgorithm/bin/openmp_version_base_schedule 5
#

set -euo pipefail

BINARY="${1:-./bin/openmp_version}"
RUNS="${2:-1}"                     # how many repetitions per configuration (for averaging later)
OUTPUT_CSV="/mnt/POD/MCP_GD/JetClusteringAlgorithm/benchmarks/benchmark_results_openmp_schedules_threads.csv"
OUTPUT_H5_DIR="/mnt/POD/MCP_GD/JetClusteringAlgorithm/data/results/openmp/benchmark_runs"

# schedules to test: "name,chunk" (chunk empty = let OpenMP decide default chunk)
SCHEDULES=(
    "static,"
    "static,1"
    "static,10"
    "static,100"
    "static,1000"
    "dynamic,"
    "dynamic,10"
    "dynamic,100"
    "dynamic,1000"
    "guided,"
    "guided,10"
    "guided,100"
    "guided,1000"
)

# thread counts to test
THREAD_COUNTS=(1 2 4 8 12 15 16 20)

if [[ ! -x "$BINARY" ]]; then
    echo "Error: binary '$BINARY' not found or not executable." >&2
    echo "Build it first (e.g. 'make openmp') or pass the correct path as \$1." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"
mkdir -p "$OUTPUT_H5_DIR"

echo "Binary:          $BINARY"
echo "Runs per config: $RUNS"
echo "CSV output:      $OUTPUT_CSV"
echo "HDF5 per-run dir: $OUTPUT_H5_DIR"
echo ""

# CSV header (added h5_file column so each row also points at its times[] file)
echo "schedule,chunk,threads,run,time_sec,h5_file" > "$OUTPUT_CSV"

for sched_entry in "${SCHEDULES[@]}"; do
    sched_name="${sched_entry%%,*}"
    sched_chunk="${sched_entry#*,}"

    if [[ -n "$sched_chunk" ]]; then
        omp_schedule="${sched_name},${sched_chunk}"
        chunk_label="${sched_chunk}"
    else
        omp_schedule="${sched_name}"
        chunk_label="default"
    fi

    for threads in "${THREAD_COUNTS[@]}"; do
        for run in $(seq 1 "$RUNS"); do

            echo "Running: schedule=$omp_schedule  threads=$threads  run=$run"

            h5_out="${OUTPUT_H5_DIR}/clusters_${sched_name}_${chunk_label}_t${threads}_r${run}.h5"

            # capture stdout, extract the execution time line
            output=$(OMP_SCHEDULE="$omp_schedule" OMP_NUM_THREADS="$threads" "$BINARY" "$h5_out" 2>&1) || {
                echo "  -> run failed, skipping" >&2
                continue
            }

            time_sec=$(echo "$output" | grep -oP 'Execution time \(total.*?\):\s*\K[0-9]+\.[0-9]+')

            if [[ -z "$time_sec" ]]; then
                echo "  -> could not parse execution time, skipping" >&2
                continue
            fi

            echo "  -> ${time_sec}s (times[] saved in $h5_out)"

            echo "${sched_name},${chunk_label},${threads},${run},${time_sec},${h5_out}" >> "$OUTPUT_CSV"

        done
    done
done

echo ""
echo "Done. Raw per-run results saved to $OUTPUT_CSV"
echo "Per-run times[] arrays saved under $OUTPUT_H5_DIR (dataset /times in each .h5 file)"

# ---------------------------------------------------------------------------
# Post-processing (run separately, not part of this script):
#
#   import pandas as pd
#   df = pd.read_csv("benchmark_results_openmp_schedules_threads.csv")
#   summary = (
#       df.groupby(["schedule", "chunk", "threads"])["time_sec"]
#         .agg(["mean", "std", "count"])
#   )
#   print(summary)
# ---------------------------------------------------------------------------