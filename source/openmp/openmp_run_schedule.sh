#!/bin/bash

# Usage:
#   ./openmp_run_schedule.sh [path_to_binary] [num_runs_per_config]
#
# Example:
#   ./openmp_run_schedule.sh /mnt/POD/MCP_GD/JetClusteringAlgorithm/bin/openmp_version 5
#

set -euo pipefail

BINARY="${1:-./bin/openmp_version}"
RUNS="${2:-1}"                     
OUTPUT_CSV="/mnt/POD/MCP_GD/JetClusteringAlgorithm/benchmarks/benchmark_results_openmp_schedules_threads.csv"

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

echo "Binary:          $BINARY"
echo "Runs per config: $RUNS"
echo "CSV output:      $OUTPUT_CSV"
echo ""

# CSV header 
echo "schedule,chunk,threads,run,time_sec" > "$OUTPUT_CSV"

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

            # capture stdout, extract the execution time line
            output=$(OMP_SCHEDULE="$omp_schedule" OMP_NUM_THREADS="$threads" "$BINARY" 2>&1) || {
                echo "  -> run failed, skipping" >&2
                continue
            }

            time_sec=$(echo "$output" | grep -oP 'Execution time \(total.*?\):\s*\K[0-9]+\.[0-9]+')

            if [[ -z "$time_sec" ]]; then
                echo "  -> could not parse execution time, skipping" >&2
                continue
            fi

            echo "${sched_name},${chunk_label},${threads},${run},${time_sec},${h5_out}" >> "$OUTPUT_CSV"

        done
    done
done

echo ""
echo "Done. Raw per-run results saved to $OUTPUT_CSV"