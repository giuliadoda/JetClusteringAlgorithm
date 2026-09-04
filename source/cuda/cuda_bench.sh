#!/usr/bin/env bash
#
# changing number of threads per block and number of events (at compilation time)
# 5 run per configuration
# saving times in CSV
#


set -euo pipefail

# configuration

THR_BLOCK_VALUES=(16 32 64 128 256 512)

N_EVENTS_VALUES=(100 1000 10000 50000 100000)

RUNS=5

PROJECT_ROOT="/mnt/POD/MCP_GD/JetClusteringAlgorithm"

BIN_PATH="${PROJECT_ROOT}/bin/cuda_version"

RAW_CSV="${PROJECT_ROOT}/benchmarks/results_cuda_raw_2.csv"
SUMMARY_CSV="${PROJECT_ROOT}/benchmarks/results_cuda_summary_2.csv"


# setup

cd "${PROJECT_ROOT}"

echo "thr_block,n_events,run,wall_time_sec,h2d_ms,kernel_ms,d2h_ms,avg_event_ms" > "${RAW_CSV}"

total_combos=$(( ${#THR_BLOCK_VALUES[@]} * ${#N_EVENTS_VALUES[@]} ))
combo_idx=0


# benchmarks runs

for THR in "${THR_BLOCK_VALUES[@]}"; do
    for NEV in "${N_EVENTS_VALUES[@]}"; do

        combo_idx=$((combo_idx + 1))
        echo "=== [${combo_idx}/${total_combos}] THR_BLOCK=${THR}  N_EVENTS=${NEV} ==="

        # cleaning previous config
        make clean > /dev/null

        if ! make cuda EXTRA_DEFS="-DTHR_BLOCK=${THR} -DN_EVENTS=${NEV}" > build.log 2>&1; then
            echo "Failed compilation: THR_BLOCK=${THR} N_EVENTS=${NEV}"
            continue
        fi

        if [[ ! -x "${BIN_PATH}" ]]; then
            echo "Bin not found, skipping"
            continue
        fi

        for ((run=1; run<=RUNS; run++)); do

            echo "  -> run ${run}/${RUNS}"

            OUT=$("${BIN_PATH}" 2>&1) || {
                echo "Run failed"
                echo "${OUT}"
                continue
            }

            wall_time=$(echo "${OUT}" | grep -oP 'total wall-clock, \d+ events\): \K[0-9.]+')
            h2d=$(echo "${OUT}"       | grep -oP 'H2D copy time: \K[0-9.]+')
            kernel=$(echo "${OUT}"    | grep -oP 'Kernel time: \K[0-9.]+')
            d2h=$(echo "${OUT}"       | grep -oP 'D2H copy time: \K[0-9.]+')
            avg_ev=$(echo "${OUT}"    | grep -oP 'Average event time: \K[0-9.]+')

            if [[ -z "${wall_time}" || -z "${h2d}" || -z "${kernel}" || -z "${d2h}" || -z "${avg_ev}" ]]; then
                echo "Incomplete output parsing, skipping"
                echo "${OUT}"
                continue
            fi

            echo "${THR},${NEV},${run},${wall_time},${h2d},${kernel},${d2h},${avg_ev}" >> "${RAW_CSV}"

        done

    done
done

echo ""
echo "Raw results saved in: ${RAW_CSV}"

# summary results

awk -F',' '
NR == 1 { next }  # skip header
{
    key = $1","$2
    n[key]++

    wall_sum[key]   += $4; wall_sumsq[key]   += $4*$4
    h2d_sum[key]    += $5; h2d_sumsq[key]    += $5*$5
    kernel_sum[key] += $6; kernel_sumsq[key] += $6*$6
    d2h_sum[key]    += $7; d2h_sumsq[key]    += $7*$7
    avg_sum[key]    += $8; avg_sumsq[key]    += $8*$8

    if (!(key in thr)) { split(key, parts, ","); thr[key]=parts[1]; nev[key]=parts[2] }
}
END {
    print "thr_block,n_events,n_runs,wall_time_sec_mean,wall_time_sec_std,h2d_ms_mean,h2d_ms_std,kernel_ms_mean,kernel_ms_std,d2h_ms_mean,d2h_ms_std,avg_event_ms_mean,avg_event_ms_std"
    for (key in n) {
        cnt = n[key]

        wall_mean = wall_sum[key]/cnt
        wall_std  = sqrt(wall_sumsq[key]/cnt - wall_mean*wall_mean)

        h2d_mean = h2d_sum[key]/cnt
        h2d_std  = sqrt(h2d_sumsq[key]/cnt - h2d_mean*h2d_mean)

        kernel_mean = kernel_sum[key]/cnt
        kernel_std  = sqrt(kernel_sumsq[key]/cnt - kernel_mean*kernel_mean)

        d2h_mean = d2h_sum[key]/cnt
        d2h_std  = sqrt(d2h_sumsq[key]/cnt - d2h_mean*d2h_mean)

        avg_mean = avg_sum[key]/cnt
        avg_std  = sqrt(avg_sumsq[key]/cnt - avg_mean*avg_mean)

        printf "%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", \
            thr[key], nev[key], cnt, \
            wall_mean, wall_std, h2d_mean, h2d_std, \
            kernel_mean, kernel_std, d2h_mean, d2h_std, \
            avg_mean, avg_std
    }
}
' "${RAW_CSV}" | sort -t',' -k1,1n -k2,2n > "${SUMMARY_CSV}"

echo "Summary results saved in: ${SUMMARY_CSV}"
echo ""
echo "All done :))"