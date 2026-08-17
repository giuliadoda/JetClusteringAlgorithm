#!/usr/bin/env bash
#
# changing compiler optimization level and number of events
# 5 runs per configuration
# saving execution times in a CSV file
#


set -euo pipefail

# configuration

PROJECT_ROOT="/mnt/POD/MCP_GD/JetClusteringAlgorithm"

CONSTANTS_H="${PROJECT_ROOT}/include/constants.h"

OPT_LEVELS=("-O0" "-O1" "-O2" "-O3")

N_EVENTS_LIST=(100 1000 10000 50000 100000)

N_RUNS=5

RESULTS_DIR="${PROJECT_ROOT}/benchmarks"
OUTFILE="${RESULTS_DIR}/serial_timings.csv"
PER_EVENT_OUTFILE="${RESULTS_DIR}/serial_per_event_timings.csv"

SERIAL_BIN="${PROJECT_ROOT}/bin/serial_version"


# setup

mkdir -p "${RESULTS_DIR}"

if [ ! -f "${CONSTANTS_H}" ]; then
    echo "${CONSTANTS_H} not found." >&2
    exit 1
fi

# constants.h backup
cp "${CONSTANTS_H}" "${CONSTANTS_H}.bak"
trap 'mv "${CONSTANTS_H}.bak" "${CONSTANTS_H}"' EXIT

echo "opt_level,n_events,run,time_sec" > "${OUTFILE}"
echo "opt_level,n_events,run,event_id,n_particles,time_sec" > "${PER_EVENT_OUTFILE}"   


# benchmarking

for opt in "${OPT_LEVELS[@]}"; do
    for n in "${N_EVENTS_LIST[@]}"; do

        echo " ------- Compiling OPT=${opt}  N_EVENTS=${n} -------"

        # change N_EVENTS in constants.h
        sed -i -E "s/^#define[[:space:]]+N_EVENTS[[:space:]]+.*/#define N_EVENTS ${n}/" "${CONSTANTS_H}"

        # compiling
        make -C "${PROJECT_ROOT}" clean > /dev/null
        make -C "${PROJECT_ROOT}" serial OPT="${opt}" > /dev/null

        if [ ! -x "${SERIAL_BIN}" ]; then
            echo "${SERIAL_BIN} not found." >&2
            exit 1
        fi

        for run in $(seq 1 "${N_RUNS}"); do
            echo "    run ${run}/${N_RUNS} ..."

            output=$("${SERIAL_BIN}")

            # extracting execution time
            t=$(echo "${output}" | grep -oP 'Execution time.*?:\s*\K[0-9]+\.[0-9]+(?=\s*\(sec\))')

            if [ -z "${t}" ]; then
                echo "Execution time not found for OPT=${opt} N_EVENTS=${n} run=${run}" >&2
                echo "Output:"
                echo "${output}"
                continue
            fi

            echo "${opt},${n},${run},${t}" >> "${OUTFILE}"

            # event execution times and number of particles
            # save only for N_EVENTS=100000
            if [[ "${n}" -eq 100000 ]]; then
                while IFS=',' read -r tag ev_id ev_part ev_time; do
                    [[ "${tag}" == "EVENT" ]] || continue
                    echo "${opt},${n},${run},${ev_id},${ev_part},${ev_time}" >> "${PER_EVENT_OUTFILE}"
                done <<< "$(grep '^EVENT,' <<< "${output}")"
            fi

        done
    done
done

echo ""
echo "Done. Results saved in: ${OUTFILE}"