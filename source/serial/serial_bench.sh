#!/usr/bin/env bash
#
# Benchmark della versione seriale al variare di:
#   - livello di ottimizzazione del compilatore (OPT)
#   - numero di eventi processati (N_EVENTS, definito in utils.h)
#
# Per ogni combinazione esegue N_RUNS run e salva i tempi in un CSV.
#
# USO:
#   ./run_serial_benchmarks.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIGURAZIONE (modifica qui secondo le tue esigenze)
# ---------------------------------------------------------------------------

# Percorso del progetto (dove sta il Makefile)
PROJECT_DIR="."

# Percorso di utils.h che contiene #define N_EVENTS ...
# ADATTA questo percorso se nel tuo progetto è diverso
UTILS_H="${PROJECT_DIR}/include/serial/utils.h"

# Livelli di ottimizzazione da testare
OPT_LEVELS=("-O0" "-O1" "-O2" "-O3")

# Valori di N_EVENTS da testare
N_EVENTS_LIST=(100 500 1000 5000 10000)

# Numero di run ripetuti per ogni combinazione
N_RUNS=5

# Directory e file di output
RESULTS_DIR="${PROJECT_DIR}/results/timing"
OUTFILE="${RESULTS_DIR}/serial_timings.csv"

# Eseguibile serial prodotto dal Makefile
SERIAL_BIN="${PROJECT_DIR}/bin/serial_version"

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

mkdir -p "${RESULTS_DIR}"

if [ ! -f "${UTILS_H}" ]; then
    echo "ERRORE: non trovo ${UTILS_H}. Modifica la variabile UTILS_H nello script." >&2
    exit 1
fi

# backup di utils.h, per ripristinarlo a fine script
cp "${UTILS_H}" "${UTILS_H}.bak"
trap 'mv "${UTILS_H}.bak" "${UTILS_H}"' EXIT

echo "opt_level,n_events,run,time_sec" > "${OUTFILE}"

# ---------------------------------------------------------------------------
# LOOP SU OTTIMIZZAZIONE x N_EVENTS x RUN
# ---------------------------------------------------------------------------

for opt in "${OPT_LEVELS[@]}"; do
    for n in "${N_EVENTS_LIST[@]}"; do

        echo ">>> Compilazione: OPT=${opt}  N_EVENTS=${n}"

        # aggiorna N_EVENTS in utils.h
        sed -i -E "s/^#define[[:space:]]+N_EVENTS[[:space:]]+.*/#define N_EVENTS ${n}/" "${UTILS_H}"

        # ricompila da zero per evitare oggetti stantii tra una config e l'altra
        make -C "${PROJECT_DIR}" clean > /dev/null
        make -C "${PROJECT_DIR}" serial OPT="${opt}" > /dev/null

        if [ ! -x "${SERIAL_BIN}" ]; then
            echo "ERRORE: binario ${SERIAL_BIN} non trovato dopo la compilazione." >&2
            exit 1
        fi

        for run in $(seq 1 "${N_RUNS}"); do
            echo "    run ${run}/${N_RUNS} ..."

            output=$("${SERIAL_BIN}")

            # estrae il numero di secondi dalla riga:
            # "Execution time (total, N events): X.XXXXXX (sec)"
            t=$(echo "${output}" | grep -oP 'Execution time.*?:\s*\K[0-9]+\.[0-9]+(?=\s*\(sec\))')

            if [ -z "${t}" ]; then
                echo "ATTENZIONE: non sono riuscito a estrarre il tempo per OPT=${opt} N_EVENTS=${n} run=${run}" >&2
                echo "Output del programma:"
                echo "${output}"
                continue
            fi

            echo "${opt},${n},${run},${t}" >> "${OUTFILE}"
        done
    done
done

echo ""
echo "Fatto. Risultati salvati in: ${OUTFILE}"