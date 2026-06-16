#!/bin/bash

set -eo pipefail

M=${M:-20000}
N=${N:-5000}
K=${K:-20}
SEED=${SEED:-42}
TOL=${TOL:-1e-4}
MAX_ITER=${MAX_ITER:-20}
CSV_DIR=${CSV_DIR:-results/jobs}

mkdir -p outputs results "${CSV_DIR}"

submit_run() {
    local pr=$1
    local pc=$2
    local threads=$3
    local nodes=$4
    local ntasks_per_socket=$5
    local tag=$6

    local ntasks=$((pr * pc))
    local ntasks_per_node=$(((ntasks + nodes - 1) / nodes))
    local job_name="nmf_p${pr}x${pc}_t${threads}_${tag}"
    local csv="${CSV_DIR}/${job_name}.csv"

    sbatch \
        --job-name="${job_name}" \
        --nodes="${nodes}" \
        --ntasks="${ntasks}" \
        --ntasks-per-node="${ntasks_per_node}" \
        --ntasks-per-socket="${ntasks_per_socket}" \
        --cpus-per-task="${threads}" \
        --export=ALL,PR="${pr}",PC="${pc}",M="${M}",N="${N}",K="${K}",SEED="${SEED}",TOL="${TOL}",MAX_ITER="${MAX_ITER}",CSV="${csv}" \
        slurm/run_nmf.sh
}

submit_grid() {
    local pr=$1
    local pc=$2

    submit_run "${pr}" "${pc}" 1 1 1 "official"
    submit_run "${pr}" "${pc}" 2 1 1 "official"

    if [ $((pr * pc * 4)) -le 8 ]; then
        submit_run "${pr}" "${pc}" 4 1 1 "official"
    else
        submit_run "${pr}" "${pc}" 4 2 1 "official"
    fi
}

echo "Submitting official NMF experiment matrix"
echo "CSV directory: ${CSV_DIR}"

submit_grid 1 1
submit_grid 2 1
submit_grid 1 2
submit_grid 2 2
submit_grid 4 1
submit_grid 1 4

echo "Submitting NUMA affinity comparison"
submit_run 2 1 2 1 1 "numa_socket_split"
submit_run 2 1 2 1 2 "numa_same_socket"