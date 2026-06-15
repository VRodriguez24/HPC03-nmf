#!/bin/bash
#SBATCH --job-name=nmf
#SBATCH --partition=hpc-iic3533
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err

set -euo pipefail

PR=${PR:-1}
PC=${PC:-1}
M=${M:-20000}
N=${N:-5000}
K=${K:-20}
SEED=${SEED:-42}
TOL=${TOL:-1e-4}
MAX_ITER=${MAX_ITER:-20}
CSV=${CSV:-results/nmf_results.csv}

EXPECTED_TASKS=$((PR * PC))

if [ "${SLURM_NTASKS}" -ne "${EXPECTED_TASKS}" ]; then
    echo "ERROR: SLURM_NTASKS=${SLURM_NTASKS}, pero PR*PC=${EXPECTED_TASKS}" >&2
    exit 1
fi

mkdir -p outputs results

source ~/miniconda3/bin/activate iic3533

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "============================================================"
echo "NMF distributed run"
echo "============================================================"
echo "job_id           : ${SLURM_JOB_ID}"
echo "nodes            : ${SLURM_JOB_NUM_NODES}"
echo "ntasks           : ${SLURM_NTASKS}"
echo "cpus_per_task    : ${SLURM_CPUS_PER_TASK}"
echo "omp_threads      : ${OMP_NUM_THREADS}"
echo "grid             : ${PR} x ${PC}"
echo "matrix           : ${M} x ${N}"
echo "rank_k           : ${K}"
echo "max_iter         : ${MAX_ITER}"
echo "csv              : ${CSV}"
echo "============================================================"

mpirun -n "${SLURM_NTASKS}" python src/nmf_distributed.py \
    --pr "${PR}" \
    --pc "${PC}" \
    --m "${M}" \
    --n "${N}" \
    --k "${K}" \
    --seed "${SEED}" \
    --tol "${TOL}" \
    --max_iter "${MAX_ITER}" \
    --csv "${CSV}"
