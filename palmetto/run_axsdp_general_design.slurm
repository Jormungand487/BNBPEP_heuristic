#!/bin/bash
# Supported AXSDP_DESIGN_MODE values:
#   single_tau
#   multi_tau
#   abcd
#   full_coeff
#
# Example:
#   sbatch --export=ALL,AXSDP_DESIGN_MODE=multi_tau,N=3,QCQP_TIME_LIMIT=18000 \
#     palmetto/run_axsdp_general_design.slurm
#
# Useful optional environment variables:
#   AXSDP_RESUME=1
#   AXSDP_ALLOW_WEIGHT_ON_X0=0
#   AXSDP_ENFORCE_DEFAULT_STABILITY=1
#   AXSDP_GUROBI_THREADS=4
#   AXSDP_LOCAL_MAX_ITER=3000

#SBATCH --job-name=axsdp-general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT_90

set -euo pipefail

conda deactivate 2>/dev/null || true
module purge

if [[ -n "${GUROBI_MODULE:-}" ]]; then
    module load "${GUROBI_MODULE}"
fi

if [[ -n "${EXTRA_MODULES:-}" ]]; then
    OLD_IFS="$IFS"
    IFS=',' read -r -a extra_modules <<< "${EXTRA_MODULES}"
    IFS="$OLD_IFS"
    module load "${extra_modules[@]}"
fi

: "${REPO_DIR:?Set REPO_DIR to the repository path on Palmetto.}"
: "${JULIA_BIN:?Set JULIA_BIN to your Julia binary on Palmetto.}"

export AXSDP_DESIGN_MODE="${AXSDP_DESIGN_MODE:-multi_tau}"
export QCQP_TIME_LIMIT="${QCQP_TIME_LIMIT:-18000}"
export N="${N:-3}"
export RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/outputs/palmetto_general_runs}"

bash "${REPO_DIR}/scripts/run_axsdp_general_design_job.sh" \
    "${AXSDP_DESIGN_MODE}" \
    "${N}" \
    "${QCQP_TIME_LIMIT}" \
    "${RUN_ROOT}"
