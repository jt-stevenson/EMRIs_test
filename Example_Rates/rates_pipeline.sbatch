#!/bin/bash

#SBATCH --job-name=arise_rates
#SBATCH --partition=cpu-single
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH --output=slurm/logs/rates_%j.out
#SBATCH --error=slurm/logs/rates_%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Submit this job from the ARISE repository root with the arise environment active:
#   conda activate arise
#   sbatch slurm/rates_pipeline.sbatch

ARISE_ROOT="${SLURM_SUBMIT_DIR:?Submit from the ARISE repository root}"
PATH_CONFIG="${ARISE_ROOT}/configs/paths.env"

if [[ ! -r "${PATH_CONFIG}" ]]; then
    echo "ERROR: path configuration not found: ${PATH_CONFIG}" >&2
    exit 1
fi

# This file is trusted local shell configuration.
source "${PATH_CONFIG}"

: "${POPULATION_DIR:?POPULATION_DIR is not defined in paths.env}"
: "${OUTPUT_BASE:?OUTPUT_BASE is not defined in paths.env}"

NAGN_DIR="${POPULATION_DIR}"
PROTO_DIR="${OUTPUT_BASE}/protoMRD"
MRD_DIR="${OUTPUT_BASE}/MRD_results"

# Compact yield files produced by run_yields.py. The literal {Z} placeholder
# is expanded by ARISE, not by Bash.
YIELDS_BASE="${OUTPUT_BASE}/yields_EMRI/Z{Z}"

ALPHA="0.1"
LABEL="label"
YIELD_LABEL="EMRI"
REDSHIFT="EL"
METALLICITY="single_Z"
METALLICITY_VALUE="0.02"

PM_FILE="${POPULATION_DIR}/pM_given_z_${REDSHIFT}.npz"
PFEDD_FILE="${POPULATION_DIR}/pfEdd_given_Mz_${REDSHIFT}_shen2020_bridge.npz"

cd "${ARISE_ROOT}"

python -c "import arise" || {
    echo "ERROR: ARISE is not installed in the active Python environment." >&2
    exit 1
}

for required_path in "${PM_FILE}" "${PFEDD_FILE}" "${NAGN_DIR}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: required input not found: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${PROTO_DIR}" "${MRD_DIR}"

echo "=========================================="
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Alpha         : ${ALPHA}"
echo "Label         : ${LABEL}"
echo "Redshift model: ${REDSHIFT}"
echo "Metallicity   : ${METALLICITY} (${METALLICITY_VALUE})"
echo "Python        : $(command -v python)"
echo "Host          : $(hostname)"
echo "Start         : $(date)"
echo "=========================================="


python -m arise.rates.wrapper_build_proto_MRD \
    --yields-base "${YIELDS_BASE}" \
    --alpha "${ALPHA}" \
    --label "${LABEL}" \
    --yield-labels "${YIELD_LABEL}" \
    --redshift-model "${REDSHIFT}" \
    --metallicity-model "${METALLICITY}" \
    --metallicity-value "${METALLICITY_VALUE}" \
    --pm-file "${PM_FILE}" \
    --pfedd-file "${PFEDD_FILE}" \
    --zform-min 0.0 \
    --zform-max 15.0 \
    --zform-step 0.01 \
    --outdir "${PROTO_DIR}"


for ABUNDANCE in LAM HAM; do
    echo "Computing final MRD for ${REDSHIFT}_${ABUNDANCE}"

    python -m arise.rates.MRD_computation \
        --proto-dir "${PROTO_DIR}" \
        --alpha "${ALPHA}" \
        --label "${LABEL}" \
        --redshift-model "${REDSHIFT}" \
        --metallicity-model "${METALLICITY}" \
        --AGN-abundance-model "${ABUNDANCE}" \
        --nagn-dir "${NAGN_DIR}" \
        --outdir "${MRD_DIR}"
done

echo "Finished at $(date)"
