#!/bin/bash

#SBATCH --job-name=MRD
#SBATCH --partition=cpu-single
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --export=NONE
#SBATCH --array=0-27             ####len of PARAMS array
#SBATCH --out=slurm_files/slurm_MRD_%A_%a.out
#SBATCH --err=slurm_files/slurm_MRD_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --------------------------------------------------
# Parameter grid
# One entry = one Slurm array task
# Format: alpha|label|redshift
# --------------------------------------------------
alphas=("0.01" "0.1")
labels=(
        "G24_K18-3bb_0.0-IG25-agnostic-tau_x_1."
        "G24_K18-3bb_0.0-IG25-all_prograde-tau_x_1."
        "G24_K18-3bb_0.0-IG25-all_retrograde-tau_x_1."
        "G24_K18-3bb_0.1-IG25-agnostic-tau_x_1."
        "G24_K18-3bb_0.0-IG20-agnostic-tau_x_1."
        "G24_K18-3bb_0.0-Calcino23-agnostic-tau_x_1."
        "B16_K18-3bb_0.0-IG25-agnostic-tau_x_1."
        )
redshift_models=("EL" "SE")

PARAMS=()

for alpha in "${alphas[@]}"; do
    for label in "${labels[@]}"; do
        for redshift in "${redshift_models[@]}"; do
            PARAMS+=("${alpha}|${label}|${redshift}")
        done
    done
done


# --------------------------------------------------
# Batches
# --------------------------------------------------
ENTRY="${PARAMS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r ALPHA LABEL REDSHIFT <<< "$ENTRY"

echo "=========================================="
echo "Task ID   : ${SLURM_ARRAY_TASK_ID}"
echo "Alpha     : ${ALPHA}"
echo "Label     : ${LABEL}"
echo "Redshift  : ${REDSHIFT}"
echo "Host      : $(hostname)"
echo "Start     : $(date)"
echo "=========================================="



# --------------------------------------------------
# Paths
# --------------------------------------------------
RUNS_BASE="/gpfs/bwfor/work/ws/hd_tn184-AGN_fastcluster_grid/RUNS_chi_zero/SG/"
PM_DIR="../input/population_models"
NAGN_DIR="../input/population_models/nAGN_models"
PROTO_DIR="../outputs/protoMRD"
MRD_DIR="../outputs/MRD_results"

PM_FILE="${PM_DIR}/pM_given_z_${REDSHIFT}.npz"
PFEDD_FILE="${PM_DIR}/pfEdd_given_Mz_${REDSHIFT}_shen2020_bridge.npz"


# --------------------------------------------------
# Build protoMRD for this (alpha, label, redshift)
# --------------------------------------------------
cd AGNRates/

python3 wrapper_build_proto_MRD.py \
    --runs-base "${RUNS_BASE}" \
    --alpha "${ALPHA}" \
    --label "${LABEL}" \
    --redshift-model "${REDSHIFT}" \
    --pm-file "${PM_FILE}" \
    --pfedd-file "${PFEDD_FILE}" \
    --zform-max 15


# --------------------------------------------------
# Compute final MRD for both abundance models
# --------------------------------------------------

for ABUNDANCE in LAM HAM; do
    echo "Running MRD_computation for ${REDSHIFT}_${ABUNDANCE}"

    python3 MRD_computation.py \
        --proto-dir "${PROTO_DIR}" \
        --alpha "${ALPHA}" \
        --label "${LABEL}" \
        --redshift-model "${REDSHIFT}" \
        --AGN-abundance-model "${ABUNDANCE}" \
        --nagn-dir "${NAGN_DIR}" \
        --outdir "${MRD_DIR}"
done

echo "Finished at $(date)"