#!/bin/bash

#SBATCH --job-name=catalog_plotting
#SBATCH --partition=cpu-single
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --export=NONE
#SBATCH --array=0-23              ####len of PARAMS array
#SBATCH --out=slurm_files/slurm_catalog_%A_%a.out
#SBATCH --err=slurm_files/slurm_catalog_%A_%a.err

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
labels=("G24_K18-3bb_0.0-IG25-agnostic-tau_x_1."
        "G24_K18-3bb_0.0-IG25-all_prograde-tau_x_1."
        "G24_K18-3bb_0.0-IG25-all_retrograde-tau_x_1.")
redshift_models=("EL" "SE")
abundance_models=("LAM" "HAM")

PARAMS=()

for alpha in "${alphas[@]}"; do
    for label in "${labels[@]}"; do
        for redshift in "${redshift_models[@]}"; do
            for abundance in "${abundance_models[@]}"; do
                PARAMS+=("${alpha}|${label}|${redshift}|${abundance}")
            done
        done
    done
done


# --------------------------------------------------
# Batches
# --------------------------------------------------
ENTRY="${PARAMS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r ALPHA LABEL REDSHIFT ABUNDANCE <<< "$ENTRY"

echo "=========================================="
echo "Task ID   : ${SLURM_ARRAY_TASK_ID}"
echo "Alpha     : ${ALPHA}"
echo "Label     : ${LABEL}"
echo "Redshift  : ${REDSHIFT}"
echo "Abundance : ${ABUNDANCE}"
echo "Host      : $(hostname)"
echo "Start     : $(date)"
echo "=========================================="



python ../plotting_scripts/catalog_plotting.py \
    --input "../outputs/catalogs_by_zmerg/redshift_model_${REDSHIFT}/agn_abundance_${ABUNDANCE}/alpha_${ALPHA}/z_0.0_10.0/BBH_mergers_${LABEL}txt" \
    --output "../plots/intrinsic_catalog/pM1_joyplot_zmerg_0.0_10.0_alpha_${ALPHA}_${REDSHIFT}_${ABUNDANCE}_${LABEL}.png" --log-m1 --z-edges 0 1 2 3 4 5 6 7 8 \


echo "Finished at $(date)"