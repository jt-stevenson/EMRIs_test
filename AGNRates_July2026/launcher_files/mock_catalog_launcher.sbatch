#!/bin/bash

#SBATCH --job-name=mock_catalog
#SBATCH --partition=cpu-single
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --export=NONE
#SBATCH --array=0-23              ####len of PARAMS array
#SBATCH --out=slurm_files/slurm_mock_catalog_%A_%a.out
#SBATCH --err=slurm_files/slurm_mock_catalog_%A_%a.err

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
labels=("G24_K18-3bb_0.0-IG25-agnostic-tau_x_1"
        "G24_K18-3bb_0.0-IG25-all_prograde-tau_x_1"
        "G24_K18-3bb_0.0-IG25-all_retrograde-tau_x_1")
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




cd AGNRates/
python3 build_mock_catalog.py --catalog "../outputs/catalogs_by_zmerg/redshift_model_"${REDSHIFT}"/agn_abundance_"${ABUNDANCE}"/alpha_"${ALPHA}"/z_0.0_10.0/BBH_mergers_"${LABEL}".txt" \
        --mrd "../outputs/MRD_results/alpha_"${ALPHA}"/"${LABEL}"./"${REDSHIFT}"_"${ABUNDANCE}"/MRD_vs_z.txt" --z-column z_merg --mrd-z-column z --mrd-rate-column R_of_z \
        --duration-yr 10 --z-min 0.0 --z-max 10.0 --outdir "../outputs/mock_ET_catalogs/alpha_"${ALPHA}"_"${REDSHIFT}"_"${ABUNDANCE}"/"${LABEL}"/" \
        --prefix "mock_catalog_ET_10yr_Rz"

python3 ../plotting_scripts/mock_catalog_plotting.py --input "../outputs/mock_ET_catalogs/alpha_"${ALPHA}"_"${REDSHIFT}"_"${ABUNDANCE}"/"${LABEL}"/mock_catalog_ET_10yr_Rz.csv" \
        --mass-quantity m1 --output-dir "../outputs/mock_ET_catalogs/alpha_"${ALPHA}"_"${REDSHIFT}"_"${ABUNDANCE}"/"${LABEL}"/figures"

cp "../outputs/mock_ET_catalogs/alpha_"${ALPHA}"_"${REDSHIFT}"_"${ABUNDANCE}"/"${LABEL}"/figures/mock_catalog_N_z_m1_raw_counts.png" \
   "../plots/mock_catalogs/mock_ET_10yr_alpha_${ALPHA}_${REDSHIFT}_${ABUNDANCE}_${LABEL}.png"