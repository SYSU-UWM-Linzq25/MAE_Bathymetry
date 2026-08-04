#!/usr/bin/env bash
#SBATCH -J H055_AGU_export
#SBATCH -p HydroIntel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
PROJECT_ROOT=${PROJECT_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
SCRIPT_DIR=${SCRIPT_DIR:-$PROJECT_ROOT/script}
RESULTS_ROOT=${RESULTS_ROOT:-$PROJECT_ROOT/results}

SCRIPT=${SCRIPT:-$SCRIPT_DIR/H054_export_AGU_selected_reach_data.py}
FIGURE_SCRIPT=${FIGURE_SCRIPT:-$SCRIPT_DIR/H052_make_AGU_relaxed_mask_representative_figure.py}
HELPER_SCRIPT=${HELPER_SCRIPT:-$SCRIPT_DIR/H052_AGU_geospatial_utils.py}

SELECTED_CSV=${SELECTED_CSV:-$RESULTS_ROOT/H052_AGU_RelaxedMask_RepresentativeFigure/H052_selected_representative_reaches.csv}
PREDICTION_ROOT=${PREDICTION_ROOT:-$RESULTS_ROOT/FullRiver_Predictions_F044_MeterOnly_D001cAnyVisiblePatch}
ERROR_ROOT=${ERROR_ROOT:-$RESULTS_ROOT/FullRiver_GT_Error_F046_MeterOnly_D001cAnyVisiblePatch}
TILE_BASE=${TILE_BASE:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_FullRiver_E001c_AnyVisiblePatch}
OUT_DIR=${OUT_DIR:-$RESULTS_ROOT/H054_AGU_SelectedReach_DataBundle}
OVERWRITE=${OVERWRITE:-1}

CONDA_SH=${CONDA_SH:-/home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-$ROOT/conda_envs/mae_zequn}
PYTHON_BIN=${PYTHON_BIN:-$CONDA_ENV/bin/python}

for path in \
  "$SCRIPT" \
  "$FIGURE_SCRIPT" \
  "$HELPER_SCRIPT" \
  "$SELECTED_CSV" \
  "$PREDICTION_ROOT" \
  "$ERROR_ROOT" \
  "$TILE_BASE" \
  "$CONDA_SH" \
  "$PYTHON_BIN"; do
  [[ -e "$path" ]] || {
    echo "[ERROR] Missing required input: $path" >&2
    exit 2
  }
done

module purge || true
source "$CONDA_SH"
conda activate "$CONDA_ENV"

ARGS=(
  --figure_script "$FIGURE_SCRIPT"
  --helper_script "$HELPER_SCRIPT"
  --selected_csv "$SELECTED_CSV"
  --prediction_root "$PREDICTION_ROOT"
  --error_root "$ERROR_ROOT"
  --tile_base "$TILE_BASE"
  --output_dir "$OUT_DIR"
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"$PYTHON_BIN" -u "$SCRIPT" "${ARGS[@]}"

echo
echo "Upload this file:"
echo "$RESULTS_ROOT/H054_AGU_SelectedReach_DataBundle.zip"
