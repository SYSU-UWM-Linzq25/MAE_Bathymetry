#!/usr/bin/env bash
#SBATCH -J H051_AGU_strict
#SBATCH -p HydroIntel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Strict-mask AGU figure.
# Everything project-specific lives under Downstream_Task_Bathy:
#   script  -> $PROJECT_ROOT/script
#   results -> $PROJECT_ROOT/Results
#
# No relaxed-project result path is used.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
PROJECT_ROOT=${PROJECT_ROOT:-$ROOT/Downstream_Task_Bathy}
SCRIPT_DIR=${SCRIPT_DIR:-$PROJECT_ROOT/script}
RESULTS_ROOT=${RESULTS_ROOT:-$PROJECT_ROOT/Results}

SCRIPT=${SCRIPT:-$SCRIPT_DIR/H050_make_AGU_strict_mask_representative_figure.py}
H046_SCRIPT=${H046_SCRIPT:-$SCRIPT_DIR/H050_AGU_geospatial_utils.py}

PREDICTION_ROOT=${PREDICTION_ROOT:-$RESULTS_ROOT/FullRiver_Predictions_F060_TileAvgVRT_D003MeterMAE_BaselineEval_D001NoDataSafe}
ERROR_ROOT=${ERROR_ROOT:-$RESULTS_ROOT/FullRiver_GT_Error_F062_UniquePixel_D003MeterMAE_BaselineEval_D001NoDataSafe}
TILE_BASE=${TILE_BASE:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_FullRiver_E001}

OUT_DIR=${OUT_DIR:-$RESULTS_ROOT/H050_AGU_StrictMask_RepresentativeFigure}
MANUAL_SELECTION_CSV=${MANUAL_SELECTION_CSV:-}
CANDIDATE_METRICS_CSV=${CANDIDATE_METRICS_CSV:-}

SEGMENT_SIZE=${SEGMENT_SIZE:-10}
SEGMENT_STRIDE=${SEGMENT_STRIDE:-10}
MIN_SEGMENT_POINTS=${MIN_SEGMENT_POINTS:-5}
REQUIRED_SAMPLING_POINTS=${REQUIRED_SAMPLING_POINTS:-10}
MIN_FINAL_PIXELS=${MIN_FINAL_PIXELS:-1000}
CROP_PADDING=${CROP_PADDING:-24}
ABS_ERROR_MAX_M=${ABS_ERROR_MAX_M:-2.0}
DPI=${DPI:-400}
OVERWRITE=${OVERWRITE:-0}

CONDA_SH=${CONDA_SH:-/home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-$ROOT/conda_envs/mae_zequn}
PYTHON_BIN=${PYTHON_BIN:-$CONDA_ENV/bin/python}

for path in \
  "$SCRIPT" \
  "$H046_SCRIPT" \
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

if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$OUT_DIR"
  else
    echo "[ERROR] Output exists: $OUT_DIR" >&2
    echo "Set OVERWRITE=1 to rebuild." >&2
    exit 3
  fi
fi
mkdir -p "$OUT_DIR/logs"

JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$OUT_DIR/logs/H051_AGU_strict_${JOB_ID}.out" \
     2>"$OUT_DIR/logs/H051_AGU_strict_${JOB_ID}.err"

module purge || true
source "$CONDA_SH"
conda activate "$CONDA_ENV"

EXTRA_ARGS=()
if [[ -n "$MANUAL_SELECTION_CSV" ]]; then
  [[ -f "$MANUAL_SELECTION_CSV" ]] || {
    echo "[ERROR] Missing MANUAL_SELECTION_CSV: $MANUAL_SELECTION_CSV" >&2
    exit 4
  }
  EXTRA_ARGS+=(--manual_selection_csv "$MANUAL_SELECTION_CSV")
fi
if [[ -n "$CANDIDATE_METRICS_CSV" ]]; then
  [[ -f "$CANDIDATE_METRICS_CSV" ]] || {
    echo "[ERROR] Missing CANDIDATE_METRICS_CSV: $CANDIDATE_METRICS_CSV" >&2
    exit 5
  }
  EXTRA_ARGS+=(--candidate_metrics_csv "$CANDIDATE_METRICS_CSV")
fi

echo "============================================================"
echo "H051 AGU strict-mask representative figure"
date
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "PREDICTION_ROOT=$PREDICTION_ROOT"
echo "ERROR_ROOT=$ERROR_ROOT"
echo "TILE_BASE=$TILE_BASE"
echo "OUT_DIR=$OUT_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "============================================================"

"$PYTHON_BIN" -u "$SCRIPT" \
  --h046_script "$H046_SCRIPT" \
  --prediction_root "$PREDICTION_ROOT" \
  --error_root "$ERROR_ROOT" \
  --tile_base "$TILE_BASE" \
  --output_dir "$OUT_DIR" \
  --segment_size "$SEGMENT_SIZE" \
  --segment_stride "$SEGMENT_STRIDE" \
  --min_segment_points "$MIN_SEGMENT_POINTS" \
  --required_sampling_points "$REQUIRED_SAMPLING_POINTS" \
  --min_final_pixels "$MIN_FINAL_PIXELS" \
  --crop_padding "$CROP_PADDING" \
  --absolute_error_max_m "$ABS_ERROR_MAX_M" \
  --dpi "$DPI" \
  "${EXTRA_ARGS[@]}"

echo "============================================================"
echo "DONE"
echo "$OUT_DIR/AGU_strict_mask_representative_reaches.png"
echo "$OUT_DIR/AGU_strict_mask_representative_reaches.pdf"
echo "$OUT_DIR/H050_selected_representative_reaches.csv"
echo "$OUT_DIR/H050_all_candidate_reach_metrics.csv"
date
echo "============================================================"
