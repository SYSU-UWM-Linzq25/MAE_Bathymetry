#!/usr/bin/env bash
#SBATCH -J H045_fullriver
#SBATCH -p HydroIntel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Analysis only. Reads existing full-river prediction manifests.
# It does not call F044/F045/F046/F047 and does not rerun inference.
#
# Main footprint:
#   Core_Loss_Mask_Pixel
#   AND valid GT
#   AND valid predictions from all four configurations
#
# Every overlap-averaged geospatial pixel is counted once.
#
# Configurations:
#   strict normalized
#   strict meter
#   relaxed normalized
#   relaxed meter

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
STRICT_RESULTS=${STRICT_RESULTS:-$ROOT/Downstream_Task_Bathy/Results}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
RELAX_RESULTS=${RELAX_RESULTS:-$RELAX_ROOT/results}
SCRIPT=${SCRIPT:-$RELAX_ROOT/script/H045_compare_fullriver_common_loss_pixels.py}
RESOLVER=${RESOLVER:-$RELAX_ROOT/script/H044_resolve_relaxed_normalized_prediction_root.py}

STRICT_NORMALIZED_PRED_ROOT=${STRICT_NORMALIZED_PRED_ROOT:-$STRICT_RESULTS/FullRiver_Predictions_F010_TileAvgVRT_D001NoDataSafe}
STRICT_METER_PRED_ROOT=${STRICT_METER_PRED_ROOT:-$STRICT_RESULTS/FullRiver_Predictions_F060_TileAvgVRT_D003MeterMAE_BaselineEval_D001NoDataSafe}
RELAX_METER_PRED_ROOT=${RELAX_METER_PRED_ROOT:-$RELAX_RESULTS/FullRiver_Predictions_F044_MeterOnly_D001cAnyVisiblePatch}
RELAX_NORMALIZED_PRED_ROOT=${RELAX_NORMALIZED_PRED_ROOT:-$RELAX_RESULTS/FullRiver_Predictions_F049_NormalizedObjective_D001cAnyVisiblePatch}
INPUT_AUDIT_JSON=${INPUT_AUDIT_JSON:-$RELAX_RESULTS/H045_input_resolution.json}

OUT_DIR=${OUT_DIR:-$RELAX_RESULTS/H045_FullRiver_CommonLossPixel_Analysis}
OVERWRITE=${OVERWRITE:-0}
DPI=${DPI:-220}
DENSITY_MAX_ERROR_M=${DENSITY_MAX_ERROR_M:-2.0}

CONDA_SH=${CONDA_SH:-/home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-$ROOT/conda_envs/mae_zequn}
PYTHON_BIN=${PYTHON_BIN:-$CONDA_ENV/bin/python}

[[ -f "$CONDA_SH" ]] || {
  echo "[ERROR] Missing conda initialization script: $CONDA_SH" >&2
  exit 2
}
[[ -x "$PYTHON_BIN" ]] || {
  echo "[ERROR] Python executable is missing or not executable: $PYTHON_BIN" >&2
  exit 2
}


if [[ ! -f "$RESOLVER" ]]; then
  echo "[ERROR] Missing input resolver: $RESOLVER" >&2
  exit 2
fi

if [[ -z "$RELAX_NORMALIZED_PRED_ROOT" ]]; then
  RELAX_NORMALIZED_PRED_ROOT=$(
    "$PYTHON_BIN" "$RESOLVER" \
      --relax-root "$RELAX_ROOT" \
      --output-json "$INPUT_AUDIT_JSON" \
      --print-root
  )
else
  RELAX_NORMALIZED_PRED_ROOT=$(
    "$PYTHON_BIN" "$RESOLVER" \
      --relax-root "$RELAX_ROOT" \
      --explicit-root "$RELAX_NORMALIZED_PRED_ROOT" \
      --output-json "$INPUT_AUDIT_JSON" \
      --print-root
  )
fi
export RELAX_NORMALIZED_PRED_ROOT

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$LOG_DIR/H045_fullriver_common_loss_${JOB_ID}.out" \
     2>"$LOG_DIR/H045_fullriver_common_loss_${JOB_ID}.err"

module purge || true
source "$CONDA_SH"
conda activate "$CONDA_ENV"

for path in \
  "$SCRIPT" \
  "$RESOLVER" \
  "$STRICT_NORMALIZED_PRED_ROOT" \
  "$STRICT_METER_PRED_ROOT" \
  "$RELAX_NORMALIZED_PRED_ROOT" \
  "$RELAX_METER_PRED_ROOT"; do
  [[ -e "$path" ]] || { echo "[ERROR] Missing: $path" >&2; exit 2; }
done

if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -not -path "$OUT_DIR/logs*" -print -quit | grep -q .; then
  if [[ "$OVERWRITE" == "1" ]]; then
    find "$OUT_DIR" -mindepth 1 -maxdepth 1 ! -name logs -exec rm -rf {} +
  else
    echo "[ERROR] Output exists: $OUT_DIR" >&2
    echo "Set OVERWRITE=1 to rebuild analysis figures." >&2
    exit 3
  fi
fi
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "H045 analysis only: full-river common loss pixels"
date
echo "STRICT_NORMALIZED_PRED_ROOT=$STRICT_NORMALIZED_PRED_ROOT"
echo "STRICT_METER_PRED_ROOT=$STRICT_METER_PRED_ROOT"
echo "RELAX_NORMALIZED_PRED_ROOT=$RELAX_NORMALIZED_PRED_ROOT"
echo "INPUT_AUDIT_JSON=$INPUT_AUDIT_JSON"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "RELAX_METER_PRED_ROOT=$RELAX_METER_PRED_ROOT"
echo "COMMON_FOOTPRINT=Core_Loss_Mask_Pixel AND valid_GT AND all four valid predictions"
echo "UNIQUE_PIXEL_RULE=each overlap-averaged geospatial pixel counted once"
echo "OUT_DIR=$OUT_DIR"
echo "============================================================"

"$PYTHON_BIN" -u "$SCRIPT" \
  --strict_normalized_pred_root "$STRICT_NORMALIZED_PRED_ROOT" \
  --strict_meter_pred_root "$STRICT_METER_PRED_ROOT" \
  --relaxed_normalized_pred_root "$RELAX_NORMALIZED_PRED_ROOT" \
  --relaxed_meter_pred_root "$RELAX_METER_PRED_ROOT" \
  --output_dir "$OUT_DIR" \
  --dpi "$DPI" \
  --density_max_error_m "$DENSITY_MAX_ERROR_M"

echo "=== DONE H045 ==="
echo "$OUT_DIR"
date
