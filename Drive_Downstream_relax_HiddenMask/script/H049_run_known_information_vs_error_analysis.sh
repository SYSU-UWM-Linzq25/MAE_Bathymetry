#!/usr/bin/env bash
#SBATCH -J H049_known_error
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

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
STRICT_RESULTS=${STRICT_RESULTS:-$ROOT/Downstream_Task_Bathy/Results}
RELAX_RESULTS=${RELAX_RESULTS:-$RELAX_ROOT/results}
SCRIPT=${SCRIPT:-$RELAX_ROOT/script/H049_analyze_known_information_vs_error.py}

STRICT_NORMALIZED_PRED_ROOT=${STRICT_NORMALIZED_PRED_ROOT:-$STRICT_RESULTS/FullRiver_Predictions_F010_TileAvgVRT_D001NoDataSafe}
STRICT_METER_PRED_ROOT=${STRICT_METER_PRED_ROOT:-$STRICT_RESULTS/FullRiver_Predictions_F060_TileAvgVRT_D003MeterMAE_BaselineEval_D001NoDataSafe}
RELAX_NORMALIZED_PRED_ROOT=${RELAX_NORMALIZED_PRED_ROOT:-$RELAX_RESULTS/FullRiver_Predictions_F049_NormalizedObjective_D001cAnyVisiblePatch}
RELAX_METER_PRED_ROOT=${RELAX_METER_PRED_ROOT:-$RELAX_RESULTS/FullRiver_Predictions_F044_MeterOnly_D001cAnyVisiblePatch}
RELAX_TILE_BASE=${RELAX_TILE_BASE:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_FullRiver_E001c_AnyVisiblePatch}

OUT_DIR=${OUT_DIR:-$RELAX_RESULTS/H049_KnownInformation_vs_Error}
OVERWRITE=${OVERWRITE:-0}
PATCH_SIZE=${PATCH_SIZE:-16}
MIN_COMMON_PIXELS_PER_POINT=${MIN_COMMON_PIXELS_PER_POINT:-100}
BIN_WIDTH_PERCENT=${BIN_WIDTH_PERCENT:-10}
DPI=${DPI:-220}

CONDA_SH=${CONDA_SH:-/home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-$ROOT/conda_envs/mae_zequn}
PYTHON_BIN=${PYTHON_BIN:-$CONDA_ENV/bin/python}

for file in "$SCRIPT" "$PYTHON_BIN"; do
  [[ -e "$file" ]] || { echo "[ERROR] Missing: $file" >&2; exit 2; }
done

if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -not -path "$OUT_DIR/logs*" -print -quit | grep -q .; then
  if [[ "$OVERWRITE" == "1" ]]; then
    find "$OUT_DIR" -mindepth 1 -maxdepth 1 ! -name logs -exec rm -rf {} +
  else
    echo "[ERROR] Output exists: $OUT_DIR" >&2
    echo "Set OVERWRITE=1 to rebuild." >&2
    exit 3
  fi
fi
mkdir -p "$OUT_DIR/logs"

JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$OUT_DIR/logs/H049_known_error_${JOB_ID}.out" \
     2>"$OUT_DIR/logs/H049_known_error_${JOB_ID}.err"

module purge || true
source "$CONDA_SH"
conda activate "$CONDA_ENV"

printf '%s\n' \
  "============================================================" \
  "H049 sampling-point known-information versus error" \
  "PYTHON_BIN=$PYTHON_BIN" \
  "OUT_DIR=$OUT_DIR" \
  "PRIMARY_KNOWN=visible valid 16x16 patches / all valid patches" \
  "PRIMARY_ERROR=MAE on exact per-point four-way common final pixels" \
  "PREDICTION_SOURCE=existing F010/F060/F049/F044 full-river averaged prediction tiles" \
  "============================================================"

"$PYTHON_BIN" -u "$SCRIPT" \
  --strict_normalized_pred_root "$STRICT_NORMALIZED_PRED_ROOT" \
  --strict_meter_pred_root "$STRICT_METER_PRED_ROOT" \
  --relaxed_normalized_pred_root "$RELAX_NORMALIZED_PRED_ROOT" \
  --relaxed_meter_pred_root "$RELAX_METER_PRED_ROOT" \
  --relax_tile_base "$RELAX_TILE_BASE" \
  --output_dir "$OUT_DIR" \
  --patch_size "$PATCH_SIZE" \
  --min_common_pixels_per_point "$MIN_COMMON_PIXELS_PER_POINT" \
  --bin_width_percent "$BIN_WIDTH_PERCENT" \
  --dpi "$DPI"

echo "=== DONE H049 ==="
echo "$OUT_DIR"
