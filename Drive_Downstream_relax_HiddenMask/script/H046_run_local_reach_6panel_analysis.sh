#!/usr/bin/env bash
#SBATCH -J H046_reach6
#SBATCH -p HydroIntel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Analysis only. Reads existing full-river prediction manifests.
# Strict GT/masks are read from processed E001 tiles; relaxed GT/masks
# are read from processed E001c AnyVisiblePatch tiles.
#
# For every successfully assembled continuous reach, four separate 6-panel figures are made:
#   1) strict normalized
#   2) strict meter
#   3) relaxed normalized
#   4) relaxed meter
#
# Six panels:
#   full processed GT + sampling centers
#   patch-processed Hidden Mask (0/1)
#   final prediction/loss mask (0/1)
#   GT inside final mask
#   prediction inside final mask
#   signed error inside final mask
#
# Reach metrics use exact four-way common loss pixels.
# The common-pixel threshold is used only to select best/median/worst examples;
# it does not limit the all-reach visual archive.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
STRICT_RESULTS=${STRICT_RESULTS:-$ROOT/Downstream_Task_Bathy/Results}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
RELAX_RESULTS=${RELAX_RESULTS:-$RELAX_ROOT/results}
SCRIPT=${SCRIPT:-$RELAX_ROOT/script/H046_visualize_local_reaches_6panel.py}
RESOLVER=${RESOLVER:-$RELAX_ROOT/script/H044_resolve_relaxed_normalized_prediction_root.py}

STRICT_NORMALIZED_PRED_ROOT=${STRICT_NORMALIZED_PRED_ROOT:-$STRICT_RESULTS/FullRiver_Predictions_F010_TileAvgVRT_D001NoDataSafe}
STRICT_METER_PRED_ROOT=${STRICT_METER_PRED_ROOT:-$STRICT_RESULTS/FullRiver_Predictions_F060_TileAvgVRT_D003MeterMAE_BaselineEval_D001NoDataSafe}
RELAX_METER_PRED_ROOT=${RELAX_METER_PRED_ROOT:-$RELAX_RESULTS/FullRiver_Predictions_F044_MeterOnly_D001cAnyVisiblePatch}
RELAX_NORMALIZED_PRED_ROOT=${RELAX_NORMALIZED_PRED_ROOT:-$RELAX_RESULTS/FullRiver_Predictions_F049_NormalizedObjective_D001cAnyVisiblePatch}
INPUT_AUDIT_JSON=${INPUT_AUDIT_JSON:-$RELAX_RESULTS/H046_input_resolution.json}

STRICT_TILE_BASE=${STRICT_TILE_BASE:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_FullRiver_E001}
RELAX_TILE_BASE=${RELAX_TILE_BASE:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_FullRiver_E001c_AnyVisiblePatch}
OUT_DIR=${OUT_DIR:-$RELAX_RESULTS/H046_LocalReach_6Panel_Analysis}

OVERWRITE=${OVERWRITE:-0}
SEGMENT_SIZE=${SEGMENT_SIZE:-10}
SEGMENT_STRIDE=${SEGMENT_STRIDE:-10}
MIN_POINTS=${MIN_POINTS:-5}
MIN_COMMON_PIXELS=${MIN_COMMON_PIXELS:-1000}
N_BEST=${N_BEST:-3}
N_MEDIAN=${N_MEDIAN:-3}
N_WORST=${N_WORST:-3}
N_METER_ADVANTAGE=${N_METER_ADVANTAGE:-3}
N_RELAXED_ADVANTAGE=${N_RELAXED_ADVANTAGE:-3}
DPI=${DPI:-220}
RENDER_ALL_REACHES=${RENDER_ALL_REACHES:-1}
RESUME_VISUALS=${RESUME_VISUALS:-1}
RENDER_OVERVIEW_FOR_ALL=${RENDER_OVERVIEW_FOR_ALL:-0}
MAX_RENDER_REACHES=${MAX_RENDER_REACHES:-0}
RENDER_PROGRESS_EVERY=${RENDER_PROGRESS_EVERY:-10}

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
exec >"$LOG_DIR/H046_local_reach_6panel_${JOB_ID}.out" \
     2>"$LOG_DIR/H046_local_reach_6panel_${JOB_ID}.err"

module purge || true
source "$CONDA_SH"
conda activate "$CONDA_ENV"

for path in \
  "$SCRIPT" \
  "$RESOLVER" \
  "$STRICT_NORMALIZED_PRED_ROOT" \
  "$STRICT_METER_PRED_ROOT" \
  "$RELAX_NORMALIZED_PRED_ROOT" \
  "$RELAX_METER_PRED_ROOT" \
  "$STRICT_TILE_BASE" \
  "$RELAX_TILE_BASE"; do
  [[ -e "$path" ]] || { echo "[ERROR] Missing: $path" >&2; exit 2; }
done

if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -not -path "$OUT_DIR/logs*" -print -quit | grep -q .; then
  if [[ "$OVERWRITE" == "1" ]]; then
    find "$OUT_DIR" -mindepth 1 -maxdepth 1 ! -name logs -exec rm -rf {} +
  elif [[ "$RESUME_VISUALS" == "1" || "$RESUME_VISUALS" == "true" || "$RESUME_VISUALS" == "TRUE" ]]; then
    echo "[RESUME] Existing H046 archive will be scanned and complete reach figures reused: $OUT_DIR"
  else
    echo "[ERROR] Output exists: $OUT_DIR" >&2
    echo "Use RESUME_VISUALS=1 to continue, or OVERWRITE=1 to rebuild." >&2
    exit 3
  fi
fi
mkdir -p "$OUT_DIR"

RENDER_ALL_ARGS=(--render_all_reaches)
if [[ "$RENDER_ALL_REACHES" == "0" || "$RENDER_ALL_REACHES" == "false" || "$RENDER_ALL_REACHES" == "FALSE" ]]; then
  RENDER_ALL_ARGS=(--no-render_all_reaches)
fi

RESUME_VISUAL_ARGS=(--resume_visuals)
if [[ "$RESUME_VISUALS" == "0" || "$RESUME_VISUALS" == "false" || "$RESUME_VISUALS" == "FALSE" ]]; then
  RESUME_VISUAL_ARGS=(--no-resume_visuals)
fi

OVERVIEW_ARGS=(--no-render_overview_for_all)
if [[ "$RENDER_OVERVIEW_FOR_ALL" == "1" || "$RENDER_OVERVIEW_FOR_ALL" == "true" || "$RENDER_OVERVIEW_FOR_ALL" == "TRUE" ]]; then
  OVERVIEW_ARGS=(--render_overview_for_all)
fi

echo "============================================================"
echo "H046 analysis only: local continuous reaches"
date
echo "STRICT_NORMALIZED_PRED_ROOT=$STRICT_NORMALIZED_PRED_ROOT"
echo "STRICT_METER_PRED_ROOT=$STRICT_METER_PRED_ROOT"
echo "RELAX_NORMALIZED_PRED_ROOT=$RELAX_NORMALIZED_PRED_ROOT"
echo "INPUT_AUDIT_JSON=$INPUT_AUDIT_JSON"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "RELAX_METER_PRED_ROOT=$RELAX_METER_PRED_ROOT"
echo "STRICT_TILE_BASE=$STRICT_TILE_BASE"
echo "RELAX_TILE_BASE=$RELAX_TILE_BASE"
echo "TILE_POLICY=Strict GT/masks from E001; relaxed GT/masks from E001c AnyVisiblePatch"
echo "SEGMENT_SIZE=$SEGMENT_SIZE"
echo "SEGMENT_STRIDE=$SEGMENT_STRIDE"
echo "COMMON_FOOTPRINT=four-way common loss pixels"
echo "RENDER_ALL_REACHES=$RENDER_ALL_REACHES"
echo "RESUME_VISUALS=$RESUME_VISUALS"
echo "RENDER_OVERVIEW_FOR_ALL=$RENDER_OVERVIEW_FOR_ALL"
echo "MAX_RENDER_REACHES=$MAX_RENDER_REACHES"
echo "OUT_DIR=$OUT_DIR"
echo "============================================================"

"$PYTHON_BIN" -u "$SCRIPT" \
  --strict_normalized_pred_root "$STRICT_NORMALIZED_PRED_ROOT" \
  --strict_meter_pred_root "$STRICT_METER_PRED_ROOT" \
  --relaxed_normalized_pred_root "$RELAX_NORMALIZED_PRED_ROOT" \
  --relaxed_meter_pred_root "$RELAX_METER_PRED_ROOT" \
  --strict_tile_base "$STRICT_TILE_BASE" \
  --relax_tile_base "$RELAX_TILE_BASE" \
  --output_dir "$OUT_DIR" \
  --segment_size "$SEGMENT_SIZE" \
  --segment_stride "$SEGMENT_STRIDE" \
  --min_points "$MIN_POINTS" \
  --min_common_pixels "$MIN_COMMON_PIXELS" \
  --n_best "$N_BEST" \
  --n_median "$N_MEDIAN" \
  --n_worst "$N_WORST" \
  --n_meter_advantage "$N_METER_ADVANTAGE" \
  --n_relaxed_advantage "$N_RELAXED_ADVANTAGE" \
  --dpi "$DPI" \
  --max_render_reaches "$MAX_RENDER_REACHES" \
  --render_progress_every "$RENDER_PROGRESS_EVERY" \
  "${RENDER_ALL_ARGS[@]}" \
  "${RESUME_VISUAL_ARGS[@]}" \
  "${OVERVIEW_ARGS[@]}"

echo "=== DONE H046 ==="
echo "$OUT_DIR"
date
