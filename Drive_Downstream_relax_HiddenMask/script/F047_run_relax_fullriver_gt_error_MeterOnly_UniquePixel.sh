#!/usr/bin/env bash
#SBATCH -J F047_relax_meter_gt_error
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
RESULTS_ROOT=${RESULTS_ROOT:-$RELAX_ROOT/results}
SCRIPT=${SCRIPT:-$RELAX_ROOT/script/F046_relax_fullriver_gt_error_MeterOnly_UniquePixel.py}
RUN_TAG=${RUN_TAG:-D044MeterOnly_D001cAnyVisiblePatch_D001NoDataSafe}

F044_BASE=${F044_BASE:-$RESULTS_ROOT/FullRiver_Predictions_F044_MeterOnly_D001cAnyVisiblePatch}
OUT_BASE=${OUT_BASE:-$RESULTS_ROOT/FullRiver_GT_Error_F046_MeterOnly_D001cAnyVisiblePatch}

HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}
RIVERS=${RIVERS:-}
OVERWRITE=${OVERWRITE:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}
F044_OUT_DIR=${F044_OUT_DIR:-}
OUT_DIR=${OUT_DIR:-}

RUNTIME_LOG_DIR="$OUT_BASE/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/F047_relax_gt_error_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.out" \
     2>"$RUNTIME_LOG_DIR/F047_relax_gt_error_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.err"

safe_name() { echo "$1" | sed 's/[^A-Za-z0-9_]/_/g'; }

case "$HOLDOUT_PRESET" in
  CO)      DEFAULT_RIVERS="CO_UpperColorado_Topobathy_1_2020" ;;
  CA)      DEFAULT_RIVERS="CA_KlamathRiver_TopoBathy_2018_D18" ;;
  Santiam) DEFAULT_RIVERS="OR_SantiamRiverTB_Topobathy_1_D23" ;;
  *) echo "[ERROR] F047 formal workflow supports CA, CO, Santiam. Got $HOLDOUT_PRESET" >&2; exit 2 ;;
esac
RIVERS=${RIVERS:-$DEFAULT_RIVERS}

SAFE_PRESET=$(safe_name "$HOLDOUT_PRESET")
DEFAULT_TAG="holdout_${SAFE_PRESET}_${RUN_TAG}"
F044_OUT_DIR=${F044_OUT_DIR:-$F044_BASE/$DEFAULT_TAG}
OUT_DIR=${OUT_DIR:-$OUT_BASE/$DEFAULT_TAG}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing script: $SCRIPT" >&2; exit 2; }
[[ -d "$F044_OUT_DIR" ]] || { echo "[ERROR] Missing F044_OUT_DIR: $F044_OUT_DIR" >&2; exit 2; }

if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$OUT_DIR"
  else
    echo "[ERROR] Output is not empty: $OUT_DIR" >&2
    exit 3
  fi
fi
mkdir -p "$OUT_DIR"

read -r -a RIVER_ARRAY <<< "$RIVERS"
[[ ${#RIVER_ARRAY[@]} -gt 0 ]] || { echo "[ERROR] RIVERS is empty." >&2; exit 2; }

echo "============================================================"
echo "F047 D001c/E001c MeterOnly GT/error and unique-pixel metrics"
date
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "F044_OUT_DIR=$F044_OUT_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "RIVERS=$RIVERS"
echo "MASK=Core_Loss_Mask_Pixel AND valid_GT AND corrected valid_prediction"
echo "PREDICTION_QA=F044 must have prediction_patch_filter_applied=true"
echo "PRIMARY_METRICS=each overlap-averaged geospatial pixel counted once"
echo "============================================================"

python -u "$SCRIPT" \
  --f044_out_dir "$F044_OUT_DIR" \
  --output_dir "$OUT_DIR" \
  --rivers "${RIVER_ARRAY[@]}" \
  --nodata -999999 \
  --nodata_threshold -9999 \
  --progress_every "$PROGRESS_EVERY"

echo "=== DONE F047 ==="
echo "$OUT_DIR"
date
