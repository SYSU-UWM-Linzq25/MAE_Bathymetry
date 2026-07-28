#!/usr/bin/env bash
#SBATCH -J F073_n2m_error
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

# F073: exact unique-geospatial GT/error evaluation for F071 predictions.
# Reuses the validated F062 evaluator so both model families use the same
# comparison mask and overlap accounting.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
SCRIPT=${SCRIPT:-$WORK/script/F027_fullriver_gt_error_NormThenMeter_UniquePixel_20260727.py}
MODEL_CV_ROOT=${MODEL_CV_ROOT:-$WORK/cross_validation_v6_Stage2MeterMAE_FromNorm}
RUN_TAG=${RUN_TAG:-D005Stage2MeterMAE_FromNorm_D001NoDataSafe}

NORM_METER_PRED_BASE=${NORM_METER_PRED_BASE:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Predictions_G001_NormThenMeter_D001NoDataSafe}
OUT_BASE=${OUT_BASE:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_GT_Error_G002_NormThenMeter_D001NoDataSafe}

HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}
RIVERS=${RIVERS:-}
OVERWRITE=${OVERWRITE:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}
NORM_METER_PRED_DIR=${NORM_METER_PRED_DIR:-}
OUT_DIR=${OUT_DIR:-}

RUNTIME_LOG_DIR="$MODEL_CV_ROOT/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/F073_n2m_error_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.out" \
     2>"$RUNTIME_LOG_DIR/F073_n2m_error_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.err"

safe_name() { echo "$1" | sed 's/[^A-Za-z0-9_]/_/g'; }

case "$HOLDOUT_PRESET" in
  CO) DEFAULT_RIVERS="CO_UpperColorado_Topobathy_1_2020" ;;
  CA) DEFAULT_RIVERS="CA_KlamathRiver_TopoBathy_2018_D18" ;;
  Santiam) DEFAULT_RIVERS="OR_SantiamRiverTB_Topobathy_1_D23" ;;
  *) echo "[ERROR] F073 formal comparison supports CA, CO, and Santiam. Got $HOLDOUT_PRESET" >&2; exit 2 ;;
esac
RIVERS=${RIVERS:-$DEFAULT_RIVERS}

SAFE_PRESET=$(safe_name "$HOLDOUT_PRESET")
DEFAULT_TAG="holdout_${SAFE_PRESET}_${RUN_TAG}"
NORM_METER_PRED_DIR=${NORM_METER_PRED_DIR:-$NORM_METER_PRED_BASE/$DEFAULT_TAG}
OUT_DIR=${OUT_DIR:-$OUT_BASE/$DEFAULT_TAG}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing script: $SCRIPT" >&2; exit 2; }
[[ -d "$NORM_METER_PRED_DIR" ]] || { echo "[ERROR] Missing NORM_METER_PRED_DIR: $NORM_METER_PRED_DIR" >&2; exit 2; }

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
echo "F073 normalized -> meter full-river GT/error evaluation"
date
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "NORM_METER_PRED_DIR=$NORM_METER_PRED_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "RIVERS=$RIVERS"
echo "MASK=Core_Loss_Mask_Pixel AND valid_GT AND valid_prediction"
echo "PRIMARY_METRICS=each overlap-averaged geospatial pixel counted once"
echo "============================================================"

python -u "$SCRIPT" \
  --f060_out_dir "$NORM_METER_PRED_DIR" \
  --output_dir "$OUT_DIR" \
  --rivers "${RIVER_ARRAY[@]}" \
  --nodata -999999 \
  --nodata_threshold -9999 \
  --progress_every "$PROGRESS_EVERY"

echo "=== DONE F073 ==="
echo "$OUT_DIR"
date
