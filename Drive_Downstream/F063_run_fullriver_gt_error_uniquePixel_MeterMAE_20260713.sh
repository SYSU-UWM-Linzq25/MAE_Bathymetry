#!/usr/bin/env bash
#SBATCH -J F063_meter_gt_error
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
SCRIPT=${SCRIPT:-$WORK/script/F062_fullriver_gt_error_uniquePixel_MeterMAE_20260713.py}
MODEL_CV_ROOT=${MODEL_CV_ROOT:-$WORK/cross_validation_v4_meterMAE_BaselineEval}
RUN_TAG=${RUN_TAG:-D003MeterMAE_BaselineEval_D001NoDataSafe}

F060_BASE=${F060_BASE:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Predictions_F060_TileAvgVRT_D003MeterMAE_BaselineEval_D001NoDataSafe}
OUT_BASE=${OUT_BASE:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_GT_Error_F062_UniquePixel_D003MeterMAE_BaselineEval_D001NoDataSafe}

HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}
RIVERS=${RIVERS:-}
OVERWRITE=${OVERWRITE:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}
F060_OUT_DIR=${F060_OUT_DIR:-}
OUT_DIR=${OUT_DIR:-}

RUNTIME_LOG_DIR="$MODEL_CV_ROOT/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/F063_meter_gt_error_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.out" \
     2>"$RUNTIME_LOG_DIR/F063_meter_gt_error_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.err"

safe_name() { echo "$1" | sed 's/[^A-Za-z0-9_]/_/g'; }

case "$HOLDOUT_PRESET" in
  CO) DEFAULT_RIVERS="CO_UpperColorado_Topobathy_1_2020" ;;
  CA) DEFAULT_RIVERS="CA_KlamathRiver_TopoBathy_2018_D18" ;;
  Santiam) DEFAULT_RIVERS="OR_SantiamRiverTB_Topobathy_1_D23" ;;
  NE) DEFAULT_RIVERS="NE_Niobrara_Topobathy_2018" ;;
  OR_MKRC) DEFAULT_RIVERS="OR_MKRC_Topobathy_2021" ;;
  Nisqually) DEFAULT_RIVERS="WA_Nisqually_Bathymetric_2020" ;;
  MD) DEFAULT_RIVERS="MD_PotomacRiver_Bathy_2019" ;;
  Chehalis) DEFAULT_RIVERS="WA_ChehalisRiverTB_Topobathy_1_D23" ;;
  MilwaukeeGroup) DEFAULT_RIVERS="BadgerFinNull Estabrook_Combined KewaFix2Null Kletzch_Combined_UpMax3Null" ;;
  *) echo "[ERROR] Unknown HOLDOUT_PRESET=$HOLDOUT_PRESET" >&2; exit 2 ;;
esac
RIVERS=${RIVERS:-$DEFAULT_RIVERS}

SAFE_PRESET=$(safe_name "$HOLDOUT_PRESET")
DEFAULT_TAG="holdout_${SAFE_PRESET}_${RUN_TAG}"
F060_OUT_DIR=${F060_OUT_DIR:-$F060_BASE/$DEFAULT_TAG}
OUT_DIR=${OUT_DIR:-$OUT_BASE/$DEFAULT_TAG}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing script: $SCRIPT" >&2; exit 2; }
[[ -d "$F060_OUT_DIR" ]] || { echo "[ERROR] Missing F060_OUT_DIR: $F060_OUT_DIR" >&2; exit 2; }

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
echo "F063 meter-MAE GT/error and exact unique-pixel metrics"
date
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "F060_OUT_DIR=$F060_OUT_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "RIVERS=$RIVERS"
echo "MASK=Core_Loss_Mask_Pixel AND valid_GT AND valid_prediction"
echo "PRIMARY_METRICS=each overlap-averaged geospatial pixel counted once"
echo "LEGACY_METRICS=tile-footprint fields retained"
echo "============================================================"

python -u "$SCRIPT" \
  --f060_out_dir "$F060_OUT_DIR" \
  --output_dir "$OUT_DIR" \
  --rivers "${RIVER_ARRAY[@]}" \
  --nodata -999999 \
  --nodata_threshold -9999 \
  --progress_every "$PROGRESS_EVERY"

echo "=== DONE F063 ==="
echo "$OUT_DIR"
date
