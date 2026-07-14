#!/usr/bin/env bash
#SBATCH -J D035_s2_norm_meter
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2/Tiles_1m}

SOURCE_CV_ROOT=${SOURCE_CV_ROOT:-$WORK/cross_validation_v2}
STAGE1_CV_ROOT=${STAGE1_CV_ROOT:-$WORK/cross_validation_v4_meterMAE_BaselineEval}
CV_ROOT=${CV_ROOT:-$WORK/cross_validation_v5_Stage2NormMSE_MeterSelect}

SOURCE_SPLIT_TAG=${SOURCE_SPLIT_TAG:-D001NoDataSafe}
STAGE1_RUN_TAG=${STAGE1_RUN_TAG:-D003MeterMAE_BaselineEval_D001NoDataSafe}
RUN_TAG=${RUN_TAG:-D004Stage2NormMSE_MeterSelect_D001NoDataSafe}
TRAIN_BACKEND=${TRAIN_BACKEND:-$WORK/script/D034_train_stage2_NormalizedMSE_MeterSelect_backend_20260713.sh}

HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}
HOLDOUT_NAME=${HOLDOUT_NAME:-}
HOLDOUT_RIVERS=${HOLDOUT_RIVERS:-}

GPU_ID=${GPU_ID:-0}
EPOCHS=${EPOCHS:-120}
BATCH_SIZE=${BATCH_SIZE:-4}
ACCUM_ITER=${ACCUM_ITER:-4}
NUM_WORKERS=${NUM_WORKERS:-1}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-7}
PATIENCE=${PATIENCE:-30}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
EARLY_STOP_WARMUP_EPOCHS=${EARLY_STOP_WARMUP_EPOCHS:-0}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.001}
FRESH_RUN=${FRESH_RUN:-1}
OVERWRITE_STAGE2=${OVERWRITE_STAGE2:-0}

STAGE1_RUN_DIR=${STAGE1_RUN_DIR:-}
STAGE1_CKPT=${STAGE1_CKPT:-}
OUT_DIR=${OUT_DIR:-}
RUN_NAME=${RUN_NAME:-}

RUNTIME_LOG_DIR="$CV_ROOT/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/D035_stage2_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.out" \
     2>"$RUNTIME_LOG_DIR/D035_stage2_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.err"

latest_run_with_ckpt() {
  local parent="$1"
  if [[ ! -d "$parent" ]]; then echo ""; return 0; fi
  find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth -printf '%T@ %h\n' 2>/dev/null \
    | sort -nr | awk 'NR==1{print $2}'
}

case "$HOLDOUT_PRESET" in
  CO)
    HOLDOUT_NAME=${HOLDOUT_NAME:-CO_UpperColorado_Topobathy_1_2020}
    HOLDOUT_RIVERS=${HOLDOUT_RIVERS:-CO_UpperColorado_Topobathy_1_2020}
    ;;
  CA)
    HOLDOUT_NAME=${HOLDOUT_NAME:-CA_KlamathRiver_TopoBathy_2018_D18}
    HOLDOUT_RIVERS=${HOLDOUT_RIVERS:-CA_KlamathRiver_TopoBathy_2018_D18}
    ;;
  Santiam)
    HOLDOUT_NAME=${HOLDOUT_NAME:-OR_SantiamRiverTB_Topobathy_1_D23}
    HOLDOUT_RIVERS=${HOLDOUT_RIVERS:-OR_SantiamRiverTB_Topobathy_1_D23}
    ;;
  *)
    echo "[ERROR] D035 formal submission currently supports CA, CO, and Santiam. Got $HOLDOUT_PRESET" >&2
    exit 2
    ;;
esac

SAFE_PRESET=$(echo "$HOLDOUT_PRESET" | sed 's/[^A-Za-z0-9_]/_/g')
SPLIT_DIR=${SPLIT_DIR:-$SOURCE_CV_ROOT/splits/holdout_${SAFE_PRESET}_${SOURCE_SPLIT_TAG}}

STAGE1_PARENT="$STAGE1_CV_ROOT/runs/holdout_${SAFE_PRESET}_${STAGE1_RUN_TAG}"
if [[ -z "$STAGE1_RUN_DIR" ]]; then STAGE1_RUN_DIR=$(latest_run_with_ckpt "$STAGE1_PARENT"); fi
if [[ -z "$STAGE1_RUN_DIR" ]]; then
  echo "[ERROR] Could not find Stage-1 run under $STAGE1_PARENT" >&2
  exit 2
fi
STAGE1_CKPT=${STAGE1_CKPT:-$STAGE1_RUN_DIR/checkpoint-best.pth}

RUN_NAME=${RUN_NAME:-train_stage2_normMSE_meterSelect_${HOLDOUT_NAME}_fromStage1Best_e${EPOCHS}_lr${LR}_b${BATCH_SIZE}_acc${ACCUM_ITER}}
OUT_DIR=${OUT_DIR:-$CV_ROOT/runs/holdout_${SAFE_PRESET}_${RUN_TAG}/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

for f in \
  "$TRAIN_BACKEND" "$STAGE1_CKPT" \
  "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/val_tiles.txt" \
  "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/val_hidden.txt" \
  "$SPLIT_DIR/train_loss.txt" "$SPLIT_DIR/val_loss.txt"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing required file: $f" >&2; exit 2; }
done

echo "============================================================"
echo "D035 holdout Stage-2 normalized-MSE optimization / meter-MAE selection"
date
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "HOLDOUT_NAME=$HOLDOUT_NAME"
echo "SOURCE_SPLIT=$SPLIT_DIR"
echo "STAGE1_RUN_DIR=$STAGE1_RUN_DIR"
echo "STAGE1_CKPT=$STAGE1_CKPT"
echo "OUT_DIR=$OUT_DIR"
echo "OPTIMIZATION_LOSS=normalized_mse"
echo "BEST_METRIC=val_mae_m_mask"
echo "EARLY_STOP_METRIC=val_mae_m_mask"
echo "BASELINE=Stage-1 checkpoint evaluated at epoch -1"
echo "LR=$LR"
echo "EPOCHS=$EPOCHS"
echo "PATIENCE=$PATIENCE"
echo "============================================================"

SPLIT_DIR="$SPLIT_DIR" \
TILE_ROOT="$TILE_ROOT" \
STAGE1_CKPT="$STAGE1_CKPT" \
RUN_NAME="$RUN_NAME" \
OUT_DIR="$OUT_DIR" \
LOG_DIR="$LOG_DIR" \
GPU_ID="$GPU_ID" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
ACCUM_ITER="$ACCUM_ITER" \
NUM_WORKERS="$NUM_WORKERS" \
LR="$LR" \
MIN_LR="$MIN_LR" \
PATIENCE="$PATIENCE" \
WARMUP_EPOCHS="$WARMUP_EPOCHS" \
EARLY_STOP_WARMUP_EPOCHS="$EARLY_STOP_WARMUP_EPOCHS" \
EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
FRESH_RUN="$FRESH_RUN" \
OVERWRITE_STAGE2="$OVERWRITE_STAGE2" \
bash "$TRAIN_BACKEND"

echo "=== DONE D035 ==="
echo "$OUT_DIR"
date
