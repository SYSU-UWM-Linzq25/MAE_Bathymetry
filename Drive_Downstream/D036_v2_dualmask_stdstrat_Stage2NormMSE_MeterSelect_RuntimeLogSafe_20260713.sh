#!/usr/bin/env bash
#SBATCH -J D036_s2_norm_std
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2/Tiles_1m}

SOURCE_CV_ROOT=${SOURCE_CV_ROOT:-$WORK/cross_validation_v2}
STAGE1_CV_ROOT=${STAGE1_CV_ROOT:-$WORK/cross_validation_v4_meterMAE_BaselineEval}
CV_ROOT=${CV_ROOT:-$WORK/cross_validation_v5_Stage2NormMSE_MeterSelect}

SOURCE_SPLIT_NAME=${SOURCE_SPLIT_NAME:-stdStratRiver_manualVal_CO_Nisqually_NE_seed42_D001NoDataSafe}
RUN_TAG=${RUN_TAG:-D004Stage2NormMSE_MeterSelect_D001NoDataSafe}
TRAIN_BACKEND=${TRAIN_BACKEND:-$WORK/script/D034_train_stage2_NormalizedMSE_MeterSelect_backend_20260713.sh}

GPU_ID=${GPU_ID:-0}
EPOCHS=${EPOCHS:-120}
BATCH_SIZE=${BATCH_SIZE:-4}
ACCUM_ITER=${ACCUM_ITER:-4}
NUM_WORKERS=${NUM_WORKERS:-1}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-7}
PATIENCE=${PATIENCE:-30}
FRESH_RUN=${FRESH_RUN:-1}
OVERWRITE_STAGE2=${OVERWRITE_STAGE2:-0}

SPLIT_DIR=${SPLIT_DIR:-$SOURCE_CV_ROOT/splits/$SOURCE_SPLIT_NAME}
STAGE1_RUN_DIR=${STAGE1_RUN_DIR:-}
STAGE1_CKPT=${STAGE1_CKPT:-}

RUNTIME_LOG_DIR="$CV_ROOT/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/D036_stage2_stdstrat_${RUNTIME_JOB_ID}.out" \
     2>"$RUNTIME_LOG_DIR/D036_stage2_stdstrat_${RUNTIME_JOB_ID}.err"

latest_run_with_ckpt() {
  local parent="$1"
  if [[ ! -d "$parent" ]]; then echo ""; return 0; fi
  find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth -printf '%T@ %h\n' 2>/dev/null \
    | sort -nr | awk 'NR==1{print $2}'
}

STAGE1_PARENT="$STAGE1_CV_ROOT/runs/$SOURCE_SPLIT_NAME"
if [[ -z "$STAGE1_RUN_DIR" ]]; then STAGE1_RUN_DIR=$(latest_run_with_ckpt "$STAGE1_PARENT"); fi
if [[ -z "$STAGE1_RUN_DIR" ]]; then
  echo "[ERROR] No Stage-1 std-strat checkpoint under $STAGE1_PARENT" >&2
  exit 2
fi
STAGE1_CKPT=${STAGE1_CKPT:-$STAGE1_RUN_DIR/checkpoint-best.pth}

RUN_NAME=${RUN_NAME:-train_stage2_normMSE_meterSelect_${SOURCE_SPLIT_NAME}_fromStage1Best_e${EPOCHS}_lr${LR}}
OUT_DIR=${OUT_DIR:-$CV_ROOT/runs/$SOURCE_SPLIT_NAME/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

echo "D036 Stage-2 std-strat"
echo "SOURCE_SPLIT=$SPLIT_DIR"
echo "STAGE1_CKPT=$STAGE1_CKPT"
echo "OUT_DIR=$OUT_DIR"

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
WARMUP_EPOCHS=0 \
EARLY_STOP_WARMUP_EPOCHS=0 \
EARLY_STOP_MIN_DELTA=0.001 \
FRESH_RUN="$FRESH_RUN" \
OVERWRITE_STAGE2="$OVERWRITE_STAGE2" \
bash "$TRAIN_BACKEND"

echo "=== DONE D036 ==="
echo "$OUT_DIR"
