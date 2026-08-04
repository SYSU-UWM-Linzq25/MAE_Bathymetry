#!/usr/bin/env bash
#SBATCH -J D056_m2n_std_AP
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
RESULTS_ROOT=${RESULTS_ROOT:-$RELAX_ROOT/results}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2_D001c_AnyVisiblePatch/Tiles_1m}

SPLIT_ROOT=${SPLIT_ROOT:-$RESULTS_ROOT/NormOnly}
STAGE1_CV_ROOT=${STAGE1_CV_ROOT:-$RESULTS_ROOT/MeterOnly}
CV_ROOT=${CV_ROOT:-$RESULTS_ROOT/MeterThenNorm}

SOURCE_SPLIT_NAME=${SOURCE_SPLIT_NAME:-stdStratRiver_manualVal_CO_Nisqually_NE_seed42_D001cAnyVisiblePatch_D001NoDataSafe}
TRAIN_BACKEND=${TRAIN_BACKEND:-$RELAX_ROOT/script/D054_relax_train_meter2norm.sh}

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

SPLIT_DIR=${SPLIT_DIR:-$SPLIT_ROOT/splits/$SOURCE_SPLIT_NAME}
STAGE1_RUN_DIR=${STAGE1_RUN_DIR:-}
STAGE1_CKPT=${STAGE1_CKPT:-}

RUNTIME_LOG_DIR="$CV_ROOT/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/D056_m2n_std_${RUNTIME_JOB_ID}.out" \
     2>"$RUNTIME_LOG_DIR/D056_m2n_std_${RUNTIME_JOB_ID}.err"

latest_run_with_ckpt() {
  local parent="$1"
  [[ -d "$parent" ]] || { echo ""; return 0; }
  find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth \
    -printf '%T@ %h\n' 2>/dev/null | sort -nr | awk 'NR==1{print $2}'
}

STAGE1_PARENT="$STAGE1_CV_ROOT/runs/$SOURCE_SPLIT_NAME"
if [[ -z "$STAGE1_RUN_DIR" ]]; then
  STAGE1_RUN_DIR=$(latest_run_with_ckpt "$STAGE1_PARENT")
fi
[[ -n "$STAGE1_RUN_DIR" ]] || {
  echo "[ERROR] No D001c MeterOnly std-strat checkpoint under $STAGE1_PARENT" >&2
  exit 2
}
STAGE1_CKPT=${STAGE1_CKPT:-$STAGE1_RUN_DIR/checkpoint-best.pth}

RUN_NAME=${RUN_NAME:-train_stage2_normMSE_meterSelect_fromMeterBest_${SOURCE_SPLIT_NAME}_e${EPOCHS}_lr${LR}}
OUT_DIR=${OUT_DIR:-$CV_ROOT/runs/$SOURCE_SPLIT_NAME/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

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
