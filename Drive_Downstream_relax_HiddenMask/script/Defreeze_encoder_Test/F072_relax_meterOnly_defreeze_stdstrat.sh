#!/usr/bin/env bash
#SBATCH -J F072_meter_defreeze_std
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Optional std-strat version of the MeterOnly defreeze-last-1 experiment.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
RESULTS_ROOT=${RESULTS_ROOT:-$RELAX_ROOT/results}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2_D001c_AnyVisiblePatch/Tiles_1m}

SPLIT_ROOT=${SPLIT_ROOT:-$RESULTS_ROOT/NormOnly}
SOURCE_METER_ROOT=${SOURCE_METER_ROOT:-$RESULTS_ROOT/MeterOnly}
CV_ROOT=${CV_ROOT:-$RESULTS_ROOT/MeterOnly_DefreezeLast1}

SOURCE_SPLIT_NAME=${SOURCE_SPLIT_NAME:-stdStratRiver_manualVal_CO_Nisqually_NE_seed42_D001cAnyVisiblePatch_D001NoDataSafe}
TRAIN_BACKEND=${TRAIN_BACKEND:-$RELAX_ROOT/script/F070_relax_meterOnly_defreeze_train.sh}

GPU_ID=${GPU_ID:-0}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4}
ACCUM_ITER=${ACCUM_ITER:-4}
NUM_WORKERS=${NUM_WORKERS:-1}
LR=${LR:-1e-6}
MIN_LR=${MIN_LR:-1e-8}
PATIENCE=${PATIENCE:-20}
FRESH_RUN=${FRESH_RUN:-1}
OVERWRITE_DEFREEZE=${OVERWRITE_DEFREEZE:-0}

SPLIT_DIR=${SPLIT_DIR:-$SPLIT_ROOT/splits/$SOURCE_SPLIT_NAME}
SOURCE_METER_RUN_DIR=${SOURCE_METER_RUN_DIR:-}
INIT_CKPT=${INIT_CKPT:-}

RUNTIME_LOG_DIR="$CV_ROOT/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/F072_defreeze_std_${RUNTIME_JOB_ID}.out" \
     2>"$RUNTIME_LOG_DIR/F072_defreeze_std_${RUNTIME_JOB_ID}.err"

latest_run_with_ckpt() {
  local parent="$1"
  [[ -d "$parent" ]] || { echo ""; return 0; }
  find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth \
    -printf '%T@ %h\n' 2>/dev/null | sort -nr | awk 'NR==1{print $2}'
}

SOURCE_PARENT="$SOURCE_METER_ROOT/runs/$SOURCE_SPLIT_NAME"
if [[ -z "$SOURCE_METER_RUN_DIR" ]]; then
  SOURCE_METER_RUN_DIR=$(latest_run_with_ckpt "$SOURCE_PARENT")
fi
if [[ -z "$SOURCE_METER_RUN_DIR" ]]; then
  echo "[ERROR] No MeterOnly std-strat checkpoint under $SOURCE_PARENT" >&2
  exit 2
fi
INIT_CKPT=${INIT_CKPT:-$SOURCE_METER_RUN_DIR/checkpoint-best.pth}

RUN_TAG=${RUN_TAG:-F070MeterOnlyDefreezeLast1_D001cAnyVisiblePatch_D001NoDataSafe}
RUN_NAME=${RUN_NAME:-train_${RUN_TAG}_${SOURCE_SPLIT_NAME}_fromMeterBest_e${EPOCHS}_lr${LR}}
OUT_DIR=${OUT_DIR:-$CV_ROOT/runs/$SOURCE_SPLIT_NAME/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

SPLIT_DIR="$SPLIT_DIR" \
TILE_ROOT="$TILE_ROOT" \
INIT_CKPT="$INIT_CKPT" \
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
TRAINABLE_LAST_N_ENCODER_BLOCKS=1 \
OPTIMIZATION_LOSS=meter_mae \
BEST_METRIC=val_mae_m_mask \
EARLY_STOP_METRIC=val_mae_m_mask \
EARLY_STOP_MIN_DELTA=0.001 \
EARLY_STOP_WARMUP_EPOCHS=0 \
BASELINE_EVAL_BEFORE_TRAINING=1 \
WARMUP_EPOCHS=0 \
FRESH_RUN="$FRESH_RUN" \
OVERWRITE_DEFREEZE="$OVERWRITE_DEFREEZE" \
bash "$TRAIN_BACKEND"
