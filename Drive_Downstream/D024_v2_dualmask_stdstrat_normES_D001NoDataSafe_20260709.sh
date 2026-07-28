#!/usr/bin/env bash
#SBATCH -J d024_stdstrat_NDsafe
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 7-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/D024_stdstrat_NDsafe_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/D024_stdstrat_NDsafe_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# D024 D001NoDataSafe: make high/mid/low tile_std_safe river split, then train using normalized early stopping.
#
# Submit example:
#   sbatch -p HydroIntel -w execute-4006 --export=ALL,GPU_ID=1,STD_SPLIT_SEED=42 D024_v2_dualmask_stdstrat_normES_sbatch_20260708.sh

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
CODE=${CODE:-$ROOT/mae_Retrain}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2/Tiles_1m}
CV_ROOT=${CV_ROOT:-$WORK/cross_validation_v2}

# Tag all new split/run folders so they do not mix with pre-NoDataSafe outputs.
# Set DATA_TAG="" to recover the original naming behavior.
DATA_TAG=${DATA_TAG:-D001NoDataSafe}
if [[ -n "$DATA_TAG" ]]; then
  SAFE_DATA_TAG=$(echo "$DATA_TAG" | sed 's/[^A-Za-z0-9_]/_/g')
  DATA_SUFFIX="_${SAFE_DATA_TAG}"
else
  SAFE_DATA_TAG=""
  DATA_SUFFIX=""
fi

SPLIT_SCRIPT=${SPLIT_SCRIPT:-$WORK/script/A020s_prepare_v2_dualmask_stdstratified_river_split_20260708.py}
TRAIN_BACKEND=${TRAIN_BACKEND:-$WORK/script/D020_v2_dualmask_coreloss_train_backend_normES_20260708.sh}

GPU_ID=${GPU_ID:-0}
EPOCHS=${EPOCHS:-400}
BATCH_SIZE=${BATCH_SIZE:-4}
ACCUM_ITER=${ACCUM_ITER:-4}
NUM_WORKERS=${NUM_WORKERS:-1}
PATIENCE=${PATIENCE:-60}
FRESH_RUN=${FRESH_RUN:-1}

BEST_METRIC=${BEST_METRIC:-val_loss}
EARLY_STOP_METRIC=${EARLY_STOP_METRIC:-$BEST_METRIC}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.0001}
EARLY_STOP_WARMUP_EPOCHS=${EARLY_STOP_WARMUP_EPOCHS:-20}

STD_SCALE=${STD_SCALE:-1.5}
STD_SPLIT_SEED=${STD_SPLIT_SEED:-42}
VAL_PER_BIN=${VAL_PER_BIN:-1}
BIN_STAT=${BIN_STAT:-median}

# Optional manual validation rivers for the std-stratified experiment.
# Example:
#   VAL_RIVERS="CO_UpperColorado_Topobathy_1_2020 WA_Nisqually_Bathymetric_2020 NE_Niobrara_Topobathy_2018"
VAL_RIVERS=${VAL_RIVERS:-}
MANUAL_VAL_TAG=${MANUAL_VAL_TAG:-CO_Nisqually_NE}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

cd "$CODE"

if [[ -z "${RUN_STAGE:-}" ]]; then
  if [[ "$EPOCHS" -le 5 ]]; then
    RUN_STAGE="smoke_stdstrat${DATA_SUFFIX}"
  else
    RUN_STAGE="train_stdstrat${DATA_SUFFIX}"
  fi
fi

METRIC_TAG=$(echo "$EARLY_STOP_METRIC" | sed 's/[^A-Za-z0-9_]/_/g')
if [[ -n "$VAL_RIVERS" ]]; then
  SAFE_MANUAL_VAL_TAG=$(echo "$MANUAL_VAL_TAG" | sed 's/[^A-Za-z0-9_]/_/g')
  SPLIT_NAME="stdStratRiver_manualVal_${SAFE_MANUAL_VAL_TAG}_seed${STD_SPLIT_SEED}${DATA_SUFFIX}"
else
  SPLIT_NAME="stdStratRiver_${BIN_STAT}_valPerBin${VAL_PER_BIN}_seed${STD_SPLIT_SEED}${DATA_SUFFIX}"
fi
SPLIT_DIR=${SPLIT_DIR:-$CV_ROOT/splits/$SPLIT_NAME}
RUN_NAME=${RUN_NAME:-${RUN_STAGE}_${SPLIT_NAME}_v2_dualmask_corePixelLoss_ES-${METRIC_TAG}_e${EPOCHS}_b${BATCH_SIZE}_acc${ACCUM_ITER}}
OUT_DIR=${OUT_DIR:-$CV_ROOT/runs/$SPLIT_NAME/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

mkdir -p "$CV_ROOT/logs"

echo "=== D024 std-stratified river training, normalized early stop ==="
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOST=$(hostname)"
echo "DATA_TAG=$DATA_TAG"
echo "DATA_SUFFIX=$DATA_SUFFIX"
echo "SPLIT_NAME=$SPLIT_NAME"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "OUT_DIR=$OUT_DIR"
echo "GPU_ID=$GPU_ID"
echo "BEST_METRIC=$BEST_METRIC"
echo "EARLY_STOP_METRIC=$EARLY_STOP_METRIC"
echo "STD_SCALE=$STD_SCALE"
echo "BIN_STAT=$BIN_STAT"
echo "VAL_PER_BIN=$VAL_PER_BIN"
echo "STD_SPLIT_SEED=$STD_SPLIT_SEED"
echo "VAL_RIVERS=$VAL_RIVERS"
echo "MANUAL_VAL_TAG=$MANUAL_VAL_TAG"
date
nvidia-smi || true

VAL_ARGS=()
if [[ -n "$VAL_RIVERS" ]]; then
  read -r -a VAL_RIVER_ARRAY <<< "$VAL_RIVERS"
  VAL_ARGS+=(--val_rivers "${VAL_RIVER_ARRAY[@]}")
fi

python "$SPLIT_SCRIPT" \
  --tile_root "$TILE_ROOT" \
  --out_dir "$SPLIT_DIR" \
  --std_scale "$STD_SCALE" \
  --bin_stat "$BIN_STAT" \
  --val_per_bin "$VAL_PER_BIN" \
  --seed "$STD_SPLIT_SEED" \
  --visible_only \
  "${VAL_ARGS[@]}"

SPLIT_DIR="$SPLIT_DIR" \
RUN_NAME="$RUN_NAME" \
OUT_DIR="$OUT_DIR" \
LOG_DIR="$LOG_DIR" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
ACCUM_ITER="$ACCUM_ITER" \
GPU_ID="$GPU_ID" \
NUM_WORKERS="$NUM_WORKERS" \
PATIENCE="$PATIENCE" \
BEST_METRIC="$BEST_METRIC" \
EARLY_STOP_METRIC="$EARLY_STOP_METRIC" \
EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
EARLY_STOP_WARMUP_EPOCHS="$EARLY_STOP_WARMUP_EPOCHS" \
FRESH_RUN="$FRESH_RUN" \
bash "$TRAIN_BACKEND"
