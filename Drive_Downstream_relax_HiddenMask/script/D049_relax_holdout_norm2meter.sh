#!/usr/bin/env bash
#SBATCH -J D049_n2m_AP
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

# RELAX PROJECT: isolated code/results under Downstream_Task_Bathy_relax_HiddenMask.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
WORK=${WORK:-$RELAX_ROOT}
RESULTS_ROOT=${RESULTS_ROOT:-$RELAX_ROOT/results}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2_D001c_AnyVisiblePatch/Tiles_1m}

SOURCE_CV_ROOT=${SOURCE_CV_ROOT:-$RESULTS_ROOT/NormOnly}
CV_ROOT=${CV_ROOT:-$RESULTS_ROOT/NormThenMeter}

SOURCE_SPLIT_TAG=${SOURCE_SPLIT_TAG:-D001cAnyVisiblePatch_D001NoDataSafe}
RUN_TAG=${RUN_TAG:-D048NormThenMeter_D001cAnyVisiblePatch_D001NoDataSafe}
TRAIN_BACKEND=${TRAIN_BACKEND:-$RELAX_ROOT/script/D048_relax_train_norm2meter.sh}

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

STAGE1_NORM_RUN_DIR=${STAGE1_NORM_RUN_DIR:-}
STAGE1_NORM_CKPT=${STAGE1_NORM_CKPT:-}
OUT_DIR=${OUT_DIR:-}
RUN_NAME=${RUN_NAME:-}

RUNTIME_LOG_DIR="$CV_ROOT/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/D049_n2m_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.out" \
     2>"$RUNTIME_LOG_DIR/D049_n2m_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.err"

latest_normalized_run_with_ckpt() {
  local parent="$1"
  if [[ ! -d "$parent" ]]; then
    echo ""
    return 0
  fi

  # Prefer the formal normalized-loss run whose name records ES-val_loss.
  find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth \
    -printf '%T@ %h\n' 2>/dev/null \
    | awk '$2 ~ /ES-val_loss/ {print}' \
    | sort -nr \
    | awk 'NR==1{print $2}'
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
    echo "[ERROR] D049 formal submission currently supports CA, CO, and Santiam. Got $HOLDOUT_PRESET" >&2
    exit 2
    ;;
esac

SAFE_PRESET=$(echo "$HOLDOUT_PRESET" | sed 's/[^A-Za-z0-9_]/_/g')
SPLIT_DIR=${SPLIT_DIR:-$SOURCE_CV_ROOT/splits/holdout_${SAFE_PRESET}_${SOURCE_SPLIT_TAG}}

STAGE1_NORM_PARENT="$SOURCE_CV_ROOT/runs/holdout_${SAFE_PRESET}_${SOURCE_SPLIT_TAG}"
if [[ -z "$STAGE1_NORM_RUN_DIR" ]]; then
  STAGE1_NORM_RUN_DIR=$(latest_normalized_run_with_ckpt "$STAGE1_NORM_PARENT")
fi
if [[ -z "$STAGE1_NORM_RUN_DIR" ]]; then
  echo "[ERROR] Could not find normalized-loss ES-val_loss run under:" >&2
  echo "        $STAGE1_NORM_PARENT" >&2
  exit 2
fi
STAGE1_NORM_CKPT=${STAGE1_NORM_CKPT:-$STAGE1_NORM_RUN_DIR/checkpoint-best.pth}

RUN_NAME=${RUN_NAME:-train_stage2_meterMAE_fromNormBest_${HOLDOUT_NAME}_e${EPOCHS}_lr${LR}_b${BATCH_SIZE}_acc${ACCUM_ITER}}
OUT_DIR=${OUT_DIR:-$CV_ROOT/runs/holdout_${SAFE_PRESET}_${RUN_TAG}/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

for f in \
  "$TRAIN_BACKEND" "$STAGE1_NORM_CKPT" \
  "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/val_tiles.txt" \
  "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/val_hidden.txt" \
  "$SPLIT_DIR/train_loss.txt" "$SPLIT_DIR/val_loss.txt"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing required file: $f" >&2; exit 2; }
done

echo "============================================================"
echo "D049 holdout NormThenMeter: D001c AnyVisiblePatch RELAX project"
date
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "HOLDOUT_NAME=$HOLDOUT_NAME"
echo "SOURCE_SPLIT=$SPLIT_DIR"
echo "STAGE1_NORM_RUN_DIR=$STAGE1_NORM_RUN_DIR"
echo "STAGE1_NORM_CKPT=$STAGE1_NORM_CKPT"
echo "OUT_DIR=$OUT_DIR"
echo "OPTIMIZATION_LOSS=meter_mae"
echo "BEST_METRIC=val_mae_m_mask"
echo "EARLY_STOP_METRIC=val_mae_m_mask"
echo "BASELINE=normalized-loss checkpoint evaluated at epoch -1"
echo "LR=$LR"
echo "EPOCHS=$EPOCHS"
echo "PATIENCE=$PATIENCE"
echo "============================================================"

SPLIT_DIR="$SPLIT_DIR" \
TILE_ROOT="$TILE_ROOT" \
STAGE1_NORM_CKPT="$STAGE1_NORM_CKPT" \
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

echo "=== DONE D049 ==="
echo "$OUT_DIR"
date
