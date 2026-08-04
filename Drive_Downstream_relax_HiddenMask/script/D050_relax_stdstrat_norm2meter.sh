#!/usr/bin/env bash
#SBATCH -J D050_n2m_std_AP
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask

set -euo pipefail

# RELAX PROJECT: isolated code/results under Downstream_Task_Bathy_relax_HiddenMask.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
WORK=${WORK:-$RELAX_ROOT}
RESULTS_ROOT=${RESULTS_ROOT:-$RELAX_ROOT/results}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2_D001c_AnyVisiblePatch/Tiles_1m}

SOURCE_CV_ROOT=${SOURCE_CV_ROOT:-$RESULTS_ROOT/NormOnly}
CV_ROOT=${CV_ROOT:-$RESULTS_ROOT/NormThenMeter}

SOURCE_SPLIT_NAME=${SOURCE_SPLIT_NAME:-stdStratRiver_manualVal_CO_Nisqually_NE_seed42_D001cAnyVisiblePatch_D001NoDataSafe}
RUN_TAG=${RUN_TAG:-D048NormThenMeter_D001cAnyVisiblePatch_D001NoDataSafe}
TRAIN_BACKEND=${TRAIN_BACKEND:-$RELAX_ROOT/script/D048_relax_train_norm2meter.sh}

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
STAGE1_NORM_RUN_DIR=${STAGE1_NORM_RUN_DIR:-}
STAGE1_NORM_CKPT=${STAGE1_NORM_CKPT:-}

RUNTIME_LOG_DIR="$CV_ROOT/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/D050_n2m_std_${RUNTIME_JOB_ID}.out" \
     2>"$RUNTIME_LOG_DIR/D050_n2m_std_${RUNTIME_JOB_ID}.err"

latest_normalized_run_with_ckpt() {
  local parent="$1"
  if [[ ! -d "$parent" ]]; then
    echo ""
    return 0
  fi
  find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth \
    -printf '%T@ %h\n' 2>/dev/null \
    | awk '$2 ~ /ES-val_loss/ {print}' \
    | sort -nr \
    | awk 'NR==1{print $2}'
}

STAGE1_NORM_PARENT="$SOURCE_CV_ROOT/runs/$SOURCE_SPLIT_NAME"
if [[ -z "$STAGE1_NORM_RUN_DIR" ]]; then
  STAGE1_NORM_RUN_DIR=$(latest_normalized_run_with_ckpt "$STAGE1_NORM_PARENT")
fi
if [[ -z "$STAGE1_NORM_RUN_DIR" ]]; then
  echo "[ERROR] No normalized-loss ES-val_loss std-strat checkpoint under:" >&2
  echo "        $STAGE1_NORM_PARENT" >&2
  exit 2
fi
STAGE1_NORM_CKPT=${STAGE1_NORM_CKPT:-$STAGE1_NORM_RUN_DIR/checkpoint-best.pth}

RUN_NAME=${RUN_NAME:-train_stage2_meterMAE_fromNormBest_${SOURCE_SPLIT_NAME}_e${EPOCHS}_lr${LR}}
OUT_DIR=${OUT_DIR:-$CV_ROOT/runs/$SOURCE_SPLIT_NAME/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

for f in \
  "$TRAIN_BACKEND" "$STAGE1_NORM_CKPT" \
  "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/val_tiles.txt" \
  "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/val_hidden.txt" \
  "$SPLIT_DIR/train_loss.txt" "$SPLIT_DIR/val_loss.txt"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing required file: $f" >&2; exit 2; }
done

echo "============================================================"
echo "D050 std-strat NormThenMeter: D001c AnyVisiblePatch RELAX project"
echo "SOURCE_SPLIT=$SPLIT_DIR"
echo "STAGE1_NORM_RUN_DIR=$STAGE1_NORM_RUN_DIR"
echo "STAGE1_NORM_CKPT=$STAGE1_NORM_CKPT"
echo "OUT_DIR=$OUT_DIR"
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
WARMUP_EPOCHS=0 \
EARLY_STOP_WARMUP_EPOCHS=0 \
EARLY_STOP_MIN_DELTA=0.001 \
FRESH_RUN="$FRESH_RUN" \
OVERWRITE_STAGE2="$OVERWRITE_STAGE2" \
bash "$TRAIN_BACKEND"

echo "=== DONE D050 ==="
echo "$OUT_DIR"
