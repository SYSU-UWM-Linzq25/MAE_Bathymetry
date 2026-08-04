#!/usr/bin/env bash
#SBATCH -J F071_meter_defreeze
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

# F071: whole-river holdout driver for MeterOnly defreeze-last-1.
# Reuses the exact D001c split and starts from the matching formal MeterOnly best.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
RESULTS_ROOT=${RESULTS_ROOT:-$RELAX_ROOT/results}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2_D001c_AnyVisiblePatch/Tiles_1m}

SPLIT_ROOT=${SPLIT_ROOT:-$RESULTS_ROOT/NormOnly}
SOURCE_METER_ROOT=${SOURCE_METER_ROOT:-$RESULTS_ROOT/MeterOnly}
CV_ROOT=${CV_ROOT:-$RESULTS_ROOT/MeterOnly_DefreezeLast1}

SOURCE_SPLIT_TAG=${SOURCE_SPLIT_TAG:-D001cAnyVisiblePatch_D001NoDataSafe}
SOURCE_METER_RUN_TAG=${SOURCE_METER_RUN_TAG:-D044MeterOnly_D001cAnyVisiblePatch_D001NoDataSafe}
RUN_TAG=${RUN_TAG:-F070MeterOnlyDefreezeLast1_D001cAnyVisiblePatch_D001NoDataSafe}
TRAIN_BACKEND=${TRAIN_BACKEND:-$RELAX_ROOT/script/F070_relax_meterOnly_defreeze_train.sh}

HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}
HOLDOUT_NAME=${HOLDOUT_NAME:-}

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

EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.001}
EARLY_STOP_WARMUP_EPOCHS=${EARLY_STOP_WARMUP_EPOCHS:-0}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}

SOURCE_METER_RUN_DIR=${SOURCE_METER_RUN_DIR:-}
INIT_CKPT=${INIT_CKPT:-}
OUT_DIR=${OUT_DIR:-}
RUN_NAME=${RUN_NAME:-}

RUNTIME_LOG_DIR="$CV_ROOT/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/F071_defreeze_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.out" \
     2>"$RUNTIME_LOG_DIR/F071_defreeze_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.err"

latest_formal_run_with_ckpt() {
  local parent="$1"
  local hit=""
  [[ -d "$parent" ]] || { echo ""; return 0; }

  hit=$(find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth \
    -printf '%T@ %h\n' 2>/dev/null \
    | awk '$2 ~ /train_holdout_/ && $2 !~ /smoke/' \
    | sort -nr | awk 'NR==1{print $2}')

  if [[ -z "$hit" ]]; then
    hit=$(find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth \
      -printf '%T@ %h\n' 2>/dev/null \
      | sort -nr | awk 'NR==1{print $2}')
  fi
  echo "$hit"
}

case "$HOLDOUT_PRESET" in
  CA)      HOLDOUT_NAME=${HOLDOUT_NAME:-CA_KlamathRiver_TopoBathy_2018_D18} ;;
  CO)      HOLDOUT_NAME=${HOLDOUT_NAME:-CO_UpperColorado_Topobathy_1_2020} ;;
  Santiam) HOLDOUT_NAME=${HOLDOUT_NAME:-OR_SantiamRiverTB_Topobathy_1_D23} ;;
  *)
    echo "[ERROR] F071 formal experiment supports CA, CO, Santiam. Got $HOLDOUT_PRESET" >&2
    exit 2
    ;;
esac

SAFE_PRESET=$(echo "$HOLDOUT_PRESET" | sed 's/[^A-Za-z0-9_]/_/g')
SPLIT_DIR=${SPLIT_DIR:-$SPLIT_ROOT/splits/holdout_${SAFE_PRESET}_${SOURCE_SPLIT_TAG}}

SOURCE_PARENT="$SOURCE_METER_ROOT/runs/holdout_${SAFE_PRESET}_${SOURCE_METER_RUN_TAG}"
if [[ -z "$SOURCE_METER_RUN_DIR" ]]; then
  SOURCE_METER_RUN_DIR=$(latest_formal_run_with_ckpt "$SOURCE_PARENT")
fi
if [[ -z "$SOURCE_METER_RUN_DIR" ]]; then
  echo "[ERROR] No formal MeterOnly run found under:" >&2
  echo "        $SOURCE_PARENT" >&2
  exit 2
fi
INIT_CKPT=${INIT_CKPT:-$SOURCE_METER_RUN_DIR/checkpoint-best.pth}

RUN_NAME=${RUN_NAME:-train_holdout_${RUN_TAG}_${HOLDOUT_NAME}_fromMeterBest_e${EPOCHS}_lr${LR}_b${BATCH_SIZE}_acc${ACCUM_ITER}}
OUT_DIR=${OUT_DIR:-$CV_ROOT/runs/holdout_${SAFE_PRESET}_${RUN_TAG}/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

for f in \
  "$TRAIN_BACKEND" "$INIT_CKPT" \
  "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/val_tiles.txt" \
  "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/val_hidden.txt" \
  "$SPLIT_DIR/train_loss.txt" "$SPLIT_DIR/val_loss.txt"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing required file: $f" >&2; exit 2; }
done

echo "============================================================"
echo "F071 MeterOnly defreeze-last-1 holdout"
date
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "HOLDOUT_NAME=$HOLDOUT_NAME"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "SOURCE_METER_RUN_DIR=$SOURCE_METER_RUN_DIR"
echo "INIT_CKPT=$INIT_CKPT"
echo "OUT_DIR=$OUT_DIR"
echo "TRAINABLE_LAST_N_ENCODER_BLOCKS=1"
echo "OPTIMIZATION_LOSS=meter_mae"
echo "BEST/ES=val_mae_m_mask"
echo "BASELINE=untouched frozen MeterOnly at epoch -1"
echo "LR=$LR"
echo "EPOCHS=$EPOCHS"
echo "PATIENCE=$PATIENCE"
echo "============================================================"

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
EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
EARLY_STOP_WARMUP_EPOCHS="$EARLY_STOP_WARMUP_EPOCHS" \
BASELINE_EVAL_BEFORE_TRAINING=1 \
WARMUP_EPOCHS="$WARMUP_EPOCHS" \
FRESH_RUN="$FRESH_RUN" \
OVERWRITE_DEFREEZE="$OVERWRITE_DEFREEZE" \
bash "$TRAIN_BACKEND"

echo "============================================================"
echo "DONE F071"
echo "OUT_DIR=$OUT_DIR"
date
echo "============================================================"
