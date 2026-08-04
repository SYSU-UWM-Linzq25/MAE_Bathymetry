#!/usr/bin/env bash
#SBATCH -J D046_meter_std_AP
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 7-00:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# RELAX PROJECT: isolated code/results under Downstream_Task_Bathy_relax_HiddenMask.

# Runtime logs are redirected after CV_ROOT is defined.

# D046: standard-stratified/manual-validation training with meter-space MAE.
# Reuses the exact D001c std-strat split from the NormOnly CV root and writes
# results to the isolated D001c MeterOnly CV root.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
WORK=${WORK:-$RELAX_ROOT}
RESULTS_ROOT=${RESULTS_ROOT:-$RELAX_ROOT/results}
CODE=${CODE:-$ROOT/mae_Retrain}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2_D001c_AnyVisiblePatch/Tiles_1m}

SOURCE_CV_ROOT=${SOURCE_CV_ROOT:-$RESULTS_ROOT/NormOnly}
CV_ROOT=${CV_ROOT:-$RESULTS_ROOT/MeterOnly}
SOURCE_SPLIT_NAME=${SOURCE_SPLIT_NAME:-stdStratRiver_manualVal_CO_Nisqually_NE_seed42_D001cAnyVisiblePatch_D001NoDataSafe}
RUN_TAG=${RUN_TAG:-D044MeterOnly_D001cAnyVisiblePatch_D001NoDataSafe}
SPLIT_DIR=${SPLIT_DIR:-$SOURCE_CV_ROOT/splits/$SOURCE_SPLIT_NAME}

TRAIN_BACKEND=${TRAIN_BACKEND:-$RELAX_ROOT/script/D044_relax_train_meter.sh}

RUNTIME_LOG_DIR="$CV_ROOT/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/D046_meter_std_${RUNTIME_JOB_ID}.out" 2>"$RUNTIME_LOG_DIR/D046_meter_std_${RUNTIME_JOB_ID}.err"

GPU_ID=${GPU_ID:-0}
EPOCHS=${EPOCHS:-400}
BATCH_SIZE=${BATCH_SIZE:-4}
ACCUM_ITER=${ACCUM_ITER:-4}
NUM_WORKERS=${NUM_WORKERS:-1}
PATIENCE=${PATIENCE:-60}
FRESH_RUN=${FRESH_RUN:-1}

OPTIMIZATION_LOSS=${OPTIMIZATION_LOSS:-meter_mae}
BEST_METRIC=${BEST_METRIC:-val_mae_m_mask}
EARLY_STOP_METRIC=${EARLY_STOP_METRIC:-$BEST_METRIC}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.001}
EARLY_STOP_WARMUP_EPOCHS=${EARLY_STOP_WARMUP_EPOCHS:-20}
BASELINE_EVAL_BEFORE_TRAINING=${BASELINE_EVAL_BEFORE_TRAINING:-1}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}

if [[ "$EPOCHS" -le 5 ]]; then
  RUN_STAGE=${RUN_STAGE:-smoke_stdstrat_${RUN_TAG}}
else
  RUN_STAGE=${RUN_STAGE:-train_stdstrat_${RUN_TAG}}
fi

METRIC_TAG=$(echo "$EARLY_STOP_METRIC" | sed 's/[^A-Za-z0-9_]/_/g')
RUN_NAME=${RUN_NAME:-${RUN_STAGE}_${SOURCE_SPLIT_NAME}_v2_dualmask_meterMAE_BaselineEval_corePixel_Best-${METRIC_TAG}_e${EPOCHS}_b${BATCH_SIZE}_acc${ACCUM_ITER}}
OUT_DIR=${OUT_DIR:-$CV_ROOT/runs/$SOURCE_SPLIT_NAME/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

mkdir -p "$CV_ROOT/logs"

for f in \
  "$TRAIN_BACKEND" \
  "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/val_tiles.txt" \
  "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/val_hidden.txt" \
  "$SPLIT_DIR/train_loss.txt" "$SPLIT_DIR/val_loss.txt"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required existing split/backend file: $f" >&2
    exit 2
  fi
done

echo "============================================================"
echo "D046 std-stratified MeterOnly: D001c AnyVisiblePatch RELAX project"
date
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOST=$(hostname)"
echo "SOURCE_SPLIT_NAME=$SOURCE_SPLIT_NAME"
echo "SOURCE_SPLIT=$SPLIT_DIR"
echo "RUN_TAG=$RUN_TAG"
echo "OUT_DIR=$OUT_DIR"
echo "OPTIMIZATION_LOSS=$OPTIMIZATION_LOSS"
echo "BEST_METRIC=$BEST_METRIC"
echo "EARLY_STOP_METRIC=$EARLY_STOP_METRIC"
echo "EARLY_STOP_MIN_DELTA=$EARLY_STOP_MIN_DELTA m"
echo "BASELINE_EVAL_BEFORE_TRAINING=$BASELINE_EVAL_BEFORE_TRAINING"
echo "BASELINE=validation before optimizer update, recorded as epoch -1"
echo "BEST_CHECKPOINT_RULE=baseline remains checkpoint-best unless a training epoch improves $BEST_METRIC"
echo "WARMUP_EPOCHS=$WARMUP_EPOCHS"
echo "NOTE=Same D001c AnyVisiblePatch RELAX project split as NormOnly experiment"
echo "NOTE=All outputs are isolated under results/MeterOnly"
echo "============================================================"
nvidia-smi || true

SPLIT_DIR="$SPLIT_DIR" \
TILE_ROOT="$TILE_ROOT" \
RUN_NAME="$RUN_NAME" \
OUT_DIR="$OUT_DIR" \
LOG_DIR="$LOG_DIR" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
ACCUM_ITER="$ACCUM_ITER" \
GPU_ID="$GPU_ID" \
NUM_WORKERS="$NUM_WORKERS" \
PATIENCE="$PATIENCE" \
OPTIMIZATION_LOSS="$OPTIMIZATION_LOSS" \
BEST_METRIC="$BEST_METRIC" \
EARLY_STOP_METRIC="$EARLY_STOP_METRIC" \
EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
EARLY_STOP_WARMUP_EPOCHS="$EARLY_STOP_WARMUP_EPOCHS" \
BASELINE_EVAL_BEFORE_TRAINING="$BASELINE_EVAL_BEFORE_TRAINING" \
WARMUP_EPOCHS="$WARMUP_EPOCHS" \
FRESH_RUN="$FRESH_RUN" \
bash "$TRAIN_BACKEND"

echo "============================================================"
echo "DONE D046"
echo "OUT_DIR=$OUT_DIR"
date
echo "============================================================"
