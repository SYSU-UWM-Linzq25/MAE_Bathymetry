#!/usr/bin/env bash
#SBATCH -J D031_holdout_meterMAE
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 7-00:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Slurm opens #SBATCH -o/-e before executing this script.  Because the new
# cross_validation_v4_meterMAE_BaselineEval directory may not exist on the first run, this
# revision intentionally uses Slurm's temporary/default bootstrap output first,
# creates the target log directory at runtime, and then redirects all subsequent
# stdout/stderr into the isolated v3 log directory.
RUNTIME_LOG_DIR=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v4_meterMAE_BaselineEval/logs
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
RUNTIME_OUT="$RUNTIME_LOG_DIR/D031_holdout_meterMAE_${RUNTIME_JOB_ID}.out"
RUNTIME_ERR="$RUNTIME_LOG_DIR/D031_holdout_meterMAE_${RUNTIME_JOB_ID}.err"
exec >"$RUNTIME_OUT" 2>"$RUNTIME_ERR"

echo "[RUNTIME-LOG] stdout=$RUNTIME_OUT"
echo "[RUNTIME-LOG] stderr=$RUNTIME_ERR"

# D031: formal whole-river holdout meter-MAE training with validation baseline before any optimizer update.
#
# Important:
#   * Reuses the exact D001NoDataSafe split files from cross_validation_v2.
#   * Writes all checkpoints/logs to cross_validation_v4_meterMAE_BaselineEval.
#   * Does not overwrite normalized-MSE experiments.
#   * Optimization objective:
#       pixel-weighted |prediction_m - GT_m|
#     on Loss_Mask_Pixel AND core/prediction patch mask AND valid patch.
#
# Examples:
#   sbatch --export=ALL,HOLDOUT_PRESET=CA,GPU_ID=0 D031_v2_dualmask_holdout_meterMAE_BaselineEval_RuntimeLogSafe_D001NoDataSafe_20260713.sh
#   sbatch --export=ALL,HOLDOUT_PRESET=CO,GPU_ID=0 D031_v2_dualmask_holdout_meterMAE_BaselineEval_RuntimeLogSafe_D001NoDataSafe_20260713.sh
#   sbatch --export=ALL,HOLDOUT_PRESET=Santiam,GPU_ID=0 D031_v2_dualmask_holdout_meterMAE_BaselineEval_RuntimeLogSafe_D001NoDataSafe_20260713.sh

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
CODE=${CODE:-$ROOT/mae_Retrain}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2/Tiles_1m}

SOURCE_CV_ROOT=${SOURCE_CV_ROOT:-$WORK/cross_validation_v2}
CV_ROOT=${CV_ROOT:-$WORK/cross_validation_v4_meterMAE_BaselineEval}
SOURCE_SPLIT_TAG=${SOURCE_SPLIT_TAG:-D001NoDataSafe}
RUN_TAG=${RUN_TAG:-D003MeterMAE_BaselineEval_D001NoDataSafe}

TRAIN_BACKEND=${TRAIN_BACKEND:-$WORK/script/D030_train_v2_dualmask_corePixel_meterMAE_BaselineEval_backend_20260713.sh}

HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}
HOLDOUT_NAME=${HOLDOUT_NAME:-}
HOLDOUT_RIVERS=${HOLDOUT_RIVERS:-}

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

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

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
  NE)
    HOLDOUT_NAME=${HOLDOUT_NAME:-NE_Niobrara_Topobathy_2018}
    HOLDOUT_RIVERS=${HOLDOUT_RIVERS:-NE_Niobrara_Topobathy_2018}
    ;;
  OR_MKRC)
    HOLDOUT_NAME=${HOLDOUT_NAME:-OR_MKRC_Topobathy_2021}
    HOLDOUT_RIVERS=${HOLDOUT_RIVERS:-OR_MKRC_Topobathy_2021}
    ;;
  Nisqually)
    HOLDOUT_NAME=${HOLDOUT_NAME:-WA_Nisqually_Bathymetric_2020}
    HOLDOUT_RIVERS=${HOLDOUT_RIVERS:-WA_Nisqually_Bathymetric_2020}
    ;;
  MD)
    HOLDOUT_NAME=${HOLDOUT_NAME:-MD_PotomacRiver_Bathy_2019}
    HOLDOUT_RIVERS=${HOLDOUT_RIVERS:-MD_PotomacRiver_Bathy_2019}
    ;;
  Chehalis)
    HOLDOUT_NAME=${HOLDOUT_NAME:-WA_ChehalisRiverTB_Topobathy_1_D23}
    HOLDOUT_RIVERS=${HOLDOUT_RIVERS:-WA_ChehalisRiverTB_Topobathy_1_D23}
    ;;
  MilwaukeeGroup)
    HOLDOUT_NAME=${HOLDOUT_NAME:-MilwaukeeRiverGroup}
    HOLDOUT_RIVERS=${HOLDOUT_RIVERS:-"BadgerFinNull Estabrook_Combined KewaFix2Null Kletzch_Combined_UpMax3Null"}
    ;;
  *)
    echo "[ERROR] Unknown HOLDOUT_PRESET=$HOLDOUT_PRESET" >&2
    exit 2
    ;;
esac

SAFE_PRESET=$(echo "$HOLDOUT_PRESET" | sed 's/[^A-Za-z0-9_]/_/g')
SPLIT_DIR=${SPLIT_DIR:-$SOURCE_CV_ROOT/splits/holdout_${SAFE_PRESET}_${SOURCE_SPLIT_TAG}}

if [[ "$EPOCHS" -le 5 ]]; then
  RUN_STAGE=${RUN_STAGE:-smoke_holdout_${RUN_TAG}}
else
  RUN_STAGE=${RUN_STAGE:-train_holdout_${RUN_TAG}}
fi

METRIC_TAG=$(echo "$EARLY_STOP_METRIC" | sed 's/[^A-Za-z0-9_]/_/g')
RUN_NAME=${RUN_NAME:-${RUN_STAGE}_${HOLDOUT_NAME}_v2_dualmask_meterMAE_BaselineEval_corePixel_Best-${METRIC_TAG}_e${EPOCHS}_b${BATCH_SIZE}_acc${ACCUM_ITER}}
OUT_DIR=${OUT_DIR:-$CV_ROOT/runs/holdout_${SAFE_PRESET}_${RUN_TAG}/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

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
echo "D031 holdout meter-MAE training with epoch -1 baseline"
date
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOST=$(hostname)"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "HOLDOUT_NAME=$HOLDOUT_NAME"
echo "HOLDOUT_RIVERS=$HOLDOUT_RIVERS"
echo "SOURCE_SPLIT=$SPLIT_DIR"
echo "SOURCE_SPLIT_TAG=$SOURCE_SPLIT_TAG"
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
echo "NOTE=Same D001NoDataSafe split as normalized-MSE experiment"
echo "NOTE=All new outputs are isolated under cross_validation_v4_meterMAE_BaselineEval"
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
echo "DONE D031"
echo "OUT_DIR=$OUT_DIR"
date
echo "============================================================"
