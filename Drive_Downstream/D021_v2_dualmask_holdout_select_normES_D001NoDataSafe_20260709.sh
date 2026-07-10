#!/usr/bin/env bash
#SBATCH -J d021_holdout_NDsafe
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 7-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/D021_holdout_NDsafe_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/D021_holdout_NDsafe_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# D021 D001NoDataSafe: select one whole-river holdout fold and train it with normalized early stopping.
#
# Submit examples:
#   sbatch -p gpu -w execute-3000 --export=ALL,HOLDOUT_PRESET=CO,GPU_ID=0 D021_v2_dualmask_holdout_select_normES_sbatch_20260708.sh
#   sbatch -p gpu -w execute-3000 --export=ALL,HOLDOUT_PRESET=CA,GPU_ID=1 D021_v2_dualmask_holdout_select_normES_sbatch_20260708.sh
#   sbatch -p HydroIntel -w execute-4006 --export=ALL,HOLDOUT_PRESET=Santiam,GPU_ID=0 D021_v2_dualmask_holdout_select_normES_sbatch_20260708.sh
#
# Manual river override:
#   sbatch ... --export=ALL,HOLDOUT_NAME=CO_UpperColorado_Topobathy_1_2020,HOLDOUT_RIVERS=CO_UpperColorado_Topobathy_1_2020 D021...

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

SPLIT_SCRIPT=${SPLIT_SCRIPT:-$WORK/script/A016_v2_holdout_split_20260707_final.py}
TRAIN_BACKEND=${TRAIN_BACKEND:-$WORK/script/D020_v2_dualmask_coreloss_train_backend_normES_20260708.sh}

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

BEST_METRIC=${BEST_METRIC:-val_loss}
EARLY_STOP_METRIC=${EARLY_STOP_METRIC:-$BEST_METRIC}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.0001}
EARLY_STOP_WARMUP_EPOCHS=${EARLY_STOP_WARMUP_EPOCHS:-20}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

cd "$CODE"

if [[ -z "$HOLDOUT_NAME" || -z "$HOLDOUT_RIVERS" ]]; then
  case "$HOLDOUT_PRESET" in
    CO)
      HOLDOUT_NAME="CO_UpperColorado_Topobathy_1_2020"
      HOLDOUT_RIVERS="CO_UpperColorado_Topobathy_1_2020"
      ;;
    CA)
      HOLDOUT_NAME="CA_KlamathRiver_TopoBathy_2018_D18"
      HOLDOUT_RIVERS="CA_KlamathRiver_TopoBathy_2018_D18"
      ;;
    Santiam)
      HOLDOUT_NAME="OR_SantiamRiverTB_Topobathy_1_D23"
      HOLDOUT_RIVERS="OR_SantiamRiverTB_Topobathy_1_D23"
      ;;
    NE)
      HOLDOUT_NAME="NE_Niobrara_Topobathy_2018"
      HOLDOUT_RIVERS="NE_Niobrara_Topobathy_2018"
      ;;
    OR_MKRC)
      HOLDOUT_NAME="OR_MKRC_Topobathy_2021"
      HOLDOUT_RIVERS="OR_MKRC_Topobathy_2021"
      ;;
    Nisqually)
      HOLDOUT_NAME="WA_Nisqually_Bathymetric_2020"
      HOLDOUT_RIVERS="WA_Nisqually_Bathymetric_2020"
      ;;
    MD)
      HOLDOUT_NAME="MD_PotomacRiver_Bathy_2019"
      HOLDOUT_RIVERS="MD_PotomacRiver_Bathy_2019"
      ;;
    Chehalis)
      HOLDOUT_NAME="WA_ChehalisRiverTB_Topobathy_1_D23"
      HOLDOUT_RIVERS="WA_ChehalisRiverTB_Topobathy_1_D23"
      ;;
    BadgerFinNull|Estabrook_Combined|KewaFix2Null|Kletzch_Combined_UpMax3Null)
      HOLDOUT_NAME="$HOLDOUT_PRESET"
      HOLDOUT_RIVERS="$HOLDOUT_PRESET"
      ;;
    MilwaukeeGroup)
      HOLDOUT_NAME="MilwaukeeRiverGroup"
      HOLDOUT_RIVERS="BadgerFinNull Estabrook_Combined KewaFix2Null Kletzch_Combined_UpMax3Null"
      ;;
    *)
      echo "[ERROR] Unknown HOLDOUT_PRESET=$HOLDOUT_PRESET" >&2
      echo "Allowed: CO, CA, Santiam, NE, OR_MKRC, Nisqually, MD, Chehalis, BadgerFinNull, Estabrook_Combined, KewaFix2Null, Kletzch_Combined_UpMax3Null, MilwaukeeGroup" >&2
      exit 2
      ;;
  esac
fi

if [[ -z "${SPLIT_DIR:-}" ]]; then
  if [[ -n "$DATA_TAG" ]]; then
    SAFE_HOLDOUT_PRESET=$(echo "$HOLDOUT_PRESET" | sed 's/[^A-Za-z0-9_]/_/g')
    SPLIT_DIR="$CV_ROOT/splits/holdout_${SAFE_HOLDOUT_PRESET}${DATA_SUFFIX}"
  else
    SPLIT_DIR="$CV_ROOT/splits/holdout_$HOLDOUT_NAME"
  fi
fi

if [[ -z "${RUN_STAGE:-}" ]]; then
  if [[ "$EPOCHS" -le 5 ]]; then
    RUN_STAGE="smoke_holdout${DATA_SUFFIX}"
  else
    RUN_STAGE="train_holdout${DATA_SUFFIX}"
  fi
fi

METRIC_TAG=$(echo "$EARLY_STOP_METRIC" | sed 's/[^A-Za-z0-9_]/_/g')
RUN_NAME=${RUN_NAME:-${RUN_STAGE}_${HOLDOUT_NAME}_v2_dualmask_corePixelLoss_ES-${METRIC_TAG}_e${EPOCHS}_b${BATCH_SIZE}_acc${ACCUM_ITER}}
if [[ -z "${OUT_DIR:-}" ]]; then
  if [[ -n "$DATA_TAG" ]]; then
    SAFE_HOLDOUT_PRESET=$(echo "$HOLDOUT_PRESET" | sed 's/[^A-Za-z0-9_]/_/g')
    OUT_DIR="$CV_ROOT/runs/holdout_${SAFE_HOLDOUT_PRESET}${DATA_SUFFIX}/$RUN_NAME"
  else
    OUT_DIR="$CV_ROOT/runs/holdout_$HOLDOUT_NAME/$RUN_NAME"
  fi
fi
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

mkdir -p "$CV_ROOT/logs"

echo "=== D021 selected holdout training, normalized early stop ==="
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOST=$(hostname)"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "HOLDOUT_NAME=$HOLDOUT_NAME"
echo "HOLDOUT_RIVERS=$HOLDOUT_RIVERS"
echo "DATA_TAG=$DATA_TAG"
echo "DATA_SUFFIX=$DATA_SUFFIX"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "OUT_DIR=$OUT_DIR"
echo "GPU_ID=$GPU_ID"
echo "BEST_METRIC=$BEST_METRIC"
echo "EARLY_STOP_METRIC=$EARLY_STOP_METRIC"
echo "EARLY_STOP_MIN_DELTA=$EARLY_STOP_MIN_DELTA"
date
nvidia-smi || true

python "$SPLIT_SCRIPT" \
  --holdout_name "$HOLDOUT_NAME" \
  --holdout_rivers $HOLDOUT_RIVERS \
  --tile_root "$TILE_ROOT" \
  --out_dir "$SPLIT_DIR"

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
