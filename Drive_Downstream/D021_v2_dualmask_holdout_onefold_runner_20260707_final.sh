#!/usr/bin/env bash
set -euo pipefail

# D021 v2 final: run one MAE v2 dual-mask holdout fold.
# This script does NOT randomly split tiles into train/val.
#
# It calls:
#   A016_v2_holdout_split_20260707_final.py
#   D020_v2_dualmask_coreloss_train_backend_20260707_final.sh

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
CODE=${CODE:-$ROOT/mae_Retrain}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2/Tiles_1m}

# Recommended first holdout if only one fold can be run.
HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}

# Optional manual override. If set, these override the preset.
HOLDOUT_NAME=${HOLDOUT_NAME:-}
HOLDOUT_RIVERS=${HOLDOUT_RIVERS:-}

CV_ROOT=${CV_ROOT:-$WORK/cross_validation_v2}
GPU_ID=${GPU_ID:-0}
EPOCHS=${EPOCHS:-400}
BATCH_SIZE=${BATCH_SIZE:-4}
ACCUM_ITER=${ACCUM_ITER:-4}
NUM_WORKERS=${NUM_WORKERS:-1}
PATIENCE=${PATIENCE:-60}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

cd "$CODE"

# Resolve preset into name/rivers if manual override is not provided.
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
    MilwaukeeGroup)
      HOLDOUT_NAME="MilwaukeeRiverGroup"
      HOLDOUT_RIVERS="BadgerFinNull Estabrook_Combined KewaFix2Null Kletzch_Combined_UpMax3Null"
      ;;
    *)
      echo "[ERROR] Unknown HOLDOUT_PRESET=$HOLDOUT_PRESET" >&2
      echo "Allowed: CO, CA, Santiam, NE, OR_MKRC, Nisqually, MD, Chehalis, MilwaukeeGroup" >&2
      exit 2
      ;;
  esac
fi

SPLIT_DIR=${SPLIT_DIR:-$CV_ROOT/splits/holdout_$HOLDOUT_NAME}

# Explicit, readable run stage.
# If the user does not set RUN_STAGE, short runs are automatically labeled smoke.
if [[ -z "${RUN_STAGE:-}" ]]; then
  if [[ "$EPOCHS" -le 5 ]]; then
    RUN_STAGE="smoke"
  else
    RUN_STAGE="train"
  fi
fi

RUN_NAME=${RUN_NAME:-${RUN_STAGE}_holdout_${HOLDOUT_NAME}_v2_dualmask_corePixelLoss_e${EPOCHS}_b${BATCH_SIZE}_acc${ACCUM_ITER}}
OUT_DIR=${OUT_DIR:-$CV_ROOT/runs/holdout_$HOLDOUT_NAME/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

echo "=== D021 holdout v2 dual-mask one fold ==="
echo "RUN_STAGE=$RUN_STAGE"
echo "RUN_NAME=$RUN_NAME"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "HOLDOUT_NAME=$HOLDOUT_NAME"
echo "HOLDOUT_RIVERS=$HOLDOUT_RIVERS"
echo "TILE_ROOT=$TILE_ROOT"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "GPU_ID=$GPU_ID"
echo "EPOCHS=$EPOCHS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "ACCUM_ITER=$ACCUM_ITER"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "PATIENCE=$PATIENCE"

python "$WORK/script/A016_v2_holdout_split_20260707_final.py" \
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
bash "$WORK/script/D020_v2_dualmask_coreloss_train_backend_20260707_final.sh"
