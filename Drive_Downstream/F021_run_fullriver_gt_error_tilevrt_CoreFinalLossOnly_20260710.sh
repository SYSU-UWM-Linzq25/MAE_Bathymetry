#!/usr/bin/env bash
#SBATCH -J f021_gt_error_vrt
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/F021_gt_error_vrt_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/F021_gt_error_vrt_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
SCRIPT=${SCRIPT:-$WORK/script/F020_fullriver_gt_error_tilevrt_CoreFinalLossOnly_20260710.py}
CV_ROOT=${CV_ROOT:-$WORK/cross_validation_v2}

F010_BASE=${F010_BASE:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Predictions_F010_TileAvgVRT_D001NoDataSafe}
OUT_BASE=${OUT_BASE:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_GT_Error_F020_TileVRT_D001NoDataSafe}

MODE=${MODE:-holdout}
DATA_TAG=${DATA_TAG:-D001NoDataSafe}
HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}
STD_SPLIT_SEED=${STD_SPLIT_SEED:-42}
MANUAL_VAL_TAG=${MANUAL_VAL_TAG:-CO_Nisqually_NE}
BIN_STAT=${BIN_STAT:-median}
VAL_PER_BIN=${VAL_PER_BIN:-1}
VAL_RIVERS=${VAL_RIVERS:-CO_UpperColorado_Topobathy_1_2020 WA_Nisqually_Bathymetric_2020 NE_Niobrara_Topobathy_2018}
RIVERS=${RIVERS:-}

OVERWRITE=${OVERWRITE:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}

F010_OUT_DIR=${F010_OUT_DIR:-}
OUT_DIR=${OUT_DIR:-}

safe_name() {
  echo "$1" | sed 's/[^A-Za-z0-9_]/_/g'
}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

mkdir -p "$CV_ROOT/logs"

SAFE_DATA_TAG=$(safe_name "$DATA_TAG")
DATA_SUFFIX=""
if [[ -n "$SAFE_DATA_TAG" ]]; then
  DATA_SUFFIX="_${SAFE_DATA_TAG}"
fi

case "$MODE" in
  holdout)
    SAFE_HOLDOUT_PRESET=$(safe_name "$HOLDOUT_PRESET")
    DEFAULT_TAG="holdout_${SAFE_HOLDOUT_PRESET}${DATA_SUFFIX}"
    if [[ -z "$RIVERS" ]]; then
      case "$HOLDOUT_PRESET" in
        CO) RIVERS="CO_UpperColorado_Topobathy_1_2020" ;;
        CA) RIVERS="CA_KlamathRiver_TopoBathy_2018_D18" ;;
        Santiam) RIVERS="OR_SantiamRiverTB_Topobathy_1_D23" ;;
        NE) RIVERS="NE_Niobrara_Topobathy_2018" ;;
        OR_MKRC) RIVERS="OR_MKRC_Topobathy_2021" ;;
        Nisqually) RIVERS="WA_Nisqually_Bathymetric_2020" ;;
        MD) RIVERS="MD_PotomacRiver_Bathy_2019" ;;
        Chehalis) RIVERS="WA_ChehalisRiverTB_Topobathy_1_D23" ;;
        MilwaukeeGroup) RIVERS="BadgerFinNull Estabrook_Combined KewaFix2Null Kletzch_Combined_UpMax3Null" ;;
        *) echo "[ERROR] Unknown HOLDOUT_PRESET=$HOLDOUT_PRESET" >&2; exit 2 ;;
      esac
    fi
    ;;
  stdstrat)
    if [[ -n "$VAL_RIVERS" ]]; then
      SAFE_MANUAL_VAL_TAG=$(safe_name "$MANUAL_VAL_TAG")
      DEFAULT_TAG="stdStratRiver_manualVal_${SAFE_MANUAL_VAL_TAG}_seed${STD_SPLIT_SEED}${DATA_SUFFIX}"
    else
      DEFAULT_TAG="stdStratRiver_${BIN_STAT}_valPerBin${VAL_PER_BIN}_seed${STD_SPLIT_SEED}${DATA_SUFFIX}"
    fi
    if [[ -z "$RIVERS" ]]; then
      RIVERS="$VAL_RIVERS"
    fi
    ;;
  *)
    echo "[ERROR] MODE must be holdout or stdstrat. Got MODE=$MODE" >&2
    exit 2
    ;;
esac

F010_OUT_DIR=${F010_OUT_DIR:-$F010_BASE/$DEFAULT_TAG}
OUT_DIR=${OUT_DIR:-$OUT_BASE/$DEFAULT_TAG}

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Missing script: $SCRIPT" >&2
  exit 2
fi
if [[ ! -d "$F010_OUT_DIR" ]]; then
  echo "[ERROR] Missing F010_OUT_DIR: $F010_OUT_DIR" >&2
  exit 2
fi

if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"
  else
    echo "[ERROR] Output is not empty: $OUT_DIR"
    echo "Set OVERWRITE=1 to replace it, or set OUT_DIR to a new folder."
    exit 3
  fi
fi
mkdir -p "$OUT_DIR"

read -r -a RIVER_ARRAY <<< "$RIVERS"
if [[ ${#RIVER_ARRAY[@]} -eq 0 ]]; then
  echo "[ERROR] RIVERS is empty." >&2
  exit 2
fi

echo "=== F021 GT/error tile VRT generation ==="
date
echo "HOST=$(hostname)"
echo "MODE=$MODE"
echo "DATA_TAG=$DATA_TAG"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "F010_OUT_DIR=$F010_OUT_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "RIVERS=$RIVERS"
echo "MASK=Core_Loss_Mask_Pixel AND valid_GT AND valid_prediction"
echo "SCRIPT=$SCRIPT"

python -u "$SCRIPT" \
  --f010_out_dir "$F010_OUT_DIR" \
  --output_dir "$OUT_DIR" \
  --rivers "${RIVER_ARRAY[@]}" \
  --nodata -999999 \
  --nodata_threshold -9999 \
  --progress_every "$PROGRESS_EVERY"

echo "=== DONE F021 ==="
echo "$OUT_DIR"
date
