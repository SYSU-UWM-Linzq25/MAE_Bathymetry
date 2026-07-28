#!/usr/bin/env bash
# NUMBER-ALIGNED NAME: F021_run_fullriver_inference_NormOnly_TileAvgVRT_20260710.sh
# ORIGINAL BACKUP NAME: F013_run_fullriver_inference_tileavg_vrt_CoreFinalLossOnly_Inst1DSafe_20260710.sh
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
#SBATCH -J f013_inst1d_safe
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/F013_inst1d_safe_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/F013_inst1d_safe_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
CODE=${CODE:-$ROOT/mae_Retrain}
SCRIPT=${SCRIPT:-$WORK/script/F020_fullriver_predict_NormOnly_TileAvgVRT_CoreFinalLossOnly_20260710.py}
CV_ROOT=${CV_ROOT:-$WORK/cross_validation_v2}

TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_FullRiver_E001/Tiles_1m}
OUT_BASE=${OUT_BASE:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Predictions_F010_TileAvgVRT_D001NoDataSafe}

# MODE=holdout: restore the held-out river for the selected holdout model.
# MODE=stdstrat: restore all validation rivers from VAL_RIVERS unless RIVERS is explicitly set.
MODE=${MODE:-holdout}
DATA_TAG=${DATA_TAG:-D001NoDataSafe}
HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}
STD_SPLIT_SEED=${STD_SPLIT_SEED:-42}
MANUAL_VAL_TAG=${MANUAL_VAL_TAG:-CO_Nisqually_NE}
BIN_STAT=${BIN_STAT:-median}
VAL_PER_BIN=${VAL_PER_BIN:-1}
VAL_RIVERS=${VAL_RIVERS:-CO_UpperColorado_Topobathy_1_2020 WA_Nisqually_Bathymetric_2020 NE_Niobrara_Topobathy_2018}
RIVERS=${RIVERS:-}

GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-4}
STD_SCALE=${STD_SCALE:-1.5}
USE_AMP=${USE_AMP:-1}
OVERWRITE=${OVERWRITE:-0}
# RESUME=1 keeps completed river folders and rebuilds only incomplete rivers.
RESUME=${RESUME:-1}

# Optional direct override if auto-detection is not desired.
RUN_DIR=${RUN_DIR:-}
CKPT=${CKPT:-}
OUT_DIR=${OUT_DIR:-}

safe_name() {
  echo "$1" | sed 's/[^A-Za-z0-9_]/_/g'
}

latest_run_with_ckpt() {
  local parent="$1"
  if [[ ! -d "$parent" ]]; then
    echo ""
    return 0
  fi
  find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth -printf '%T@ %h\n' 2>/dev/null \
    | sort -nr | awk 'NR==1{print $2}'
}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

SAFE_DATA_TAG=$(safe_name "$DATA_TAG")
DATA_SUFFIX=""
if [[ -n "$SAFE_DATA_TAG" ]]; then
  DATA_SUFFIX="_${SAFE_DATA_TAG}"
fi

if [[ -z "$RUN_DIR" ]]; then
  case "$MODE" in
    holdout)
      SAFE_HOLDOUT_PRESET=$(safe_name "$HOLDOUT_PRESET")
      RUN_PARENT="$CV_ROOT/runs/holdout_${SAFE_HOLDOUT_PRESET}${DATA_SUFFIX}"
      RUN_DIR=$(latest_run_with_ckpt "$RUN_PARENT")
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
      DEFAULT_OUT_TAG="holdout_${SAFE_HOLDOUT_PRESET}${DATA_SUFFIX}"
      ;;
    stdstrat)
      if [[ -n "$VAL_RIVERS" ]]; then
        SAFE_MANUAL_VAL_TAG=$(safe_name "$MANUAL_VAL_TAG")
        SPLIT_NAME="stdStratRiver_manualVal_${SAFE_MANUAL_VAL_TAG}_seed${STD_SPLIT_SEED}${DATA_SUFFIX}"
      else
        SPLIT_NAME="stdStratRiver_${BIN_STAT}_valPerBin${VAL_PER_BIN}_seed${STD_SPLIT_SEED}${DATA_SUFFIX}"
      fi
      RUN_PARENT="$CV_ROOT/runs/$SPLIT_NAME"
      RUN_DIR=$(latest_run_with_ckpt "$RUN_PARENT")
      if [[ -z "$RIVERS" ]]; then
        RIVERS="$VAL_RIVERS"
      fi
      DEFAULT_OUT_TAG="$SPLIT_NAME"
      ;;
    *)
      echo "[ERROR] MODE must be holdout or stdstrat. Got MODE=$MODE" >&2
      exit 2
      ;;
  esac
else
  DEFAULT_OUT_TAG="custom_run${DATA_SUFFIX}"
fi

if [[ -z "$RUN_DIR" ]]; then
  echo "[ERROR] Could not auto-detect RUN_DIR." >&2
  echo "        MODE=$MODE HOLDOUT_PRESET=$HOLDOUT_PRESET DATA_TAG=$DATA_TAG" >&2
  echo "        Pass RUN_DIR explicitly if needed." >&2
  exit 2
fi

CKPT=${CKPT:-$RUN_DIR/checkpoint-best.pth}
OUT_DIR=${OUT_DIR:-$OUT_BASE/$DEFAULT_OUT_TAG}

mkdir -p "$OUT_DIR" "$CV_ROOT/logs"

if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  if [[ "$OVERWRITE" == "1" ]]; then
    echo "[OVERWRITE] Removing the entire output directory: $OUT_DIR"
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"
  elif [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "TRUE" ]]; then
    echo "[RESUME] Reusing output directory: $OUT_DIR"
  else
    echo "[ERROR] Output is not empty: $OUT_DIR"
    echo "Set RESUME=1 to preserve complete rivers and rebuild incomplete ones,"
    echo "or set OVERWRITE=1 to replace the entire output."
    exit 3
  fi
fi

AMP_ARGS=()
RESUME_ARGS=()
if [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "TRUE" ]]; then
  RESUME_ARGS+=(--resume)
fi
if [[ "$USE_AMP" == "1" || "$USE_AMP" == "true" || "$USE_AMP" == "TRUE" ]]; then
  AMP_ARGS+=(--amp)
fi


read -r -a RIVER_ARRAY <<< "$RIVERS"
if [[ ${#RIVER_ARRAY[@]} -eq 0 ]]; then
  echo "[ERROR] RIVERS is empty. For MODE=stdstrat, set VAL_RIVERS or RIVERS." >&2
  exit 2
fi

for f in "$SCRIPT" "$CKPT" "$TILE_ROOT/FullRiver_tile" "$TILE_ROOT/Hidden_Mask" "$TILE_ROOT/Loss_Mask_Pixel" "$TILE_ROOT/Core_Loss_Mask_Pixel"; do
  if [[ ! -e "$f" ]]; then
    echo "[ERROR] Missing required path: $f" >&2
    exit 2
  fi
done

echo "=== F013 Inst1D-safe full-river inference ==="
date
echo "HOST=$(hostname)"
echo "MODE=$MODE"
echo "DATA_TAG=$DATA_TAG"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "RUN_DIR=$RUN_DIR"
echo "CKPT=$CKPT"
echo "TILE_ROOT=$TILE_ROOT"
echo "FINAL_MOSAIC_MASK=Core_Loss_Mask_Pixel only; Hidden_Mask is model input only"
echo "OUTPUT=TileAvgVRT: no dense full-river array; averaged per-tile GeoTIFFs + VRT"
echo "OUT_DIR=$OUT_DIR"
echo "RIVERS=$RIVERS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "RESUME=$RESUME"
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

python -u "$SCRIPT" \
  --code_dir "$CODE" \
  --ckpt "$CKPT" \
  --tile_root "$TILE_ROOT" \
  --output_dir "$OUT_DIR" \
  --rivers "${RIVER_ARRAY[@]}" \
  --batch_size "$BATCH_SIZE" \
  --device cuda \
  --tile_norm_std_scale "$STD_SCALE" \
  --bottleneck_norm inst1d \
  --loss_region_mode core \
  --core_patch_radius 3 \
  "${RESUME_ARGS[@]}" \
  "${AMP_ARGS[@]}"

echo "=== DONE F013 ==="
echo "$OUT_DIR"
date
