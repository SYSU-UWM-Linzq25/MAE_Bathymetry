#!/usr/bin/env bash
# NUMBER-ALIGNED NAME: E021_run_NormOnly_train_val_core_pixel_eval_overlayvis_20260710.sh
# ORIGINAL BACKUP NAME: E051_run_v2_train_val_core_pixel_eval_D001NoDataSafe_overlayvis_20260710.sh
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
#SBATCH -J e051_ND_overlayvis
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/E051_ND_overlayvis_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/E051_ND_overlayvis_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
CODE=${CODE:-$ROOT/mae_Retrain}
SCRIPT=${SCRIPT:-$WORK/script/E020_evaluate_NormOnly_core_pixel_metrics_overlayvis_20260710.py}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2/Tiles_1m}
CV_ROOT=${CV_ROOT:-$WORK/cross_validation_v2}
EVAL_MODE=${EVAL_MODE:-holdout}
DATA_TAG=${DATA_TAG:-D001NoDataSafe}
HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}
STD_SPLIT_SEED=${STD_SPLIT_SEED:-42}
MANUAL_VAL_TAG=${MANUAL_VAL_TAG:-CO_Nisqually_NE}
BIN_STAT=${BIN_STAT:-median}
VAL_PER_BIN=${VAL_PER_BIN:-1}
VAL_RIVERS=${VAL_RIVERS:-}
GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-1}
STD_SCALE=${STD_SCALE:-1.5}
USE_AMP=${USE_AMP:-1}
OVERWRITE_EVAL=${OVERWRITE_EVAL:-0}
RANK_METRIC=${RANK_METRIC:-rmse_m_core_loss_pixel}
WORST_VIS=${WORST_VIS:-20}
MEDIAN_VIS=${MEDIAN_VIS:-10}
BEST_VIS=${BEST_VIS:-10}
VIS_DPI=${VIS_DPI:-180}
NO_VISUALS=${NO_VISUALS:-0}
RUN_DIR=${RUN_DIR:-}
SPLIT_DIR=${SPLIT_DIR:-}
CKPT=${CKPT:-}
OUT_ROOT=${OUT_ROOT:-}

safe_name() { echo "$1" | sed 's/[^A-Za-z0-9_]/_/g'; }
latest_run_with_ckpt() {
  local parent="$1"
  if [[ ! -d "$parent" ]]; then echo ""; return 0; fi
  find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth -printf '%T@ %h\n' 2>/dev/null | sort -nr | awk 'NR==1{print $2}'
}

SAFE_DATA_TAG=$(safe_name "$DATA_TAG")
DATA_SUFFIX=""
if [[ -n "$SAFE_DATA_TAG" ]]; then DATA_SUFFIX="_${SAFE_DATA_TAG}"; fi

if [[ -z "$SPLIT_DIR" || -z "$RUN_DIR" ]]; then
  case "$EVAL_MODE" in
    holdout)
      SAFE_HOLDOUT_PRESET=$(safe_name "$HOLDOUT_PRESET")
      DEFAULT_SPLIT_DIR="$CV_ROOT/splits/holdout_${SAFE_HOLDOUT_PRESET}${DATA_SUFFIX}"
      DEFAULT_RUN_PARENT="$CV_ROOT/runs/holdout_${SAFE_HOLDOUT_PRESET}${DATA_SUFFIX}"
      ;;
    stdstrat)
      if [[ -n "$VAL_RIVERS" ]]; then
        SAFE_MANUAL_VAL_TAG=$(safe_name "$MANUAL_VAL_TAG")
        SPLIT_NAME="stdStratRiver_manualVal_${SAFE_MANUAL_VAL_TAG}_seed${STD_SPLIT_SEED}${DATA_SUFFIX}"
      else
        SPLIT_NAME="stdStratRiver_${BIN_STAT}_valPerBin${VAL_PER_BIN}_seed${STD_SPLIT_SEED}${DATA_SUFFIX}"
      fi
      DEFAULT_SPLIT_DIR="$CV_ROOT/splits/$SPLIT_NAME"
      DEFAULT_RUN_PARENT="$CV_ROOT/runs/$SPLIT_NAME"
      ;;
    *) echo "[ERROR] Unknown EVAL_MODE=$EVAL_MODE. Use holdout or stdstrat." >&2; exit 2 ;;
  esac
  SPLIT_DIR=${SPLIT_DIR:-$DEFAULT_SPLIT_DIR}
  if [[ -z "$RUN_DIR" ]]; then RUN_DIR=$(latest_run_with_ckpt "$DEFAULT_RUN_PARENT"); fi
fi

if [[ -z "$RUN_DIR" ]]; then echo "[ERROR] Could not auto-detect RUN_DIR. Pass RUN_DIR explicitly." >&2; exit 2; fi
CKPT=${CKPT:-$RUN_DIR/checkpoint-best.pth}
OUT_ROOT=${OUT_ROOT:-$RUN_DIR/eval_E050_D001NoDataSafe_overlayvis_core_pixel_diagnosis}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "$OUT_ROOT" "$CV_ROOT/logs"
echo "=== E051 D001NoDataSafe overlay visualization evaluation ==="
date
echo "HOST=$(hostname)"
echo "EVAL_MODE=$EVAL_MODE"
echo "DATA_TAG=$DATA_TAG"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "RUN_DIR=$RUN_DIR"
echo "CKPT=$CKPT"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "SCRIPT=$SCRIPT"
echo "TILE_ROOT=$TILE_ROOT"
echo "GPU_ID=$GPU_ID"
echo "NO_VISUALS=$NO_VISUALS"
nvidia-smi || true
for f in "$SCRIPT" "$CKPT" "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/train_loss.txt" "$SPLIT_DIR/val_tiles.txt" "$SPLIT_DIR/val_hidden.txt" "$SPLIT_DIR/val_loss.txt"; do
  if [[ ! -f "$f" ]]; then echo "[ERROR] Missing required file: $f" >&2; exit 2; fi
done
if [[ -d "$OUT_ROOT" ]] && find "$OUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
  if [[ "$OVERWRITE_EVAL" == "1" ]]; then rm -rf "$OUT_ROOT"; mkdir -p "$OUT_ROOT"; else echo "[ERROR] Output is not empty: $OUT_ROOT"; echo "Set OVERWRITE_EVAL=1 to replace it."; exit 3; fi
fi
AMP_ARGS=(); if [[ "$USE_AMP" == "1" || "$USE_AMP" == "true" || "$USE_AMP" == "TRUE" ]]; then AMP_ARGS+=(--amp); fi
VIS_ARGS=(); if [[ "$NO_VISUALS" == "1" || "$NO_VISUALS" == "true" || "$NO_VISUALS" == "TRUE" ]]; then VIS_ARGS+=(--no_visuals); fi
run_split() {
  local split="$1"
  local out_dir="$OUT_ROOT/$split"
  echo; echo "=== Evaluate split=$split ==="
  python -u "$SCRIPT" --code_dir "$CODE" --ckpt "$CKPT" --data_root "$TILE_ROOT" --list "$SPLIT_DIR/${split}_tiles.txt" --hidden_list "$SPLIT_DIR/${split}_hidden.txt" --loss_list "$SPLIT_DIR/${split}_loss.txt" --output_dir "$out_dir" --split_name "$split" --model mae_vit_large_patch16 --input_size 336 --in_chans 1 --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" --device cuda --nodata -999999 --nodata_threshold -9999 --tile_norm --tile_norm_visible_only --tile_norm_eps 1e-3 --tile_norm_std_scale "$STD_SCALE" --bottleneck_norm inst1d --loss_mode mse --mask_ratio 0.75 --lcc_patch_threshold 0.5 --loss_region_mode core --core_patch_radius 3 --min_valid_visible_patch_ratio 0.70 --min_loss_pixel_count 1 --rank_metric "$RANK_METRIC" --worst_vis "$WORST_VIS" --median_vis "$MEDIAN_VIS" --best_vis "$BEST_VIS" --vis_dpi "$VIS_DPI" "${VIS_ARGS[@]}" "${AMP_ARGS[@]}"
}
run_split train
run_split val
echo "=== DONE E051 ==="
echo "$OUT_ROOT"
date
