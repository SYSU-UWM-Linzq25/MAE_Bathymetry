#!/usr/bin/env bash
# NUMBER-ALIGNED NEW FAMILY COPY: E026_run_NormThenMeter_train_val_core_pixel_eval_20260727.sh
# TEMPLATE SOURCE: E031_run_MeterOnly_train_val_core_pixel_eval_20260713.sh
# NUMBER-ALIGNED NAME: E031_run_MeterOnly_train_val_core_pixel_eval_20260713.sh
# ORIGINAL BACKUP NAME: E026_run_v4_NormThenMeter_train_val_core_pixel_eval_20260713.sh
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
#SBATCH -J E026_v4_meter_eval
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
CODE=${CODE:-$ROOT/mae_Retrain}
SCRIPT=${SCRIPT:-$WORK/script/E025_evaluate_NormThenMeter_core_pixel_metrics_overlayvis_20260727.py}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2/Tiles_1m}

SOURCE_CV_ROOT=${SOURCE_CV_ROOT:-$WORK/cross_validation_v2}
MODEL_CV_ROOT=${MODEL_CV_ROOT:-$WORK/cross_validation_v6_Stage2MeterMAE_FromNorm}
SOURCE_SPLIT_TAG=${SOURCE_SPLIT_TAG:-D001NoDataSafe}
RUN_TAG=${RUN_TAG:-D005Stage2MeterMAE_FromNorm_D001NoDataSafe}

HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}
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

RUNTIME_LOG_DIR="$MODEL_CV_ROOT/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/E026_v4_meter_eval_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.out" \
     2>"$RUNTIME_LOG_DIR/E026_v4_meter_eval_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.err"

safe_name() { echo "$1" | sed 's/[^A-Za-z0-9_]/_/g'; }
latest_run_with_ckpt() {
  local parent="$1"
  if [[ ! -d "$parent" ]]; then echo ""; return 0; fi
  find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth -printf '%T@ %h\n' 2>/dev/null \
    | sort -nr | awk 'NR==1{print $2}'
}

SAFE_PRESET=$(safe_name "$HOLDOUT_PRESET")
SPLIT_DIR=${SPLIT_DIR:-$SOURCE_CV_ROOT/splits/holdout_${SAFE_PRESET}_${SOURCE_SPLIT_TAG}}
RUN_PARENT="$MODEL_CV_ROOT/runs/holdout_${SAFE_PRESET}_${RUN_TAG}"
if [[ -z "$RUN_DIR" ]]; then RUN_DIR=$(latest_run_with_ckpt "$RUN_PARENT"); fi
if [[ -z "$RUN_DIR" ]]; then
  echo "[ERROR] Could not find checkpoint-best.pth under $RUN_PARENT" >&2
  exit 2
fi
CKPT=${CKPT:-$RUN_DIR/checkpoint-best.pth}
OUT_ROOT=${OUT_ROOT:-$RUN_DIR/eval_E025_NormThenMeter_core_pixel_overlayvis}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

for f in \
  "$SCRIPT" "$CKPT" \
  "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/train_loss.txt" \
  "$SPLIT_DIR/val_tiles.txt" "$SPLIT_DIR/val_hidden.txt" "$SPLIT_DIR/val_loss.txt"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing required file: $f" >&2; exit 2; }
done

if [[ -d "$OUT_ROOT" ]] && find "$OUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
  if [[ "$OVERWRITE_EVAL" == "1" ]]; then
    rm -rf "$OUT_ROOT"
  else
    echo "[ERROR] Evaluation output is not empty: $OUT_ROOT" >&2
    exit 3
  fi
fi
mkdir -p "$OUT_ROOT"

AMP_ARGS=()
[[ "$USE_AMP" == "1" || "$USE_AMP" == "true" || "$USE_AMP" == "TRUE" ]] && AMP_ARGS+=(--amp)
VIS_ARGS=()
[[ "$NO_VISUALS" == "1" || "$NO_VISUALS" == "true" || "$NO_VISUALS" == "TRUE" ]] && VIS_ARGS+=(--no_visuals)

echo "============================================================"
echo "E026 NormThenMeter train/val core-pixel evaluation"
date
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOST=$(hostname)"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "SOURCE_SPLIT=$SPLIT_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "CKPT=$CKPT"
echo "OUT_ROOT=$OUT_ROOT"
echo "PRIMARY_METRIC=MAE in meters on exact core loss pixels"
echo "SECONDARY_METRICS=RMSE meters, normalized MSE, bias, tile-scale diagnostics"
echo "============================================================"
nvidia-smi || true

run_split() {
  local split="$1"
  python -u "$SCRIPT" \
    --code_dir "$CODE" \
    --ckpt "$CKPT" \
    --data_root "$TILE_ROOT" \
    --list "$SPLIT_DIR/${split}_tiles.txt" \
    --hidden_list "$SPLIT_DIR/${split}_hidden.txt" \
    --loss_list "$SPLIT_DIR/${split}_loss.txt" \
    --output_dir "$OUT_ROOT/$split" \
    --split_name "$split" \
    --model mae_vit_large_patch16 \
    --input_size 336 \
    --in_chans 1 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device cuda \
    --nodata -999999 \
    --nodata_threshold -9999 \
    --tile_norm \
    --tile_norm_visible_only \
    --tile_norm_eps 1e-3 \
    --tile_norm_std_scale "$STD_SCALE" \
    --bottleneck_norm inst1d \
    --loss_mode mse \
    --mask_ratio 0.75 \
    --lcc_patch_threshold 0.5 \
    --loss_region_mode core \
    --core_patch_radius 3 \
    --min_valid_visible_patch_ratio 0.70 \
    --min_loss_pixel_count 1 \
    --rank_metric "$RANK_METRIC" \
    --worst_vis "$WORST_VIS" \
    --median_vis "$MEDIAN_VIS" \
    --best_vis "$BEST_VIS" \
    --vis_dpi "$VIS_DPI" \
    "${VIS_ARGS[@]}" \
    "${AMP_ARGS[@]}"
}

echo "=== Evaluate train ==="
run_split train
echo "=== Evaluate val ==="
run_split val

echo "=== DONE E026 ==="
echo "$OUT_ROOT"
date
