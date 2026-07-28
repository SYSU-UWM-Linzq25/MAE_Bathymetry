#!/usr/bin/env bash
# NUMBER-ALIGNED NAME: E010a_eval_crossval_all_holdouts_predictionOnly_coreBox.sh
# ORIGINAL BACKUP NAME: E041_eval_crossval_all_holdouts_predictionOnly_coreBox.sh
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
#SBATCH -J e041_cv_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation/logs/E041_eval_cv_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation/logs/E041_eval_cv_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

# ============================================================
# E041: evaluate all cross-validation holdout runs with E030.
#
# Default behavior:
#   * discover all holdout_* folders under cross_validation/runs;
#   * include all folds, including holdout_OR_SantiamRiverTB_Topobathy_1_D23;
#   * evaluate each checkpoint-best.pth into:
#       cross_validation/evaluation/holdout_<FOLD>/eval_E030_predictionOnly_coreBox
#
# Usage:
#   sbatch E010a_eval_crossval_all_holdouts_predictionOnly_coreBox.sh
#
# Evaluate only selected folds:
#   sbatch E010a_eval_crossval_all_holdouts_predictionOnly_coreBox.sh \
#     WA_ChehalisRiverTB_Topobathy_1_D23 MilwaukeeRiverGroup
#
# Override GPU/node at submission:
#   sbatch -p HydroIntel -w execute-4006 --export=ALL,GPU_ID=0 \
#     E010a_eval_crossval_all_holdouts_predictionOnly_coreBox.sh
#
# Re-run/overwrite existing evaluations:
#   sbatch --export=ALL,GPU_ID=0,OVERWRITE_EVAL=1 \
#     E010a_eval_crossval_all_holdouts_predictionOnly_coreBox.sh
# ============================================================

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
CV_ROOT=${CV_ROOT:-$WORK/cross_validation}
RUNS_ROOT=${RUNS_ROOT:-$CV_ROOT/runs}
EVAL_ROOT=${EVAL_ROOT:-$CV_ROOT/evaluation}
LOG_ROOT=${LOG_ROOT:-$CV_ROOT/logs}

CODE=${CODE:-$ROOT/mae_Retrain}
SCRIPT=${SCRIPT:-$WORK/script/E005_evaluate_stage4_predictionOnly_coreBox.py}
BATH=${BATH:-$ROOT/Data/Tiles_for_Training_1m/1m_Tiles}
MASK=${MASK:-$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles}

# Evaluation/training-consistent settings.
STD_SCALE=${STD_SCALE:-1.5}
NODATA=${NODATA:-"-999999"}
NODATA_THRESHOLD=${NODATA_THRESHOLD:-"-9999"}
LOSS_REGION_MODE=${LOSS_REGION_MODE:-core}
CORE_PATCH_RADIUS=${CORE_PATCH_RADIUS:-3}
MIN_VISIBLE_PATCH_RATIO=${MIN_VISIBLE_PATCH_RATIO:-0.70}
MIN_CORE_VALID_PATCH_RATIO=${MIN_CORE_VALID_PATCH_RATIO:-0.85}
MIN_CORE_PREDICTION_PATCH_RATIO=${MIN_CORE_PREDICTION_PATCH_RATIO:-0.02}
MAX_CORE_PREDICTION_PATCH_RATIO=${MAX_CORE_PREDICTION_PATCH_RATIO:-0.90}
MIN_PREDICTION_PATCH_RATIO=${MIN_PREDICTION_PATCH_RATIO:-0.0001}
MAX_PREDICTION_PATCH_RATIO=${MAX_PREDICTION_PATCH_RATIO:-0.80}

BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-8}
RANK_METRIC=${RANK_METRIC:-rmse_m_core_exact_pixel}
WORST_VIS=${WORST_VIS:-30}
MEDIAN_VIS=${MEDIAN_VIS:-10}
BEST_VIS=${BEST_VIS:-10}
OVERWRITE_EVAL=${OVERWRITE_EVAL:-0}
USE_AMP=${USE_AMP:-1}

# Default is empty: evaluate all folds, including Santiam.
# Example to skip intentionally:
#   SKIP_FOLDS="OR_SantiamRiverTB_Topobathy_1_D23"
SKIP_FOLDS=${SKIP_FOLDS:-""}

mkdir -p "$EVAL_ROOT" "$LOG_ROOT"

echo "=== E041 CROSS-VALIDATION EVALUATION ==="
date
echo "HOST=$(hostname)"
echo "ROOT=$ROOT"
echo "WORK=$WORK"
echo "CV_ROOT=$CV_ROOT"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "EVAL_ROOT=$EVAL_ROOT"
echo "SCRIPT=$SCRIPT"
echo "BATH=$BATH"
echo "MASK=$MASK"
echo "RANK_METRIC=$RANK_METRIC"
echo "SKIP_FOLDS=$SKIP_FOLDS"
echo "OVERWRITE_EVAL=$OVERWRITE_EVAL"

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

GPU_ID=${GPU_ID:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count()); print('name0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
nvidia-smi || true

for f in "$SCRIPT" "$BATH" "$MASK" "$RUNS_ROOT"; do
  if [[ ! -e "$f" ]]; then
    echo "[ERROR] Missing required path: $f" >&2
    exit 2
  fi
done

safe_name() {
  echo "$1" | sed 's/[^A-Za-z0-9_]/_/g'
}

is_skipped() {
  local fold="$1"
  for s in $SKIP_FOLDS; do
    [[ "$fold" == "$s" ]] && return 0
  done
  return 1
}

find_run_dir() {
  local holdout_dir="$1"
  local ckpt
  ckpt=$(find "$holdout_dir" -mindepth 2 -maxdepth 3 -type f -name 'checkpoint-best.pth' -print 2>/dev/null | sort | tail -n 1 || true)
  if [[ -z "$ckpt" ]]; then
    ckpt=$(find "$holdout_dir" -type f -name 'checkpoint-best.pth' -print 2>/dev/null | sort | tail -n 1 || true)
  fi
  if [[ -z "$ckpt" ]]; then
    return 1
  fi
  dirname "$ckpt"
}

if [[ $# -gt 0 ]]; then
  FOLDS=("$@")
else
  mapfile -t FOLDS < <(find "$RUNS_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'holdout_*' -printf '%f\n' | sed 's/^holdout_//' | sort)
fi

if [[ ${#FOLDS[@]} -eq 0 ]]; then
  echo "[ERROR] No folds found." >&2
  exit 3
fi

echo "=== Fold list ==="
printf '  %s\n' "${FOLDS[@]}"

for FOLD in "${FOLDS[@]}"; do
  SAFE_FOLD=$(safe_name "$FOLD")
  if is_skipped "$FOLD"; then
    echo "[SKIP] $FOLD"
    continue
  fi

  HOLDOUT_DIR="$RUNS_ROOT/holdout_${SAFE_FOLD}"
  SPLIT="$CV_ROOT/splits/holdout_${SAFE_FOLD}"
  OUT="$EVAL_ROOT/holdout_${SAFE_FOLD}/eval_E030_predictionOnly_coreBox"
  STATUS_DIR="$EVAL_ROOT/holdout_${SAFE_FOLD}"

  echo
  echo "============================================================"
  echo "[FOLD] $FOLD"
  echo "HOLDOUT_DIR=$HOLDOUT_DIR"
  echo "SPLIT=$SPLIT"
  echo "OUT=$OUT"
  echo "============================================================"

  if [[ ! -d "$HOLDOUT_DIR" ]]; then
    echo "[WARN] Missing holdout run dir: $HOLDOUT_DIR"
    continue
  fi
  if [[ ! -d "$SPLIT" ]]; then
    echo "[WARN] Missing split dir: $SPLIT"
    continue
  fi

  for f in "$SPLIT/val.txt" "$SPLIT/val_masks.txt"; do
    if [[ ! -f "$f" ]]; then
      echo "[WARN] Missing split file: $f"
      continue 2
    fi
  done

  RUN_DIR=$(find_run_dir "$HOLDOUT_DIR" || true)
  if [[ -z "${RUN_DIR:-}" ]]; then
    echo "[WARN] No checkpoint-best.pth found under: $HOLDOUT_DIR"
    continue
  fi
  CKPT="$RUN_DIR/checkpoint-best.pth"

  if [[ -f "$OUT/summary.json" && "$OVERWRITE_EVAL" != "1" ]]; then
    echo "[SKIP] Existing evaluation: $OUT/summary.json"
    continue
  fi

  if [[ "$OVERWRITE_EVAL" == "1" && -d "$OUT" ]]; then
    echo "[OVERWRITE] Removing old OUT=$OUT"
    rm -rf "$OUT"
  fi

  mkdir -p "$OUT" "$STATUS_DIR"
  echo "$RUN_DIR" > "$OUT/run_dir.txt"
  echo "$CKPT" > "$OUT/checkpoint.txt"
  echo "$FOLD" > "$OUT/fold_name.txt"

  AMP_ARGS=()
  if [[ "$USE_AMP" == "1" || "$USE_AMP" == "true" || "$USE_AMP" == "TRUE" ]]; then
    AMP_ARGS+=(--amp)
  fi

  echo "[RUN] checkpoint=$CKPT"
  PYTHONUNBUFFERED=1 python -u "$SCRIPT" \
    --code_dir "$CODE" \
    --ckpt "$CKPT" \
    --data_root "$BATH" \
    --list "$SPLIT/val.txt" \
    --lcc_mask_path "$MASK" \
    --lcc_list "$SPLIT/val_masks.txt" \
    --output_dir "$OUT" \
    --split_name "$FOLD" \
    --model mae_vit_large_patch16 \
    --input_size 336 \
    --in_chans 1 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device cuda \
    --nodata "$NODATA" \
    --nodata_threshold "$NODATA_THRESHOLD" \
    --tile_norm \
    --tile_norm_visible_only \
    --tile_norm_eps 1e-3 \
    --tile_norm_std_scale "$STD_SCALE" \
    --bottleneck_norm inst1d \
    --mask_ratio 0.75 \
    --lcc_priority 10.0 \
    --lcc_patch_threshold 0.5 \
    --min_prediction_patch_ratio "$MIN_PREDICTION_PATCH_RATIO" \
    --max_prediction_patch_ratio "$MAX_PREDICTION_PATCH_RATIO" \
    --min_valid_visible_patch_ratio "$MIN_VISIBLE_PATCH_RATIO" \
    --loss_region_mode "$LOSS_REGION_MODE" \
    --core_patch_radius "$CORE_PATCH_RADIUS" \
    --min_core_valid_patch_ratio "$MIN_CORE_VALID_PATCH_RATIO" \
    --min_core_prediction_patch_ratio "$MIN_CORE_PREDICTION_PATCH_RATIO" \
    --max_core_prediction_patch_ratio "$MAX_CORE_PREDICTION_PATCH_RATIO" \
    --rank_metric "$RANK_METRIC" \
    --worst_vis "$WORST_VIS" \
    --median_vis "$MEDIAN_VIS" \
    --best_vis "$BEST_VIS" \
    "${AMP_ARGS[@]}"

  echo "[DONE FOLD] $FOLD -> $OUT"
done

echo
echo "=== ALL REQUESTED EVALUATIONS DONE ==="
echo "$EVAL_ROOT"
date
