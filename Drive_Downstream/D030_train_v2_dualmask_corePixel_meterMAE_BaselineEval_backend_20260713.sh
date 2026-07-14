#!/usr/bin/env bash
set -euo pipefail

# D030: MAE v2 dual-mask meter-MAE backend with pre-training baseline evaluation.
#
# Inputs:
#   Train_tile + Hidden_Mask + Loss_Mask_Pixel
#
# Main v2 logic:
#   - Hidden_Mask controls what the model cannot see.
#   - Loss_Mask_Pixel controls pixel-level supervised loss.
#   - Loss is restricted to center core by default:
#       --loss_region_mode core
#       --core_patch_radius 3
#
# This script is called by D026/D027 and can also be run directly with an
# existing dual-mask split directory.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
CODE=${CODE:-$ROOT/mae_Retrain}

TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2/Tiles_1m}
SPLIT_DIR=${SPLIT_DIR:?Please set SPLIT_DIR containing train_tiles.txt/val_tiles.txt/train_hidden.txt/val_hidden.txt/train_loss.txt/val_loss.txt}

UP_CKPT=${UP_CKPT:-$ROOT/Upstream_Model_ReTrain/runs/Small_tilenorm_viscorr_336/checkpoint-best.pth}
RUN_NAME=${RUN_NAME:-stage4_v2_dualmask_meterMAE_tilenormVis_corePixel_1m_e400}
OUT_DIR=${OUT_DIR:-$WORK/runs/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

BATCH_SIZE=${BATCH_SIZE:-4}
ACCUM_ITER=${ACCUM_ITER:-4}
EPOCHS=${EPOCHS:-400}
LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-6}
PATIENCE=${PATIENCE:-60}
GPU_ID=${GPU_ID:-0}
NUM_WORKERS=${NUM_WORKERS:-1}

# Primary objective/checkpoint metric: exact core-pixel MAE after inverse tile normalization.
OPTIMIZATION_LOSS=${OPTIMIZATION_LOSS:-meter_mae}
BEST_METRIC=${BEST_METRIC:-val_mae_m_mask}
EARLY_STOP_METRIC=${EARLY_STOP_METRIC:-$BEST_METRIC}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.001}
EARLY_STOP_WARMUP_EPOCHS=${EARLY_STOP_WARMUP_EPOCHS:-20}
BASELINE_EVAL_BEFORE_TRAINING=${BASELINE_EVAL_BEFORE_TRAINING:-1}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}

# Keep old successful Stage-4 settings unless explicitly overridden.
BOTTLENECK_NORM=${BOTTLENECK_NORM:-inst1d}
STD_SCALE=${STD_SCALE:-1.5}
NODATA=${NODATA:-"-999999"}
NODATA_THRESHOLD=${NODATA_THRESHOLD:-"-9999"}
MIN_VALID_VISIBLE_PATCH_RATIO=${MIN_VALID_VISIBLE_PATCH_RATIO:-0.70}

LOSS_REGION_MODE=${LOSS_REGION_MODE:-core}
CORE_PATCH_RADIUS=${CORE_PATCH_RADIUS:-3}
MIN_CORE_VALID_PATCH_RATIO=${MIN_CORE_VALID_PATCH_RATIO:-0.85}
MIN_CORE_PREDICTION_PATCH_RATIO=${MIN_CORE_PREDICTION_PATCH_RATIO:-0.02}
MAX_CORE_PREDICTION_PATCH_RATIO=${MAX_CORE_PREDICTION_PATCH_RATIO:-0.90}

MIN_LOSS_PIXEL_COUNT=${MIN_LOSS_PIXEL_COUNT:-1}
PLOT_EVERY=${PLOT_EVERY:-1}
VIS_EVERY=${VIS_EVERY:-20}
VIS_N=${VIS_N:-10}
STATS_MAX_FILES=${STATS_MAX_FILES:-1000}

# FRESH_RUN=1 ignores existing checkpoints in OUT_DIR.
FRESH_RUN=${FRESH_RUN:-0}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

cd "$CODE"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "=== D030 MAE v2 dual-mask meter-MAE + baseline-eval backend ==="
echo "HOST=$(hostname)"
echo "ROOT=$ROOT"
echo "CODE=$CODE"
echo "TILE_ROOT=$TILE_ROOT"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "OUT_DIR=$OUT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "UP_CKPT=$UP_CKPT"
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "ACCUM_ITER=$ACCUM_ITER"
echo "EPOCHS=$EPOCHS"
echo "LR=$LR"
echo "MIN_LR=$MIN_LR"
echo "PATIENCE=$PATIENCE"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "OPTIMIZATION_LOSS=$OPTIMIZATION_LOSS"
echo "PRIMARY_OBJECTIVE=pixel-weighted absolute error in meters after inverse tile normalization"
echo "SUPERVISION_REGION=Loss_Mask_Pixel AND prediction/core patch mask AND valid patch"
echo "BEST_METRIC=$BEST_METRIC"
echo "EARLY_STOP_METRIC=$EARLY_STOP_METRIC"
echo "EARLY_STOP_MIN_DELTA=$EARLY_STOP_MIN_DELTA"
echo "EARLY_STOP_WARMUP_EPOCHS=$EARLY_STOP_WARMUP_EPOCHS"
echo "BASELINE_EVAL_BEFORE_TRAINING=$BASELINE_EVAL_BEFORE_TRAINING"
echo "BASELINE_EPOCH_LABEL=-1"
echo "BASELINE_CHECKPOINT=checkpoint-baseline.pth"
echo "BASELINE_CAN_REMAIN_BEST=yes"
echo "WARMUP_EPOCHS=$WARMUP_EPOCHS"
echo "BOTTLENECK_NORM=$BOTTLENECK_NORM"
echo "STD_SCALE=$STD_SCALE"
echo "NODATA=$NODATA"
echo "NODATA_THRESHOLD=$NODATA_THRESHOLD"
echo "MIN_VALID_VISIBLE_PATCH_RATIO=$MIN_VALID_VISIBLE_PATCH_RATIO"
echo "LOSS_REGION_MODE=$LOSS_REGION_MODE"
echo "CORE_PATCH_RADIUS=$CORE_PATCH_RADIUS"
echo "MIN_CORE_VALID_PATCH_RATIO=$MIN_CORE_VALID_PATCH_RATIO"
echo "MIN_CORE_PREDICTION_PATCH_RATIO=$MIN_CORE_PREDICTION_PATCH_RATIO"
echo "MAX_CORE_PREDICTION_PATCH_RATIO=$MAX_CORE_PREDICTION_PATCH_RATIO"
echo "MIN_LOSS_PIXEL_COUNT=$MIN_LOSS_PIXEL_COUNT"
echo "FRESH_RUN=$FRESH_RUN"

echo "=== Required-file checks ==="
for f in \
  "$CODE/main_pretrain_dem_meterMAE_BaselineEval_D030_20260713.py" \
  "$CODE/engine_pretrain_meterMAE_D030_20260713.py" \
  "$SPLIT_DIR/train_tiles.txt" \
  "$SPLIT_DIR/val_tiles.txt" \
  "$SPLIT_DIR/train_hidden.txt" \
  "$SPLIT_DIR/val_hidden.txt" \
  "$SPLIT_DIR/train_loss.txt" \
  "$SPLIT_DIR/val_loss.txt" \
  "$UP_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required file: $f" >&2
    exit 2
  fi
done

echo "=== Split counts ==="
wc -l "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/train_loss.txt"
wc -l "$SPLIT_DIR/val_tiles.txt" "$SPLIT_DIR/val_hidden.txt" "$SPLIT_DIR/val_loss.txt"

echo "=== Exact split SHA256 for cross-experiment comparison ==="
sha256sum \
  "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/val_tiles.txt" \
  "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/val_hidden.txt" \
  "$SPLIT_DIR/train_loss.txt" "$SPLIT_DIR/val_loss.txt" \
  | tee "$OUT_DIR/input_split_sha256.txt"

echo "=== CUDA check ==="
which python
python - <<'PY'
import sys
import torch

print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), "count", torch.cuda.device_count())
if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available inside this job.")
    sys.exit(10)

free_b, total_b = torch.cuda.mem_get_info()
free_gb = free_b / (1024 ** 3)
total_gb = total_b / (1024 ** 3)
print(f"[GPU CHECK] device={torch.cuda.current_device()} name={torch.cuda.get_device_name(0)} free={free_gb:.2f} GiB total={total_gb:.2f} GiB")
if free_gb < 6.0:
    print("[ERROR] Assigned GPU has less than 6 GiB free.")
    sys.exit(11)
PY
nvidia-smi || true

# Preserve the exact split files used by this run.
cp -pf "$SPLIT_DIR"/train_tiles.txt "$OUT_DIR/input_train_tiles.txt"
cp -pf "$SPLIT_DIR"/val_tiles.txt "$OUT_DIR/input_val_tiles.txt"
cp -pf "$SPLIT_DIR"/train_hidden.txt "$OUT_DIR/input_train_hidden.txt"
cp -pf "$SPLIT_DIR"/val_hidden.txt "$OUT_DIR/input_val_hidden.txt"
cp -pf "$SPLIT_DIR"/train_loss.txt "$OUT_DIR/input_train_loss.txt"
cp -pf "$SPLIT_DIR"/val_loss.txt "$OUT_DIR/input_val_loss.txt"
if [[ -f "$SPLIT_DIR/split_summary.txt" ]]; then
  cp -pf "$SPLIT_DIR/split_summary.txt" "$OUT_DIR/input_split_summary.txt"
fi
if [[ -f "$SPLIT_DIR/split_manifest.csv" ]]; then
  cp -pf "$SPLIT_DIR/split_manifest.csv" "$OUT_DIR/input_split_manifest.csv"
fi

python "$CODE/main_pretrain_dem_meterMAE_BaselineEval_D030_20260713.py" --help > "$OUT_DIR/help.txt" 2>&1 || true
for key in \
  train_hidden_list val_hidden_list train_loss_list val_loss_list \
  min_loss_pixel_count min_core_loss_pixel_count min_core_loss_pixel_ratio \
  bottleneck_norm tile_norm_visible_only tile_norm_std_scale \
  loss_region_mode core_patch_radius min_core_valid_patch_ratio \
  min_core_prediction_patch_ratio max_core_prediction_patch_ratio \
  optimization_loss early_stop_patience early_stop_metric best_metric val_mae_m_mask eval_rmse baseline_eval_before_training; do
  if ! grep -q -- "$key" "$OUT_DIR/help.txt"; then
    echo "[ERROR] meter-MAE main script does not support expected argument containing: $key" >&2
    echo "        See $OUT_DIR/help.txt" >&2
    exit 3
  fi
done

RESUME_ARGS=()
if [[ "$FRESH_RUN" != "1" ]]; then
  CKPT=$(find "$OUT_DIR" -maxdepth 1 -type f -regextype posix-extended \
    -regex '.*/checkpoint-[0-9]{4}\.pth' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk 'NR==1{print $2}' || true)
  if [[ -z "${CKPT:-}" && -f "$OUT_DIR/checkpoint-best.pth" ]]; then
    CKPT="$OUT_DIR/checkpoint-best.pth"
  fi
  if [[ -n "${CKPT:-}" ]]; then
    echo "[RESUME] Found checkpoint: $CKPT"
    RESUME_ARGS=(--resume "$CKPT")
  else
    echo "[RESUME] No checkpoint found. Starting from upstream init_ckpt only."
  fi
else
  echo "[RESUME] FRESH_RUN=1. Ignoring checkpoints in OUT_DIR."
fi

BASELINE_ARGS=()
if [[ "$BASELINE_EVAL_BEFORE_TRAINING" == "1" || "$BASELINE_EVAL_BEFORE_TRAINING" == "true" || "$BASELINE_EVAL_BEFORE_TRAINING" == "TRUE" ]]; then
  BASELINE_ARGS+=(--baseline_eval_before_training)
else
  BASELINE_ARGS+=(--no_baseline_eval_before_training)
fi

PYTHONUNBUFFERED=1 python -u main_pretrain_dem_meterMAE_BaselineEval_D030_20260713.py \
  --data_root "$TILE_ROOT" \
  --train_list "$SPLIT_DIR/train_tiles.txt" \
  --val_list "$SPLIT_DIR/val_tiles.txt" \
  --train_hidden_list "$SPLIT_DIR/train_hidden.txt" \
  --val_hidden_list "$SPLIT_DIR/val_hidden.txt" \
  --train_loss_list "$SPLIT_DIR/train_loss.txt" \
  --val_loss_list "$SPLIT_DIR/val_loss.txt" \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --in_chans 1 \
  --batch_size "$BATCH_SIZE" \
  --accum_iter "$ACCUM_ITER" \
  --epochs "$EPOCHS" \
  --init_ckpt "$UP_CKPT" \
  --freeze_encoder \
  --freeze_last_n_encoder_blocks 0 \
  --bottleneck_norm "$BOTTLENECK_NORM" \
  --optimization_loss "$OPTIMIZATION_LOSS" \
  --loss_mode mse \
  --tile_norm \
  --tile_norm_visible_only \
  --tile_norm_eps 1e-3 \
  --tile_norm_std_scale "$STD_SCALE" \
  --nodata "$NODATA" \
  --nodata_threshold "$NODATA_THRESHOLD" \
  --lcc_mask_mode exact \
  --lcc_patch_threshold 0.5 \
  --loss_region_mode "$LOSS_REGION_MODE" \
  --core_patch_radius "$CORE_PATCH_RADIUS" \
  --min_valid_visible_patch_ratio "$MIN_VALID_VISIBLE_PATCH_RATIO" \
  --min_loss_pixel_count "$MIN_LOSS_PIXEL_COUNT" \
  --min_core_valid_patch_ratio "$MIN_CORE_VALID_PATCH_RATIO" \
  --min_core_prediction_patch_ratio "$MIN_CORE_PREDICTION_PATCH_RATIO" \
  --max_core_prediction_patch_ratio "$MAX_CORE_PREDICTION_PATCH_RATIO" \
  --eval_rmse \
  --best_metric "$BEST_METRIC" \
  --early_stop_metric "$EARLY_STOP_METRIC" \
  --early_stop_patience "$PATIENCE" \
  --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
  --early_stop_warmup_epochs "$EARLY_STOP_WARMUP_EPOCHS" \
  --plot_every "$PLOT_EVERY" \
  --vis_every "$VIS_EVERY" \
  --vis_n "$VIS_N" \
  --stats_max_files "$STATS_MAX_FILES" \
  --lr "$LR" \
  --min_lr "$MIN_LR" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --output_dir "$OUT_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda \
  --num_workers "$NUM_WORKERS" \
  --pin_mem \
  "${BASELINE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

date
echo "=== D030 MAE v2 dual-mask meter-MAE + baseline-eval training done ==="
echo "RUN=$OUT_DIR"
