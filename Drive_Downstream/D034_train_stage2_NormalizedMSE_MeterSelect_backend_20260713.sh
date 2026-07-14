#!/usr/bin/env bash
set -euo pipefail

# D034: Stage-2 dual-mask fine-tuning backend.
#
# Stage 1 (already completed):
#   optimization objective = meter MAE
#   selected checkpoint     = minimum validation meter MAE
#
# Stage 2 (this script):
#   initialization          = Stage-1 checkpoint-best.pth
#   optimization objective  = normalized MSE on the exact core loss pixels
#   selected checkpoint     = minimum validation meter MAE
#
# The untouched Stage-1 model is evaluated as epoch -1 and copied to
# checkpoint-best.pth before any Stage-2 update. Therefore Stage 2 cannot
# replace the formal best model unless validation meter MAE improves.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
CODE=${CODE:-$ROOT/mae_Retrain}

TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2/Tiles_1m}
SPLIT_DIR=${SPLIT_DIR:?Please set SPLIT_DIR}
STAGE1_CKPT=${STAGE1_CKPT:?Please set STAGE1_CKPT to the Stage-1 meter-MAE checkpoint-best.pth}

RUN_NAME=${RUN_NAME:-stage2_normalizedMSE_meterSelect}
OUT_DIR=${OUT_DIR:-$WORK/runs/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

BATCH_SIZE=${BATCH_SIZE:-4}
ACCUM_ITER=${ACCUM_ITER:-4}
EPOCHS=${EPOCHS:-120}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-7}
PATIENCE=${PATIENCE:-30}
GPU_ID=${GPU_ID:-0}
NUM_WORKERS=${NUM_WORKERS:-1}

OPTIMIZATION_LOSS=${OPTIMIZATION_LOSS:-normalized_mse}
BEST_METRIC=${BEST_METRIC:-val_mae_m_mask}
EARLY_STOP_METRIC=${EARLY_STOP_METRIC:-val_mae_m_mask}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.001}
EARLY_STOP_WARMUP_EPOCHS=${EARLY_STOP_WARMUP_EPOCHS:-0}
BASELINE_EVAL_BEFORE_TRAINING=${BASELINE_EVAL_BEFORE_TRAINING:-1}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}

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
FRESH_RUN=${FRESH_RUN:-1}
OVERWRITE_STAGE2=${OVERWRITE_STAGE2:-0}

if [[ "$OPTIMIZATION_LOSS" != "normalized_mse" ]]; then
  echo "[ERROR] D034 Stage 2 requires OPTIMIZATION_LOSS=normalized_mse, got $OPTIMIZATION_LOSS" >&2
  exit 2
fi
if [[ "$BEST_METRIC" != "val_mae_m_mask" || "$EARLY_STOP_METRIC" != "val_mae_m_mask" ]]; then
  echo "[ERROR] D034 safety rule requires BEST_METRIC=EARLY_STOP_METRIC=val_mae_m_mask" >&2
  exit 2
fi
if [[ "$BASELINE_EVAL_BEFORE_TRAINING" != "1" && "$BASELINE_EVAL_BEFORE_TRAINING" != "true" && "$BASELINE_EVAL_BEFORE_TRAINING" != "TRUE" ]]; then
  echo "[ERROR] Stage-1 checkpoint protection requires BASELINE_EVAL_BEFORE_TRAINING=1" >&2
  exit 2
fi

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

# Do not silently mix a previous Stage-2 run with a new formal run.
if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  if [[ "$OVERWRITE_STAGE2" == "1" ]]; then
    echo "[OVERWRITE] Removing existing Stage-2 output: $OUT_DIR"
    rm -rf "$OUT_DIR"
  elif [[ "$FRESH_RUN" == "1" ]]; then
    echo "[ERROR] Stage-2 output already exists and is non-empty: $OUT_DIR" >&2
    echo "Use FRESH_RUN=0 to resume, choose a new RUN_NAME, or explicitly set OVERWRITE_STAGE2=1." >&2
    exit 4
  fi
fi
mkdir -p "$OUT_DIR" "$LOG_DIR"

for f in \
  "$CODE/main_pretrain_dem_Stage2NormMeterSelect_D034_20260713.py" \
  "$CODE/engine_pretrain_Stage2NormMeterSelect_D034_20260713.py" \
  "$STAGE1_CKPT" \
  "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/val_tiles.txt" \
  "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/val_hidden.txt" \
  "$SPLIT_DIR/train_loss.txt" "$SPLIT_DIR/val_loss.txt"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required file: $f" >&2
    exit 2
  fi
done

echo "============================================================"
echo "D034 Stage-2 normalized-MSE optimization with meter-MAE checkpoint protection"
date
echo "HOST=$(hostname)"
echo "STAGE1_CKPT=$STAGE1_CKPT"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "OPTIMIZATION_LOSS=$OPTIMIZATION_LOSS"
echo "BEST_METRIC=$BEST_METRIC"
echo "EARLY_STOP_METRIC=$EARLY_STOP_METRIC"
echo "BASELINE_EPOCH=-1 (untouched Stage-1 meter checkpoint)"
echo "STAGE2_LR=$LR"
echo "MIN_LR=$MIN_LR"
echo "WARMUP_EPOCHS=$WARMUP_EPOCHS"
echo "EPOCHS=$EPOCHS"
echo "PATIENCE=$PATIENCE"
echo "FRESH_RUN=$FRESH_RUN"
echo "OVERWRITE_STAGE2=$OVERWRITE_STAGE2"
echo "SAFETY=checkpoint-best remains Stage 1 unless Stage 2 improves validation meter MAE"
echo "============================================================"

# Reproducibility and provenance.
printf '%s\n' "$STAGE1_CKPT" > "$OUT_DIR/stage1_source_checkpoint.txt"
sha256sum "$STAGE1_CKPT" | tee "$OUT_DIR/stage1_source_checkpoint_sha256.txt"
if [[ -f "$(dirname "$STAGE1_CKPT")/best_summary.json" ]]; then
  cp -pf "$(dirname "$STAGE1_CKPT")/best_summary.json" "$OUT_DIR/stage1_best_summary.json"
fi
if [[ -f "$(dirname "$STAGE1_CKPT")/baseline_summary.json" ]]; then
  cp -pf "$(dirname "$STAGE1_CKPT")/baseline_summary.json" "$OUT_DIR/stage1_baseline_summary.json"
fi

sha256sum \
  "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/val_tiles.txt" \
  "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/val_hidden.txt" \
  "$SPLIT_DIR/train_loss.txt" "$SPLIT_DIR/val_loss.txt" \
  | tee "$OUT_DIR/input_split_sha256.txt"

cp -pf "$SPLIT_DIR"/train_tiles.txt "$OUT_DIR/input_train_tiles.txt"
cp -pf "$SPLIT_DIR"/val_tiles.txt "$OUT_DIR/input_val_tiles.txt"
cp -pf "$SPLIT_DIR"/train_hidden.txt "$OUT_DIR/input_train_hidden.txt"
cp -pf "$SPLIT_DIR"/val_hidden.txt "$OUT_DIR/input_val_hidden.txt"
cp -pf "$SPLIT_DIR"/train_loss.txt "$OUT_DIR/input_train_loss.txt"
cp -pf "$SPLIT_DIR"/val_loss.txt "$OUT_DIR/input_val_loss.txt"

python "$CODE/main_pretrain_dem_Stage2NormMeterSelect_D034_20260713.py" --help > "$OUT_DIR/help.txt" 2>&1 || true
for key in \
  optimization_loss normalized_mse best_metric early_stop_metric \
  val_mae_m_mask baseline_eval_before_training init_ckpt \
  train_hidden_list val_hidden_list train_loss_list val_loss_list; do
  if ! grep -q -- "$key" "$OUT_DIR/help.txt"; then
    echo "[ERROR] Stage-2 main script lacks expected argument: $key" >&2
    exit 3
  fi
done

echo "=== CUDA check ==="
python - <<'PY'
import sys
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), "count", torch.cuda.device_count())
if not torch.cuda.is_available():
    sys.exit(10)
free_b, total_b = torch.cuda.mem_get_info()
print(f"[GPU] {torch.cuda.get_device_name(0)} free={free_b/(1024**3):.2f} GiB total={total_b/(1024**3):.2f} GiB")
if free_b / (1024**3) < 6.0:
    sys.exit(11)
PY
nvidia-smi || true

RESUME_ARGS=()
if [[ "$FRESH_RUN" != "1" ]]; then
  CKPT=$(find "$OUT_DIR" -maxdepth 1 -type f -regextype posix-extended \
    -regex '.*/checkpoint-[0-9]{4}\.pth' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk 'NR==1{print $2}' || true)
  if [[ -n "${CKPT:-}" ]]; then
    echo "[RESUME] Stage-2 checkpoint: $CKPT"
    RESUME_ARGS=(--resume "$CKPT")
  else
    echo "[RESUME] No Stage-2 epoch checkpoint found; starting from Stage 1."
  fi
else
  echo "[RESUME] FRESH_RUN=1; start a new Stage-2 run from Stage 1."
fi

PYTHONUNBUFFERED=1 python -u main_pretrain_dem_Stage2NormMeterSelect_D034_20260713.py \
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
  --init_ckpt "$STAGE1_CKPT" \
  --freeze_encoder \
  --freeze_last_n_encoder_blocks 0 \
  --bottleneck_norm "$BOTTLENECK_NORM" \
  --optimization_loss normalized_mse \
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
  --best_metric val_mae_m_mask \
  --early_stop_metric val_mae_m_mask \
  --early_stop_patience "$PATIENCE" \
  --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
  --early_stop_warmup_epochs "$EARLY_STOP_WARMUP_EPOCHS" \
  --baseline_eval_before_training \
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
  "${RESUME_ARGS[@]}"

echo "============================================================"
echo "DONE D034 Stage 2"
echo "RUN=$OUT_DIR"
date
echo "============================================================"
