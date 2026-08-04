#!/usr/bin/env bash
set -euo pipefail

# F070 RELAX MeterOnly-defreeze backend
#
# Purpose:
#   Continue from the formal D001c MeterOnly checkpoint-best.pth,
#   keep the decoder trainable, and additionally unfreeze the LAST ONE
#   encoder block.
#
# Scientific safety:
#   * objective            = exact meter-space MAE
#   * best/early-stop      = val_mae_m_mask
#   * epoch -1 baseline    = untouched frozen-encoder MeterOnly checkpoint
#   * checkpoint-best stays at epoch -1 unless defreeze improves meter MAE
#
# The unified trainer implements:
#   --freeze_encoder --freeze_last_n_encoder_blocks 1
# as "freeze the encoder first, then make the last encoder block trainable."

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
WORK=${WORK:-$RELAX_ROOT}
RESULTS_ROOT=${RESULTS_ROOT:-$RELAX_ROOT/results}
CODE=${CODE:-$ROOT/mae_Retrain}

TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2_D001c_AnyVisiblePatch/Tiles_1m}
SPLIT_DIR=${SPLIT_DIR:?Please set SPLIT_DIR}
INIT_CKPT=${INIT_CKPT:?Please set INIT_CKPT to the formal D001c MeterOnly checkpoint-best.pth}

RUN_NAME=${RUN_NAME:-F070_meterOnly_defreezeLast1_D001c_AnyVisiblePatch}
OUT_DIR=${OUT_DIR:-$RESULTS_ROOT/MeterOnly_DefreezeLast1/runs/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$OUT_DIR/tb}

BATCH_SIZE=${BATCH_SIZE:-4}
ACCUM_ITER=${ACCUM_ITER:-4}
EPOCHS=${EPOCHS:-80}
LR=${LR:-1e-6}
MIN_LR=${MIN_LR:-1e-8}
PATIENCE=${PATIENCE:-20}
GPU_ID=${GPU_ID:-0}
NUM_WORKERS=${NUM_WORKERS:-1}

TRAINABLE_LAST_N_ENCODER_BLOCKS=${TRAINABLE_LAST_N_ENCODER_BLOCKS:-1}

OPTIMIZATION_LOSS=${OPTIMIZATION_LOSS:-meter_mae}
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
VIS_EVERY=${VIS_EVERY:-10}
VIS_N=${VIS_N:-10}
STATS_MAX_FILES=${STATS_MAX_FILES:-1000}

FRESH_RUN=${FRESH_RUN:-1}
OVERWRITE_DEFREEZE=${OVERWRITE_DEFREEZE:-0}

if [[ "$TRAINABLE_LAST_N_ENCODER_BLOCKS" != "1" ]]; then
  echo "[ERROR] F070 formal experiment requires TRAINABLE_LAST_N_ENCODER_BLOCKS=1" >&2
  exit 2
fi
if [[ "$OPTIMIZATION_LOSS" != "meter_mae" ]]; then
  echo "[ERROR] F070 requires OPTIMIZATION_LOSS=meter_mae" >&2
  exit 2
fi
if [[ "$BEST_METRIC" != "val_mae_m_mask" || "$EARLY_STOP_METRIC" != "val_mae_m_mask" ]]; then
  echo "[ERROR] F070 requires BEST_METRIC=EARLY_STOP_METRIC=val_mae_m_mask" >&2
  exit 2
fi
if [[ "$BASELINE_EVAL_BEFORE_TRAINING" != "1" && \
      "$BASELINE_EVAL_BEFORE_TRAINING" != "true" && \
      "$BASELINE_EVAL_BEFORE_TRAINING" != "TRUE" ]]; then
  echo "[ERROR] F070 requires BASELINE_EVAL_BEFORE_TRAINING=1" >&2
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

if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  if [[ "$OVERWRITE_DEFREEZE" == "1" ]]; then
    echo "[OVERWRITE] Removing existing output: $OUT_DIR"
    rm -rf "$OUT_DIR"
  elif [[ "$FRESH_RUN" == "1" ]]; then
    echo "[ERROR] Output already exists and is non-empty: $OUT_DIR" >&2
    echo "Use FRESH_RUN=0 to resume, a new RUN_NAME, or OVERWRITE_DEFREEZE=1." >&2
    exit 4
  fi
fi
mkdir -p "$OUT_DIR" "$LOG_DIR"

for f in \
  "$CODE/main_pretrain_dem_unified_relax.py" \
  "$CODE/engine_pretrain_unified_relax.py" \
  "$INIT_CKPT" \
  "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/val_tiles.txt" \
  "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/val_hidden.txt" \
  "$SPLIT_DIR/train_loss.txt" "$SPLIT_DIR/val_loss.txt"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing required file: $f" >&2; exit 2; }
done

echo "============================================================"
echo "F070 MeterOnly defreeze-last-1 backend"
date
echo "HOST=$(hostname)"
echo "CODE=$CODE"
echo "TILE_ROOT=$TILE_ROOT"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "INIT_CKPT=$INIT_CKPT"
echo "OUT_DIR=$OUT_DIR"
echo "OPTIMIZATION_LOSS=$OPTIMIZATION_LOSS"
echo "BEST_METRIC=$BEST_METRIC"
echo "EARLY_STOP_METRIC=$EARLY_STOP_METRIC"
echo "TRAINABLE_LAST_N_ENCODER_BLOCKS=$TRAINABLE_LAST_N_ENCODER_BLOCKS"
echo "BASELINE_EPOCH=-1 (untouched frozen MeterOnly)"
echo "LR=$LR"
echo "MIN_LR=$MIN_LR"
echo "EPOCHS=$EPOCHS"
echo "PATIENCE=$PATIENCE"
echo "WARMUP_EPOCHS=$WARMUP_EPOCHS"
echo "============================================================"

printf '%s\n' "$INIT_CKPT" > "$OUT_DIR/source_meterOnly_checkpoint.txt"
sha256sum "$INIT_CKPT" | tee "$OUT_DIR/source_meterOnly_checkpoint_sha256.txt"

SOURCE_DIR=$(dirname "$INIT_CKPT")
for candidate in best_summary.json baseline_summary.json baseline_val.json args.json log.txt history.csv; do
  if [[ -f "$SOURCE_DIR/$candidate" ]]; then
    cp -pf "$SOURCE_DIR/$candidate" "$OUT_DIR/source_meterOnly_${candidate}"
  fi
done

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

python "$CODE/main_pretrain_dem_unified_relax.py" --help > "$OUT_DIR/help.txt" 2>&1 || true
for key in \
  optimization_loss meter_mae best_metric early_stop_metric \
  val_mae_m_mask baseline_eval_before_training init_ckpt \
  freeze_encoder freeze_last_n_encoder_blocks \
  train_hidden_list val_hidden_list train_loss_list val_loss_list; do
  if ! grep -q -- "$key" "$OUT_DIR/help.txt"; then
    echo "[ERROR] Unified trainer lacks expected argument: $key" >&2
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
    echo "[RESUME] Defreeze checkpoint: $CKPT"
    RESUME_ARGS=(--resume "$CKPT")
  else
    echo "[RESUME] No defreeze epoch checkpoint; restart from frozen MeterOnly."
  fi
else
  echo "[RESUME] FRESH_RUN=1; start from frozen MeterOnly."
fi

PYTHONUNBUFFERED=1 python -u "$CODE/main_pretrain_dem_unified_relax.py" \
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
  --init_ckpt "$INIT_CKPT" \
  --freeze_encoder \
  --freeze_last_n_encoder_blocks "$TRAINABLE_LAST_N_ENCODER_BLOCKS" \
  --bottleneck_norm "$BOTTLENECK_NORM" \
  --optimization_loss meter_mae \
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
echo "DONE F070 MeterOnly defreeze-last-1"
echo "RUN=$OUT_DIR"
date
echo "============================================================"
