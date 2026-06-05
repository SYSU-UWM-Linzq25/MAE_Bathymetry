#!/bin/bash
#SBATCH -J s4_fullnorm_dec
#SBATCH -p gpu
#SBATCH -w execute-3001
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 7-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/train_s4_fullnorm_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/train_s4_fullnorm_%j.out
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  bash  $0 <VAL_RIVER_NAME>"
  echo "  sbatch $0 <VAL_RIVER_NAME>"
  echo
  echo "Example:"
  echo "  sbatch $0 MD_PotomacRiver_Bathy_2019"
  echo
  echo "This script expects the split folder:"
  echo "  Downstream_Task_Bathy/splits/bathy_lcc_1m_holdout_<VAL_RIVER_NAME>"
  exit 1
fi

VAL_RIVER="$1"
SAFE_VAL=$(echo "$VAL_RIVER" | sed 's/[^A-Za-z0-9_]/_/g')

echo "=== STAGE4 FULL-TILE-NORM HOLDOUT TRAIN JOB ${SLURM_JOB_ID:-local} on $(hostname) ==="
date

# ==============================
# 1. Environment
# ==============================
module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

GPU_ID=${GPU_ID:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

which python
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count()); print('name0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
nvidia-smi || true

python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available inside this Slurm job.")
    sys.exit(10)

free_b, total_b = torch.cuda.mem_get_info()
free_gb = free_b / (1024 ** 3)
total_gb = total_b / (1024 ** 3)

print(f"[GPU CHECK] device={torch.cuda.current_device()} "
      f"name={torch.cuda.get_device_name(0)} "
      f"free={free_gb:.2f} GiB total={total_gb:.2f} GiB")

# The current frozen-encoder job normally uses only a few GiB.
# Fail early when the assigned GPU is already nearly full.
if free_gb < 6.0:
    print("[ERROR] Assigned GPU has less than 6 GiB free. "
          "This usually means the GPU was not reserved correctly or is shared with another job.")
    sys.exit(11)
PY

# ==============================
# 2. Paths and experiment name
# ==============================
ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
CODE=$ROOT/mae_Retrain
WORK=$ROOT/Downstream_Task_Bathy

SPLIT=$WORK/splits/bathy_lcc_1m_holdout_${SAFE_VAL}
BATH=$ROOT/Data/Tiles_for_Training_1m/1m_Tiles
# This folder currently stores the simplified final mask tiles produced from:
#   final_mask = LCC & bathy_valid
# The folder name is kept as TilesMask_for_Training_1m for compatibility.
MASK=$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles

UP_CKPT=$ROOT/Upstream_Model_ReTrain/runs/Small_tilenorm_viscorr_336/checkpoint-best.pth
ENTRY=$CODE/main_pretrain_dem.py

STD_SCALE=${STD_SCALE:-1.0}
END_EPOCH=${END_EPOCH:-400}
# Actual NoData value used by the extracted 1 m bathy+3DEP tiles.
NODATA=${NODATA:-"-999999"}
NORM_MODE=full_tile
RUN_NAME=stage4_bathy_finalmask_holdout_${SAFE_VAL}_exact_freeze_decoder_tilenormFull_std${STD_SCALE//./p}_1m_e${END_EPOCH}
OUT=$WORK/runs/$RUN_NAME

mkdir -p "$OUT" "$WORK/logs"

export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

echo "ROOT=$ROOT"
echo "CODE=$CODE"
echo "ENTRY=$ENTRY"
echo "VAL_RIVER=$VAL_RIVER"
echo "SAFE_VAL=$SAFE_VAL"
echo "SPLIT=$SPLIT"
echo "BATH=$BATH"
echo "MASK=$MASK"
echo "UP_CKPT=$UP_CKPT"
echo "STD_SCALE=$STD_SCALE"
echo "NODATA=$NODATA"
echo "NORM_MODE=$NORM_MODE"
echo "OUT=$OUT"

# ==============================
# 3. Basic checks
# ==============================
for f in \
  "$ENTRY" \
  "$SPLIT/train.txt" \
  "$SPLIT/val.txt" \
  "$SPLIT/train_masks.txt" \
  "$SPLIT/val_masks.txt" \
  "$UP_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required file: $f"
    exit 2
  fi
done

python "$ENTRY" --help > "$OUT/help.txt" 2>&1 || true
for key in \
  lcc_mask_path train_lcc_list val_lcc_list lcc_mask_mode \
  init_ckpt freeze_encoder bottleneck_norm plot_every vis_every \
  tile_norm_std_scale nodata; do
  if ! grep -q -- "$key" "$OUT/help.txt"; then
    echo "[ERROR] $ENTRY does not support expected argument containing: $key"
    echo "        Did you point ENTRY to the modified mae_Retrain/main_pretrain_dem.py?"
    echo "        See $OUT/help.txt"
    exit 3
  fi
done

echo "=== Split counts ==="
wc -l "$SPLIT/train.txt" "$SPLIT/train_masks.txt"
wc -l "$SPLIT/val.txt" "$SPLIT/val_masks.txt"

echo "=== Holdout sanity check ==="
echo "[Check] val files not containing VAL_RIVER; should print nothing:"
grep -v "$VAL_RIVER" "$SPLIT/val.txt" | head || true
echo "[Check] train files containing VAL_RIVER; should print nothing:"
grep "$VAL_RIVER" "$SPLIT/train.txt" | head || true

# ==============================
# 4. Resume logic
# ==============================
CKPT=$(find "$OUT" -maxdepth 1 -type f -name 'checkpoint-*.pth' ! -name 'checkpoint-best.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1{print $2}' || true)
if [[ -z "${CKPT:-}" && -f "$OUT/checkpoint-best.pth" ]]; then
  CKPT="$OUT/checkpoint-best.pth"
fi

RESUME_ARGS=()
if [[ -n "${CKPT:-}" ]]; then
  echo "[RESUME] Found checkpoint: $CKPT"
  RESUME_ARGS=(--resume "$CKPT")
else
  echo "[RESUME] No checkpoint found. Starting from upstream init_ckpt only."
fi

# ==============================
# 5. Training
# ==============================
# Full-tile normalization experiment:
# mean/std are computed from all finite pixels in the input tile.
# NoData pixels are excluded. The denominator is full_tile_std * STD_SCALE.
PYTHONUNBUFFERED=1 python -u "$ENTRY" \
  --device cuda \
  --data_root "$BATH" \
  --train_list "$SPLIT/train.txt" \
  --val_list "$SPLIT/val.txt" \
  --lcc_mask_path "$MASK" \
  --train_lcc_list "$SPLIT/train_masks.txt" \
  --val_lcc_list "$SPLIT/val_masks.txt" \
  --output_dir "$OUT" \
  --log_dir "$OUT/tb" \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --in_chans 1 \
  --nodata "$NODATA" \
  --batch_size 4 \
  --accum_iter 4 \
  --epochs "$END_EPOCH" \
  --num_workers 8 \
  --bottleneck_norm inst1d \
  --pin_mem \
  --tile_norm \
  --tile_norm_eps 1e-3 \
  --tile_norm_std_scale "$STD_SCALE" \
  --init_ckpt "$UP_CKPT" \
  --freeze_encoder \
  --freeze_last_n_encoder_blocks 0 \
  --lcc_mask_mode exact \
  --loss_on_lcc_only \
  --eval_rmse \
  --best_metric val_rmse_m_mask \
  --early_stop_metric val_rmse_m_mask \
  --early_stop_patience 60 \
  --early_stop_min_delta 0.001 \
  --early_stop_warmup_epochs 20 \
  --plot_every 1 \
  --vis_every 20 \
  --vis_n 10 \
  --stats_max_files 1000 \
  --min_lcc_patch_ratio 0.0001 \
  --max_lcc_patch_ratio 0.80 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --warmup_epochs 5 \
  "${RESUME_ARGS[@]}"

date
echo "=== STAGE4 FULL-TILE-NORM HOLDOUT TRAIN DONE ==="
echo "RUN=$OUT"
