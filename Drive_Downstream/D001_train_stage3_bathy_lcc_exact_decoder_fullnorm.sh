#!/bin/bash
#SBATCH -J bathy_lcc_exact_dec
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 7-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/train_exact_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/train_exact_%j.out
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "=== TRAIN EXACT LCC JOB ${SLURM_JOB_ID:-local} on $(hostname) ==="
date

# ==============================
# 1. Environment
# ==============================
module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# gpu partition may not enforce GRES isolation; override manually if needed:
# CUDA_VISIBLE_DEVICES=0 sbatch D001_train_stage3_bathy_lcc_exact_decoder_fullnorm.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count()); print('name0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
nvidia-smi || true

# ==============================
# 2. Paths
# ==============================
ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
CODE=$ROOT/mae_Retrain
WORK=$ROOT/Downstream_Task_Bathy
SPLIT=$WORK/splits/bathy_lcc_1m_trainval_seed20260428

BATH=$ROOT/Data/Tiles_for_Training_1m/1m_Tiles
LCC=$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles
UP_CKPT=$ROOT/Upstream_Model_ReTrain/runs/Small_tilenorm_viscorr_336/checkpoint-best.pth

OUT=$WORK/runs/stage3_bathy_lcc_exact_freeze_decoder_fullnorm_1m_e400
ENTRY=$CODE/main_pretrain_dem.py

mkdir -p "$OUT" "$WORK/logs"

echo "ROOT=$ROOT"
echo "CODE=$CODE"
echo "ENTRY=$ENTRY"
echo "SPLIT=$SPLIT"
echo "BATH=$BATH"
echo "LCC=$LCC"
echo "UP_CKPT=$UP_CKPT"
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
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing required file: $f"
    exit 2
  fi
done

python "$ENTRY" --help > "$OUT/help.txt" 2>&1 || true
for key in lcc_mask_path train_lcc_list val_lcc_list lcc_mask_mode init_ckpt freeze_encoder bottleneck_norm plot_every vis_every; do
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

# ==============================
# 4. Resume logic
#    Prefer latest numbered checkpoint for interrupted training.
#    Fall back to checkpoint-best.pth only if no numbered checkpoint exists.
# ==============================
END_EPOCH=${END_EPOCH:-400}

CKPT=$(find "$OUT" -maxdepth 1 -type f -name 'checkpoint-*.pth' ! -name 'checkpoint-best.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1{print $2}' || true)
if [ -z "${CKPT:-}" ] && [ -f "$OUT/checkpoint-best.pth" ]; then
  CKPT="$OUT/checkpoint-best.pth"
fi

RESUME_ARGS=()
if [ -n "${CKPT:-}" ]; then
  echo "[RESUME] Found checkpoint: $CKPT"
  RESUME_ARGS=(--resume "$CKPT")
else
  echo "[RESUME] No checkpoint found. Starting from init_ckpt only."
fi

# ==============================
# 5. Formal Stage3 / Task3 training
#    Full train/val split, exact LCC patch mask, full-tile tile normalization.
#    Encoder frozen; decoder only is trained.
# ==============================
PYTHONUNBUFFERED=1 python -u "$ENTRY" \
  --device cuda \
  --data_root "$BATH" \
  --train_list "$SPLIT/train.txt" \
  --val_list "$SPLIT/val.txt" \
  --lcc_mask_path "$LCC" \
  --train_lcc_list "$SPLIT/train_masks.txt" \
  --val_lcc_list "$SPLIT/val_masks.txt" \
  --output_dir "$OUT" \
  --log_dir "$OUT/tb" \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --in_chans 1 \
  --batch_size 4 \
  --accum_iter 4 \
  --epochs "$END_EPOCH" \
  --num_workers 8 \
  --bottleneck_norm inst1d \
  --pin_mem \
  --tile_norm \
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
echo "=== TRAIN EXACT LCC FULLNORM DONE ==="
