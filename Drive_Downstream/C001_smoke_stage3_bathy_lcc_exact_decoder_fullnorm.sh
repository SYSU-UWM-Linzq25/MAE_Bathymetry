#!/bin/bash
#SBATCH -J smoke_bathy_lcc_exact
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/smoke_exact_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/smoke_exact_%j.out
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

echo "=== SMOKE EXACT LCC JOB ${SLURM_JOB_ID:-no_slurm} on $(hostname) ==="
date

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
WORK=$ROOT/Downstream_Task_Bathy
SPLIT=$WORK/splits/bathy_lcc_1m_trainval_seed20260428
BATH=$ROOT/Data/Tiles_for_Training_1m/1m_Tiles
LCC=$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles
UP_CKPT=$ROOT/Upstream_Model_ReTrain/runs/Small_tilenorm_viscorr_336/checkpoint-best.pth
OUT=$WORK/runs/smoke_stage3_bathy_lcc_exact_${SLURM_JOB_ID:-local}
mkdir -p "$OUT" "$WORK/logs"

CODE=$ROOT/mae_Retrain
ENTRY=$CODE/main_pretrain_dem.py

python "$ENTRY" --help > "$OUT/help.txt" 2>&1 || true
for key in lcc_mask_path train_lcc_list val_lcc_list lcc_mask_mode init_ckpt freeze_encoder; do
  if ! grep -q -- "$key" "$OUT/help.txt"; then
    echo "[ERROR] $ENTRY does not support expected argument containing: $key"
    echo "        Did you copy the modified files into the MAE-Topography code folder?"
    echo "        See $OUT/help.txt"
    exit 3
  fi
done

python - <<'PY'
import torch
print('torch', torch.__version__, 'built_cuda', torch.version.cuda)
print('cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count())
print('name0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
nvidia-smi || true

echo "UP_CKPT=$UP_CKPT"
echo "OUT=$OUT"
echo "SPLIT=$SPLIT"

PYTHONUNBUFFERED=1 python -u "$ENTRY" \
  --device cuda \
  --data_root "$BATH" \
  --train_list "$SPLIT/smoke_train.txt" \
  --val_list "$SPLIT/smoke_val.txt" \
  --lcc_mask_path "$LCC" \
  --train_lcc_list "$SPLIT/smoke_train_masks.txt" \
  --val_lcc_list "$SPLIT/smoke_val_masks.txt" \
  --output_dir "$OUT" \
  --log_dir "$OUT/tb" \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --in_chans 1 \
  --batch_size 2 \
  --accum_iter 1 \
  --epochs 2 \
  --num_workers 2 \
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
  --vis_every 0 \
  --stats_max_files 16 \
  --min_lcc_patch_ratio 0.0001 \
  --max_lcc_patch_ratio 0.80 \
  --lr 1e-4 \
  --warmup_epochs 0

date
echo "=== SMOKE EXACT LCC FULLNORM DONE ==="
