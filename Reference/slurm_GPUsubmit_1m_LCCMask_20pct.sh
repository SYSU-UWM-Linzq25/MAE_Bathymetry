#!/bin/bash
#SBATCH -J 1m_20pct_mae_LCCMask
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
##SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/log/mae_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/log/mae_%j.out
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "=== JOB ${SLURM_JOB_ID} on $(hostname) ==="
echo "PWD=$(pwd)"
date

# --- conda env ---
module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn

# --- make sure conda libs first (avoid system CUDA libs interfering) ---
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# --- lock to 1 GPU (gpu partition may not enforce GRES isolation) ---
# default use GPU 1 to reduce collision with others; change to 0 if you want
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
python -c "import torch; print('torch', torch.__version__, 'built_cuda', torch.version.cuda); print('cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count()); print('name0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
nvidia-smi || true

# --- paths ---
DATA="/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Data/Tiles_for_Training_1m/1m_Tiles"
LCCDATA="/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Data/TilesMask_for_Training_1m/1m_Tiles"
LOGDIR="/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/log"
OUTROOT="/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/output/LCCMask_20pct"
OUT="${OUTROOT}/1m_MAE"
mkdir -p "$OUT"

END_EPOCH="${END_EPOCH:-100}"

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -n 1 || true)
RESUME_ARGS=()
if [ -n "${CKPT:-}" ]; then
  RESUME_ARGS=(--resume "$CKPT")
fi

echo "DATA=$DATA"
echo "LCCDATA=$LCCDATA"
echo "OUT=$OUT"
echo "LOGDIR=$LOGDIR"

echo "=== HOST $(hostname) ==="
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count())"

PYTHONUNBUFFERED=1 python -u mae_LCCMask/main_pretrain.py \
  --device cuda \
  --data_path "$DATA" \
  --lcc_mask_path "$LCCDATA" \
  --output_dir "$OUT" \
  --log_dir /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/log \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --batch_size 4 \
  --accum_iter 16 \
  --epochs "$END_EPOCH" \
  --num_workers 8 \
  --pin_mem \
  --mask_ratio 0.20 \
  --loss_on_lcc_only \
  --lcc_priority 10.0 
  "${RESUME_ARGS[@]}"

date
echo "=== DONE JOB ${SLURM_JOB_ID} ==="

