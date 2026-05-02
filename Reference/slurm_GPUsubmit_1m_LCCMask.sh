#!/bin/bash
#SBATCH -J 1m_mae_LCCMask
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -t 7-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/log/mae_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/log/mae_%j.out
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

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
OUTROOT="/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/output/LCCMask_1m"
OUT="${OUTROOT}/1m_job_${SLURM_JOB_ID}"
mkdir -p "$OUT"

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
  --epochs 400 \
  --num_workers 4 \
  --pin_mem \
  --loss_on_lcc_only \
  --lcc_priority 10.0 

date
echo "=== DONE JOB ${SLURM_JOB_ID} ==="

