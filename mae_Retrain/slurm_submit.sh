#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks=10
#SBATCH -p gpu

#SBATCH --output=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/log/mae_initial-%j.out

set -e

cd /home/uwm/zequnlin
module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn

# 强制 conda 的库优先，避免系统 CUDA 插队
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"

python3 -c "import torch; print('torch ok', torch.__version__, torch.version.cuda)"

# make sure using GPU
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

python3 /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/mae/main_pretrain.py --input_size 336 --data_path /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Data/Center_Test --output_dir /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/output/Center_Test --log_dir /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/log
