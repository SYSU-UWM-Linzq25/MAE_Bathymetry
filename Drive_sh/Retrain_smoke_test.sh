#!/bin/bash
#SBATCH -J mae_dem_smoke
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -t 02:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_dem_smoke_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_dem_smoke_%j.out
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail
module purge || true

source /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# 和你旧脚本一致：手动锁一个 GPU（避免 gpu 分区不按 GRES 隔离）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count())"
nvidia-smi || true

CODEDIR=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/mae_Retrain
ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain
OUT=$ROOT/runs/smoke_meanstd_336
mkdir -p "$OUT"
#export PYTHONPATH=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/mae_Retrain:$PYTHONPATH
export PYTHONPATH="/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/mae_Retrain${PYTHONPATH:+:$PYTHONPATH}"

# 你当前 splits 目录里未必有 splits/smoke/（脚本在用它），这里顺手自动生成一个小样本 smoke split
if [ ! -f "$ROOT/splits/smoke/train.txt" ]; then
  mkdir -p "$ROOT/splits/smoke"
  head -n 2000 "$ROOT/splits/global_default_train.txt" > "$ROOT/splits/smoke/train.txt"
  head -n 200  "$ROOT/splits/global_default_val.txt"   > "$ROOT/splits/smoke/val.txt"
  head -n 200  "$ROOT/splits/global_default_test.txt"  > "$ROOT/splits/smoke/test.txt"
fi

python -u $CODEDIR/main_pretrain_dem.py \
  --data_root "$ROOT" \
  --train_list "$ROOT/splits/smoke/train.txt" \
  --val_list   "$ROOT/splits/smoke/val.txt" \
  --test_list  "$ROOT/splits/smoke/test.txt" \
  --output_dir "$OUT" \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --in_chans 1 \
  --mask_ratio 0.75 \
  --epochs 2 \
  --batch_size 16 \
  --num_workers 8 \
  --norm_method meanstd \
  --eval_rmse \
  --stats_max_files 5000 \
  --nodata -9999


