#!/bin/bash
#SBATCH -J mae_dem_Eval_hist
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
##SBATCH -t 02:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_dem_Eval_hist_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_dem_Eval_hist_%j.out
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail
module purge || true

source /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count())"
nvidia-smi || true

CODEDIR=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/mae_Retrain
ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"

# 旧 upstream 模型
OUT=$ROOT/runs/Small_tilenorm_viscorr_336

# 新的 histogram 结果目录，避免覆盖 median 结果
OUTEva=$ROOT/eval_KY_holdout_ratio075_histogram
mkdir -p "$OUTEva"

python $CODEDIR/mae_evaluate_dem_meters_topk.py \
  --ckpt       $OUT/checkpoint-best.pth \
  --list       $ROOT/splits/smoke_small_1000/global/holdout_KY.txt \
  --norm_json  $OUT/norm_stats_train.json \
  --output_dir $OUTEva \
  --mask_ratio 0.75 \
  --batch_size 16 \
  --num_workers 8 \
  --nodata -9999 \
  --topk 200 \
  --amp \
  --tile_norm \
  --tile_norm_eps 1e-3 \
  --bottleneck_norm inst1d \
  --loss_mode mse \
  --postproc histogram
