#!/bin/bash
#SBATCH -J mae_dem_smoke_small
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
##SBATCH -t 02:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_dem_smoke_small_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_dem_smoke_small_%j.out
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail
module purge || true

source /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# 手动锁一个 GPU（避免 gpu 分区不按 GRES 隔离）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count())"
nvidia-smi || true

CODEDIR=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/mae_Retrain
ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain

# 建议单独一个输出目录，避免覆盖你正式训练的 runs
OUT=$ROOT/runs/Small_meanstd_336
mkdir -p "$OUT"

export PYTHONPATH="/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/mae_Retrain${PYTHONPATH:+:$PYTHONPATH}"

# 使用你新生成的 small split（KY 已从 train/val 排除）
TRAIN_LIST="$ROOT/splits/smoke_small_1000/global/train.txt"
VAL_LIST="$ROOT/splits/smoke_small_1000/global/val.txt"

# 可选：KY holdout（下游任务用，不参与本次上游 train/val）
# KY_HOLDOUT="$ROOT/splits/smoke_small_1000/global/holdout_KY.txt"

python -u $CODEDIR/main_pretrain_dem.py \
  --data_root "$ROOT" \
  --train_list "$TRAIN_LIST" \
  --val_list   "$VAL_LIST" \
  --output_dir "$OUT" \
  --log_dir    "$OUT" \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --in_chans 1 \
  --mask_ratio 0.75 \
  --epochs 400 \
  --batch_size 16 \
  --num_workers 8 \
  --norm_method meanstd \
  --eval_rmse \
  --best_metric val_rmse_m_mask \
  --early_stop_metric val_rmse_m_mask \
  --early_stop_patience 30 \
  --early_stop_min_delta 0.05 \
  --early_stop_warmup_epochs 50 \
  --early_stop_start_threshold 0.1 \
  --plot_every 1 \
  --stats_max_files 5000 \
  --nodata -9999

