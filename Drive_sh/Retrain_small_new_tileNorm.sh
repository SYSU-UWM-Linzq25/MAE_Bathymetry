#!/bin/bash
#SBATCH -J mae_dem_small_v2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_dem_small_v2_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_dem_small_v2_%j.out
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
OUT=$ROOT/runs/Small_tilenorm_viscorr_336
mkdir -p "$OUT"
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_LIST="$ROOT/splits/smoke_small_1000/global/train.txt"
VAL_LIST="$ROOT/splits/smoke_small_1000/global/val.txt"
# TEST_LIST="$ROOT/splits/smoke_small_1000/global/test.txt"
# KY_HOLDOUT="$ROOT/splits/smoke_small_1000/global/holdout_KY.txt"

BOTTLENECK_NORM=${BOTTLENECK_NORM:-inst1d}
MASK_RATIO=${MASK_RATIO:-0.75}
EPOCHS=${EPOCHS:-400}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-8}

python -u $CODEDIR/main_pretrain_dem.py \
  --data_root "$ROOT" \
  --train_list "$TRAIN_LIST" \
  --val_list   "$VAL_LIST" \
  --output_dir "$OUT" \
  --log_dir    "$OUT" \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --in_chans 1 \
  --mask_ratio "$MASK_RATIO" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --norm_method meanstd \
  --tile_norm \
  --tile_norm_eps 1e-3 \
  --bottleneck_norm "$BOTTLENECK_NORM" \
  --loss_mode mse \
  --eval_rmse \
  --best_metric val_rmse_m_mask_viscorr \
  --early_stop_metric val_rmse_m_mask_viscorr \
  --early_stop_patience 30 \
  --early_stop_min_delta 0.05 \
  --early_stop_warmup_epochs 50 \
  --early_stop_start_threshold 0.1 \
  --vis_every 20 \
  --vis_n 10 \
  --plot_every 1 \
  --stats_max_files 5000 \
  --nodata -9999
