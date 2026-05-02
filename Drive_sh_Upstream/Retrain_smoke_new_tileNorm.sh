#!/bin/bash
#SBATCH -J mae_dem_smoke_v2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -t 02:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_dem_smoke_v2_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_dem_smoke_v2_%j.out
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail
module purge || true

source /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# Keep one visible GPU unless the scheduler already set one.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count())"
nvidia-smi || true

CODEDIR=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/mae_Retrain
ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain
OUT=$ROOT/runs/smoke_tilenorm_viscorr_336
mkdir -p "$OUT"
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"

# Toggle these from sbatch command line if needed, e.g.
# sbatch --export=ALL,BOTTLENECK_NORM=none,MASK_RATIO=0.6 Retrain_smoke_new_tileNorm.sh
BOTTLENECK_NORM=${BOTTLENECK_NORM:-inst1d}
MASK_RATIO=${MASK_RATIO:-0.75}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-8}

# Auto-build a small smoke split if it does not exist yet.
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
  --early_stop_patience 0 \
  --vis_every 1 \
  --vis_n 8 \
  --plot_every 1 \
  --stats_max_files 5000 \
  --nodata -9999
