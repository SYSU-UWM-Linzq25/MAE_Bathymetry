#!/bin/bash
#SBATCH -J mae_stage2_dec20
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
##SBATCH -t 12:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_stage2_dec20_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_stage2_dec20_%j.out
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

# ===== 上游模型 checkpoint =====
UP_OUT=$ROOT/runs/Small_tilenorm_viscorr_336
INIT_CKPT=$UP_OUT/checkpoint-best.pth

# ===== 任务2输出目录 =====
RUN_NAME=stage2_decoder_random20_fromValPool
OUT=$ROOT/runs/$RUN_NAME
mkdir -p "$OUT"

# ===== 用原来的 upstream val 作为数据池，再切成新的 train/val =====
SRC_POOL=$ROOT/splits/smoke_small_1000/global/val.txt
SPLIT_ROOT=$ROOT/splits/stage2_decoder_random20
mkdir -p "$SPLIT_ROOT"

STAGE2_TRAIN=$SPLIT_ROOT/stage2_train.txt
STAGE2_VAL=$SPLIT_ROOT/stage2_val.txt

SPLIT_SEED=42
VAL_FRAC=0.15

python - <<PY
from pathlib import Path
import random

src = Path("$SRC_POOL")
train_out = Path("$STAGE2_TRAIN")
val_out = Path("$STAGE2_VAL")
seed = $SPLIT_SEED
val_frac = $VAL_FRAC

lines = [x.strip() for x in src.read_text().splitlines() if x.strip()]
random.Random(seed).shuffle(lines)

n = len(lines)
n_val = max(1, int(round(n * val_frac)))
val = lines[:n_val]
train = lines[n_val:]

train_out.write_text("\n".join(train) + "\n")
val_out.write_text("\n".join(val) + "\n")

print(f"[SPLIT] total={n}, train={len(train)}, val={len(val)}")
print(f"[SPLIT] train -> {train_out}")
print(f"[SPLIT] val   -> {val_out}")
PY

# ===== 启动 Stage 2：freeze encoder, train decoder =====
python $CODEDIR/main_pretrain_dem.py \
  --data_root $ROOT \
  --train_list $STAGE2_TRAIN \
  --val_list   $STAGE2_VAL \
  --input_size 336 \
  --in_chans 1 \
  --nodata -9999 \
  --norm_method meanstd \
  --mask_ratio 0.20 \
  --tile_norm \
  --tile_norm_eps 1e-3 \
  --model mae_vit_large_patch16 \
  --bottleneck_norm inst1d \
  --loss_mode mse \
  --batch_size 16 \
  --epochs 80 \
  --accum_iter 1 \
  --blr 1e-3 \
  --weight_decay 0.05 \
  --warmup_epochs 5 \
  --num_workers 8 \
  --pin_mem \
  --eval_rmse \
  --best_metric val_rmse_m_mask \
  --early_stop_metric val_rmse_m_mask \
  --early_stop_patience 10 \
  --early_stop_min_delta 0.01 \
  --plot_every 1 \
  --vis_every 1 \
  --vis_n 10 \
  --output_dir $OUT \
  --log_dir $OUT \
  --init_ckpt $INIT_CKPT \
  --freeze_encoder \
  --freeze_last_n_encoder_blocks 0 \
  --seed 42
