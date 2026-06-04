#!/bin/bash
#SBATCH -J eval_s3_val_lcc
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -t 08:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/eval_s3_val_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/eval_s3_val_%j.out
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail
module purge || true

source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
CODE=$ROOT/mae_Retrain
WORK=$ROOT/Downstream_Task_Bathy
SPLIT=$WORK/splits/bathy_lcc_1m_trainval_seed20260428
BATH=$ROOT/Data/Tiles_for_Training_1m/1m_Tiles
LCC=$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles
RUN=$WORK/runs/stage3_bathy_lcc_exact_freeze_decoder_fullnorm_1m_e400
CKPT=$RUN/checkpoint-best.pth
EVAL_SCRIPT=$WORK/scripts/E001_evaluate_stage3_lcc_exact_per_tile.py
OUT=$WORK/eval_stage3_bathy_lcc_exact_best/val_best_fullnorm_top100
mkdir -p "$OUT" "$WORK/logs"

export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Stage3 LCC exact per-tile VAL evaluation ${SLURM_JOB_ID:-local} on $(hostname) ==="
date
echo "CODE=$CODE"
echo "CKPT=$CKPT"
echo "SPLIT=$SPLIT"
echo "OUT=$OUT"
ls -lh "$CKPT"
wc -l "$SPLIT/val.txt" "$SPLIT/val_masks.txt"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count())"
nvidia-smi || true

python -u "$EVAL_SCRIPT" \
  --code_dir "$CODE" \
  --ckpt "$CKPT" \
  --data_root "$BATH" \
  --list "$SPLIT/val.txt" \
  --lcc_mask_path "$LCC" \
  --lcc_list "$SPLIT/val_masks.txt" \
  --output_dir "$OUT" \
  --split_name val \
  --device cuda \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --in_chans 1 \
  --batch_size 8 \
  --num_workers 8 \
  --nodata -9999 \
  --amp \
  --tile_norm \
  --tile_norm_eps 1e-3 \
  --bottleneck_norm inst1d \
  --loss_mode mse \
  --lcc_mask_mode exact \
  --loss_on_lcc_only \
  --lcc_patch_threshold 0.5 \
  --min_lcc_patch_ratio 0.0001 \
  --max_lcc_patch_ratio 0.80 \
  --topk 200 \
  --topk_vis 100 \
  --bestk 200 \
  --bestk_vis 80 \
  --median_vis 40 \
  --good_min_lcc_patch_ratio 0.02 \
  --good_max_lcc_patch_ratio 0.60 \
  --good_min_masked_gt_std 0.5

date
echo "=== DONE VAL evaluation ==="
