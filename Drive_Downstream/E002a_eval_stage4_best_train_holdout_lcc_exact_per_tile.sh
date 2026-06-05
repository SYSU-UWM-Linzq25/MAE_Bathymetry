#!/bin/bash
#SBATCH -J eval_s4_train_lcc
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -t 12:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/eval_s4_train_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/eval_s4_train_%j.out
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  bash  $0 <VAL_RIVER_NAME>"
  echo "  sbatch $0 <VAL_RIVER_NAME>"
  echo
  echo "Example:"
  echo "  sbatch $0 MD_PotomacRiver_Bathy_2019"
  exit 1
fi

VAL_RIVER="$1"
SAFE_VAL=$(echo "$VAL_RIVER" | sed 's/[^A-Za-z0-9_]/_/g')
STD_SCALE=${STD_SCALE:-1.5}
END_EPOCH=${END_EPOCH:-400}
# Actual NoData value in the extracted 1 m bathy+3DEP tiles.
NODATA=${NODATA:-"-999999"}

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
SPLIT=$WORK/splits/bathy_lcc_1m_holdout_${SAFE_VAL}
BATH=$ROOT/Data/Tiles_for_Training_1m/1m_Tiles
MASK=$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles
RUN=$WORK/runs/stage4_bathy_finalmask_holdout_${SAFE_VAL}_exact_freeze_decoder_tilenormVis_std${STD_SCALE//./p}_1m_e${END_EPOCH}
CKPT=$RUN/checkpoint-best.pth
EVAL_SCRIPT=$WORK/script/E002_evaluate_stage4_lcc_exact_per_tile.py
OUT=$WORK/eval_stage4_bathy_finalmask_holdout_${SAFE_VAL}/train_best_tilenormVis_std${STD_SCALE//./p}
mkdir -p "$OUT" "$WORK/logs"

export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Stage4 final-mask LCC exact TRAIN evaluation ${SLURM_JOB_ID:-local} on $(hostname) ==="
date
echo "VAL_RIVER=$VAL_RIVER"
echo "CODE=$CODE"
echo "CKPT=$CKPT"
echo "SPLIT=$SPLIT"
echo "OUT=$OUT"
echo "STD_SCALE=$STD_SCALE"
echo "NODATA=$NODATA"

for f in "$CKPT" "$EVAL_SCRIPT" "$SPLIT/train.txt" "$SPLIT/train_masks.txt"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required file: $f"
    exit 2
  fi
done

ls -lh "$CKPT"
wc -l "$SPLIT/train.txt" "$SPLIT/train_masks.txt"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count())"
nvidia-smi || true

python -u "$EVAL_SCRIPT" \
  --code_dir "$CODE" \
  --ckpt "$CKPT" \
  --data_root "$BATH" \
  --list "$SPLIT/train.txt" \
  --lcc_mask_path "$MASK" \
  --lcc_list "$SPLIT/train_masks.txt" \
  --output_dir "$OUT" \
  --split_name train \
  --device cuda \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --in_chans 1 \
  --batch_size 8 \
  --num_workers 8 \
  --nodata "$NODATA" \
  --amp \
  --tile_norm \
  --tile_norm_visible_only \
  --tile_norm_eps 1e-3 \
  --tile_norm_std_scale "$STD_SCALE" \
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
echo "=== DONE Stage4 TRAIN evaluation ==="
