#!/bin/bash
#SBATCH -J mae_stage2_up_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
##SBATCH -t 04:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_stage2_up_eval_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_stage2_up_eval_%j.out
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

# ===== 上游模型 =====
UP_OUT=$ROOT/runs/Small_tilenorm_viscorr_336

# ===== stage2 的新 val =====
STAGE2_VAL=$ROOT/splits/stage2_decoder_random20/stage2_val.txt

# ===== 输出根目录 =====
EVAL_ROOT=$ROOT/eval_stage2_compare
mkdir -p "$EVAL_ROOT"

# ===== 所有评估共用同一批可视化 tile =====
VIS_IDX=$EVAL_ROOT/fixed_vis_indices.txt

# ===== 上游模型的 mask_ratio sweep =====
MASK_RATIOS=("0.75" "0.50" "0.35" "0.20")

for MR in "${MASK_RATIOS[@]}"; do
  OUTDIR=$EVAL_ROOT/upstream_mr_${MR}
  mkdir -p "$OUTDIR"

  echo "======================================================"
  echo "[UPSTREAM] stage2_val, mask_ratio=$MR"
  echo "[OUT] $OUTDIR"
  echo "======================================================"

  python $CODEDIR/mae_evaluate_dem_meters_topk.py \
    --ckpt        $UP_OUT/checkpoint-best.pth \
    --list        $STAGE2_VAL \
    --norm_json   $UP_OUT/norm_stats_train.json \
    --output_dir  $OUTDIR \
    --mask_ratio  $MR \
    --batch_size  16 \
    --num_workers 8 \
    --nodata      -9999 \
    --topk        200 \
    --seed        42 \
    --amp \
    --tile_norm \
    --tile_norm_eps 1e-3 \
    --bottleneck_norm inst1d \
    --loss_mode   mse \
    --postproc    none \
    --save_vis_tif \
    --vis_n       10 \
    --vis_seed    42 \
    --vis_indices_txt $VIS_IDX
done
