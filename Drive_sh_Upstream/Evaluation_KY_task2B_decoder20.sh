#!/bin/bash
#SBATCH -J mae_KY_dec_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
##SBATCH -t 02:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_KY_dec_eval_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_KY_dec_eval_%j.out
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

# ===== 任务2B训练好的 decoder 模型 =====
DEC_OUT=$ROOT/runs/KY_task2B_decoder_random20

# ===== 任务2B 的 KY val =====
KY_VAL=$ROOT/splits/KY_task2B_decoder20/ky_val.txt

# ===== 输出根目录 =====
EVAL_ROOT=$ROOT/eval_KY_task2B_compare
mkdir -p "$EVAL_ROOT"

# ===== 和 upstream 共用同一批可视化 tile =====
VIS_IDX=$EVAL_ROOT/fixed_vis_indices.txt

OUTDIR=$EVAL_ROOT/stage2B_KY_decoder_mr_0.20
mkdir -p "$OUTDIR"

echo "======================================================"
echo "[DECODER] KY_val, mask_ratio=0.20"
echo "[OUT] $OUTDIR"
echo "======================================================"

python $CODEDIR/mae_evaluate_dem_meters_topk.py \
  --ckpt        $DEC_OUT/checkpoint-best.pth \
  --list        $KY_VAL \
  --norm_json   $DEC_OUT/norm_stats_train.json \
  --output_dir  $OUTDIR \
  --mask_ratio  0.20 \
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
