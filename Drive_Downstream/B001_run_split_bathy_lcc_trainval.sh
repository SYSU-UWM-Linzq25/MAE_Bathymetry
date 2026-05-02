#!/bin/bash
#SBATCH -J DOE_LS
#SBATCH --partition=HydroIntel # Specify the partition you want to use             # Node name
##SBATCH --nodelist=execute-[104-106,129,135,148-151,207,222]
#SBATCH --mem-per-cpu=6G
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=20
#SBATCH --output=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/slurm-%j.out
#SBATCH --error=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/slurm-%j.out
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
WORK=$ROOT/Downstream_Task_Bathy
BATH=$ROOT/Data/Tiles_for_Training_1m/1m_Tiles
LCC=$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles
OUT=$WORK/splits/bathy_lcc_1m_trainval_seed20260428

mkdir -p "$WORK/script" "$WORK/splits" "$WORK/runs" "$WORK/logs"

echo "ROOT=$ROOT"
echo "BATH=$BATH"
echo "LCC=$LCC"
echo "OUT=$OUT"

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

python -u "$WORK/script/A001_make_bathy_lcc_split_trainval.py" \
  --bath_dir "$BATH" \
  --mask_dir "$LCC" \
  --out_dir "$OUT" \
  --seed 20260428 \
  --train_ratio 0.80 \
  --smoke_train_n 1000 \
  --smoke_val_n 200

echo "=== Check split outputs ==="
wc -l "$OUT/train.txt" "$OUT/train_masks.txt"
wc -l "$OUT/val.txt" "$OUT/val_masks.txt"
wc -l "$OUT/smoke_train.txt" "$OUT/smoke_train_masks.txt"
wc -l "$OUT/smoke_val.txt" "$OUT/smoke_val_masks.txt"
echo "=== DONE split ==="
