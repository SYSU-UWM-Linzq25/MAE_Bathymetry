#!/bin/bash
#SBATCH -J S1_ridge_filter
#SBATCH -p HydroIntel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 02:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/Evaluation_tileVis_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/Evaluation_tileVis_%j.out
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy

set -euo pipefail

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn

cd /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/eval_stage3_bathy_lcc_exact_best/

EVAL_DIR=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/eval_stage3_bathy_lcc_exact_best/val_best_fullnorm_top100

BATH=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Data/Tiles_for_Training_1m/1m_Tiles
LCC=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Data/TilesMask_for_Training_1m/1m_Tiles

OUT=review_eval_tiles_for_gis/worst100_best_and_Nontrivial100_val

python /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/script/S002_extract_eval_tiles_for_gis.py \
  --eval_dir "$EVAL_DIR" \
  --out_dir "$OUT" \
  --bath_dir "$BATH" \
  --lcc_dir "$LCC" \
  --n 100 \
  --mode copy
