#!/usr/bin/env bash
#SBATCH -J mae_down_stats
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 1-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/mae_down_stats_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/mae_down_stats_%j.err

set -euo pipefail

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
SCRIPT=${SCRIPT:-$ROOT/Downstream_Task_Bathy/script/A100_collect_mae_tile_statistics.py}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2/Tiles_1m}
OUT_ROOT=${OUT_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Z997_MAE_Tile_Statistics_20260711}
OUT=$OUT_ROOT/Downstream_Known_Masked_ByRiver
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}

if [[ -f /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh ]]; then
  source /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh
elif [[ -f /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
else
  echo "[ERROR] conda.sh was not found." >&2
  exit 2
fi
conda activate "$ROOT/conda_envs/mae_zequn"

mkdir -p "$OUT" "$ROOT/Downstream_Task_Bathy/logs"

echo "============================================================"
echo "Downstream MAE tile statistics"
echo "  known  = valid DEM and Hidden_Mask == 0"
echo "  masked = valid DEM and Hidden_Mask == 1"
echo "  loss   = valid DEM and Loss_Mask_Pixel == 1"
echo "HOST=$(hostname)"
echo "SCRIPT=$SCRIPT"
echo "TILE_ROOT=$TILE_ROOT"
echo "OUT=$OUT"
echo "WORKERS=$WORKERS"
echo "============================================================"

python -u "$SCRIPT" downstream \
  --tile-root "$TILE_ROOT" \
  --output-dir "$OUT" \
  --dem-folder Train_tile \
  --hidden-folder Hidden_Mask \
  --loss-folder Loss_Mask_Pixel \
  --nodata -999999 \
  --nodata-threshold -9999 \
  --mask-nodata 255 \
  --mask-threshold 0.5 \
  --std-scale 1.5 \
  --eps 1e-3 \
  --workers "$WORKERS" \
  --progress-every 250 \
  --fail-on-error

echo "[DONE] $OUT"
