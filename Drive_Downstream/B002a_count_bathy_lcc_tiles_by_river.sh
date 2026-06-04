#!/bin/bash
#SBATCH -J CountTiles
#SBATCH --partition=HydroIntel
#SBATCH --mem-per-cpu=6G
#SBATCH --ntasks=1
#SBATCH --output=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/slurm-%j_count_tiles.out
#SBATCH --error=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/slurm-%j_count_tiles.out
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
WORK=$ROOT/Downstream_Task_Bathy

BATH=$ROOT/Data/Tiles_for_Training_1m/1m_Tiles
MASK=$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles

OUT=$WORK/splits/bathy_lcc_1m_byRiver_count

PY=$WORK/script/A002_make_bathy_lcc_split_by_river.py

mkdir -p "$WORK/script" "$WORK/splits" "$WORK/runs" "$WORK/logs"

echo "============================================================"
echo "Count bathy/mask tiles by river"
echo "ROOT = $ROOT"
echo "WORK = $WORK"
echo "BATH = $BATH"
echo "MASK = $MASK"
echo "OUT  = $OUT"
echo "PY   = $PY"
echo "============================================================"

if [[ ! -f "$PY" ]]; then
  echo "ERROR: Python script not found:"
  echo "$PY"
  exit 1
fi

if [[ ! -d "$BATH" ]]; then
  echo "ERROR: bath tile folder not found:"
  echo "$BATH"
  exit 1
fi

if [[ ! -d "$MASK" ]]; then
  echo "ERROR: mask tile folder not found:"
  echo "$MASK"
  exit 1
fi

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

python -u "$PY" \
  --bath_dir "$BATH" \
  --mask_dir "$MASK" \
  --out_dir "$OUT"

echo
echo "============================================================"
echo "Tile count by river"
echo "============================================================"

if command -v column >/dev/null 2>&1; then
  column -s, -t "$OUT/tile_count_by_river.csv"
else
  cat "$OUT/tile_count_by_river.csv"
fi

echo
echo "Output folder:"
echo "$OUT"
echo
echo "Main output:"
echo "$OUT/tile_count_by_river.csv"

echo "=== DONE count tiles by river ==="