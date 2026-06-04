#!/bin/bash
#SBATCH -J SplitBathy
#SBATCH --partition=HydroIntel
#SBATCH --mem-per-cpu=6G
#SBATCH --ntasks=1
#SBATCH --output=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/slurm-%j_split_holdout.out
#SBATCH --error=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/slurm-%j_split_holdout.out
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  bash $0 <VAL_RIVER_NAME>"
  echo
  echo "Example:"
  echo "  bash $0 MD_PotomacRiver_Bathy_2019"
  echo
  echo "Before this, run:"
  echo "  bash B001_count_bathy_lcc_tiles_by_river.sh"
  echo
  echo "Then choose a validation river from:"
  echo "  Downstream_Task_Bathy/splits/bathy_lcc_1m_byRiver_count/tile_count_by_river.csv"
  exit 1
fi

VAL_RIVER="$1"

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
WORK=$ROOT/Downstream_Task_Bathy

BATH=$ROOT/Data/Tiles_for_Training_1m/1m_Tiles
MASK=$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles

SAFE_VAL=$(echo "$VAL_RIVER" | sed 's/[^A-Za-z0-9_]/_/g')
OUT=$WORK/splits/bathy_lcc_1m_holdout_${SAFE_VAL}

PY=$WORK/script/A002_make_bathy_lcc_split_by_river.py

mkdir -p "$WORK/script" "$WORK/splits" "$WORK/runs" "$WORK/logs"

echo "============================================================"
echo "Create bathy/mask train-val split by holdout river"
echo "ROOT      = $ROOT"
echo "WORK      = $WORK"
echo "BATH      = $BATH"
echo "MASK      = $MASK"
echo "VAL_RIVER = $VAL_RIVER"
echo "OUT       = $OUT"
echo "PY        = $PY"
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
  --out_dir "$OUT" \
  --val_river "$VAL_RIVER" \
  --seed 20260428 \
  --smoke_train_n 1000 \
  --smoke_val_n 200

echo
echo "============================================================"
echo "Check split outputs"
echo "============================================================"

wc -l "$OUT/train.txt" "$OUT/train_masks.txt"
wc -l "$OUT/val.txt" "$OUT/val_masks.txt"
wc -l "$OUT/smoke_train.txt" "$OUT/smoke_train_masks.txt"
wc -l "$OUT/smoke_val.txt" "$OUT/smoke_val_masks.txt"

echo
echo "============================================================"
echo "Check validation river"
echo "============================================================"

echo "[Check] val files NOT containing VAL_RIVER:"
if grep -v "$VAL_RIVER" "$OUT/val.txt" | head; then
  true
fi

echo
echo "[Check] train files containing VAL_RIVER:"
if grep "$VAL_RIVER" "$OUT/train.txt" | head; then
  true
fi

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
echo "Output split folder:"
echo "$OUT"

echo
echo "Main outputs:"
echo "$OUT/all_pairs.csv"
echo "$OUT/train.txt"
echo "$OUT/train_masks.txt"
echo "$OUT/val.txt"
echo "$OUT/val_masks.txt"
echo "$OUT/smoke_train.txt"
echo "$OUT/smoke_train_masks.txt"
echo "$OUT/smoke_val.txt"
echo "$OUT/smoke_val_masks.txt"
echo "$OUT/tile_count_by_river.csv"

echo "=== DONE split by holdout river ==="