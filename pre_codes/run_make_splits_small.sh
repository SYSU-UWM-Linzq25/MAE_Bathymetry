#!/bin/bash
set -euo pipefail

ROOT="/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain"

# (可选) conda env：如果你不想激活也行，只要 python 能跑
source /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT"

# SMALL: tiles are under each state:
#   3DEP_Samples/<STATE>/04_tiles_tif_336_small/*.tif
python pre_codes/make_splits_3dep_tiles_small.py \
  --root "$ROOT" \
  --samples_dir "3DEP_Samples" \
  --patch 336 \
  --tiles_subdir "04_tiles_tif_336_small" \
  --train 0.95 --val 0.05 \
  --block_tiles 10 \
  --step 302 \
  --seed 1 \
  --holdout_state "KY" \
  --allow_partial_block \
  --out_dir "splits/smoke_small_1000"

echo
echo "[CHECK]"
wc -l splits/smoke_small_1000/global/train.txt \
      splits/smoke_small_1000/global/val.txt \
      splits/smoke_small_1000/global/holdout_KY.txt || true
echo "[CHECK] per-state counts (first few)"
for f in splits/smoke_small_1000/by_state/*_train.txt; do
  st=$(basename "$f" _train.txt)
  tr=$(wc -l < "$f")
  va=$(wc -l < "splits/smoke_small_1000/by_state/${st}_val.txt")
  echo "$st train=$tr val=$va"
done | head
