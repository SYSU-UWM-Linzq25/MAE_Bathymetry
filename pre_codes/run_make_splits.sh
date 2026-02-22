#!/bin/bash
set -euo pipefail

ROOT="/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain"

# conda env (按你给的方式)
source /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT"

python pre_codes/make_splits_3dep_tiles.py \
  --root "$ROOT" \
  --patch 336 \
  --train 0.90 --val 0.05 --test 0.05 \
  --block_tiles 10 \
  --step 302 \
  --seed 1 \
  --holdout_state KY \
  --out_dir splits

