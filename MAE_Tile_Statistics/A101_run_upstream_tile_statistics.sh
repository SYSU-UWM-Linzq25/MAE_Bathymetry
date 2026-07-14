#!/usr/bin/env bash
#SBATCH -J mae_up_stats
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 1-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_up_stats_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/log/mae_up_stats_%j.err

set -euo pipefail

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
UP_ROOT=$ROOT/Upstream_Model_ReTrain
SCRIPT=${SCRIPT:-$ROOT/Upstream_Model_ReTrain/scripts/A100_collect_mae_tile_statistics.py}
OUT_ROOT=${OUT_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Z997_MAE_Tile_Statistics_20260711}
OUT=$OUT_ROOT/Upstream_AllValid_ByState
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}

# Activate the same environment used by MAE.  Support both conda locations
# seen in the project scripts.
if [[ -f /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh ]]; then
  source /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh
elif [[ -f /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
else
  echo "[ERROR] conda.sh was not found." >&2
  exit 2
fi
conda activate "$ROOT/conda_envs/mae_zequn"

mkdir -p "$OUT" "$UP_ROOT/log"

echo "============================================================"
echo "Upstream MAE tile statistics: ALL valid pixels"
echo "HOST=$(hostname)"
echo "SCRIPT=$SCRIPT"
echo "OUT=$OUT"
echo "WORKERS=$WORKERS"
echo "============================================================"

# train + val are the tiles used during upstream model development.
# holdout_KY is retained as a separate comparison split and is never mixed
# into train/val in the output.
python -u "$SCRIPT" upstream \
  --data-root "$UP_ROOT" \
  --list "train=$UP_ROOT/splits/smoke_small_1000/global/train.txt" \
  --list "val=$UP_ROOT/splits/smoke_small_1000/global/val.txt" \
  --list "holdout_KY=$UP_ROOT/splits/smoke_small_1000/global/holdout_KY.txt" \
  --output-dir "$OUT" \
  --nodata -9999 \
  --nodata-threshold -9999 \
  --std-scale 1.0 \
  --eps 1e-3 \
  --workers "$WORKERS" \
  --progress-every 250 \
  --fail-on-error

echo "[DONE] $OUT"
