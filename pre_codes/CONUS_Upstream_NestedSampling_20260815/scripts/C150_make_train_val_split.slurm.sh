#!/usr/bin/env bash
#SBATCH --job-name=C150_CONUS_split
#SBATCH --partition=HydroIntel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/C150_CONUS_split_%j.out
#SBATCH --error=logs/C150_CONUS_split_%j.err
# Optional later data-processing step. It does not query or download USGS products.
set -euo pipefail

SUBMIT_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck disable=SC1090
source "$SUBMIT_ROOT/scripts/C080_slurm_runtime.sh"
conus_setup_paths "$SUBMIT_ROOT"
conus_activate_python
PROJECT_ROOT=${PROJECT_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/CONUS_3DEP_NestedNative1m_20260815}
TRAIN_FRACTION=${TRAIN_FRACTION:-0.80}
SEED=${SEED:-20260815}

mkdir -p "$DATA_ROOT/splits"
"$PYTHON_BIN" "$SCRIPT_DIR/C060_make_spatial_splits.py" \
  --sampling-root "$DATA_ROOT/samples" \
  --out-dir "$DATA_ROOT/splits" \
  --train-fraction "$TRAIN_FRACTION" \
  --seed "$SEED"

echo "[C150] train/val spatial split complete"
