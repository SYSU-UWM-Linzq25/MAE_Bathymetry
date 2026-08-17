#!/usr/bin/env bash
#SBATCH --job-name=C100_CONUS_inventory
#SBATCH --partition=HydroIntel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/C100_CONUS_inventory_%j.out
#SBATCH --error=logs/C100_CONUS_inventory_%j.err
set -euo pipefail

SUBMIT_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck disable=SC1090
source "$SUBMIT_ROOT/scripts/C080_slurm_runtime.sh"
conus_setup_paths "$SUBMIT_ROOT"
conus_activate_python
PROJECT_ROOT=${PROJECT_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/CONUS_3DEP_NestedNative1m_20260815}
BOUNDARY_SHP=${BOUNDARY_SHP:-$DATA_ROOT/boundaries/tl_2025_us_state.shp}
STATES=${STATES:-}
ANCHORS_PER_STATE=${ANCHORS_PER_STATE:-150}
SEED=${SEED:-20260815}

mkdir -p "$DATA_ROOT/inventory" "$DATA_ROOT/plan"
read -r -a STATE_ARGS <<< "$STATES"

"$PYTHON_BIN" "$SCRIPT_DIR/C010_query_tnm_inventory.py" \
  --state-boundaries "$BOUNDARY_SHP" \
  --out-dir "$DATA_ROOT/inventory" \
  --states "${STATE_ARGS[@]}"

"$PYTHON_BIN" "$SCRIPT_DIR/C020_plan_anchor_downloads.py" \
  --inventory-dir "$DATA_ROOT/inventory" \
  --out-dir "$DATA_ROOT/plan" \
  --states "${STATE_ARGS[@]}" \
  --anchors-per-state "$ANCHORS_PER_STATE" \
  --seed "$SEED"

echo "[C100] plan=$DATA_ROOT/plan/download_manifest.tsv"
