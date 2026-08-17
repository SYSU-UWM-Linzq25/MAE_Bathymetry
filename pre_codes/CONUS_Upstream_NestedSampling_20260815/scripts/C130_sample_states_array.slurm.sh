#!/usr/bin/env bash
#SBATCH --job-name=C130_CONUS_sample
#SBATCH --partition=HydroIntel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-47%8
#SBATCH --output=logs/C130_CONUS_sample_%A_%a.out
#SBATCH --error=logs/C130_CONUS_sample_%A_%a.err
set -euo pipefail

SUBMIT_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck disable=SC1090
source "$SUBMIT_ROOT/scripts/C080_slurm_runtime.sh"
conus_setup_paths "$SUBMIT_ROOT"
conus_activate_python
PROJECT_ROOT=${PROJECT_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/CONUS_3DEP_NestedNative1m_20260815}
BOUNDARY_SHP=${BOUNDARY_SHP:-$DATA_ROOT/boundaries/tl_2025_us_state.shp}
STATE_LIST=${STATE_LIST:-$CODE_ROOT/config/conus48.txt}
TARGET_PER_STATE=${TARGET_PER_STATE:-1000}
MIN_VALID_RATIO=${MIN_VALID_RATIO:-1.0}
OUTPUT_MODE=${OUTPUT_MODE:-VRT}
SEED=${SEED:-20260815}

TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0}
STATE=$(sed -n "$((TASK_INDEX + 1))p" "$STATE_LIST")
[[ -n "$STATE" ]] || { echo "No state for array index $TASK_INDEX" >&2; exit 2; }
mkdir -p "$DATA_ROOT/samples"

"$PYTHON_BIN" "$SCRIPT_DIR/C050_sample_nested_tiles.py" \
  --state-boundaries "$BOUNDARY_SHP" \
  --anchor-plan "$DATA_ROOT/plan/anchor_plan.csv" \
  --source-index "$DATA_ROOT/prepared_sources/source_index.csv" \
  --out-root "$DATA_ROOT/samples" \
  --states "$STATE" \
  --target-per-state "$TARGET_PER_STATE" \
  --min-valid-ratio "$MIN_VALID_RATIO" \
  --output-mode "$OUTPUT_MODE" \
  --seed "$SEED" \
  --require-target

echo "[C130] state=$STATE complete"
