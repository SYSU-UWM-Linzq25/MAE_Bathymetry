#!/usr/bin/env bash
#SBATCH --job-name=C090_CONUS_smoke
#SBATCH --partition=HydroIntel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/C090_CONUS_smoke_%j.out
#SBATCH --error=logs/C090_CONUS_smoke_%j.err
# End-to-end smoke test: WI, three source anchors, twenty nested centers.
set -euo pipefail

SUBMIT_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck disable=SC1090
source "$SUBMIT_ROOT/scripts/C080_slurm_runtime.sh"
conus_setup_paths "$SUBMIT_ROOT"
conus_activate_python
PROJECT_ROOT=${PROJECT_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain}
SMOKE_ROOT=${SMOKE_ROOT:-$PROJECT_ROOT/CONUS_3DEP_NestedNative1m_20260815_SMOKE}
BOUNDARY_SHP=${BOUNDARY_SHP:-$SMOKE_ROOT/boundaries/tl_2025_us_state.shp}
SEED=${SEED:-20260815}

mkdir -p "$SMOKE_ROOT"
if [[ ! -s "$BOUNDARY_SHP" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$SMOKE_ROOT" \
    bash "$SCRIPT_DIR/C000_download_state_boundaries.sh"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/C010_query_tnm_inventory.py" \
  --state-boundaries "$BOUNDARY_SHP" \
  --out-dir "$SMOKE_ROOT/inventory" \
  --states WI

"$PYTHON_BIN" "$SCRIPT_DIR/C020_plan_anchor_downloads.py" \
  --inventory-dir "$SMOKE_ROOT/inventory" \
  --out-dir "$SMOKE_ROOT/plan" \
  --states WI \
  --anchors-per-state 5 \
  --seed "$SEED"

bash "$SCRIPT_DIR/C030_download_selected_sources.sh" \
  "$SMOKE_ROOT/plan/download_manifest.tsv" \
  "$SMOKE_ROOT/source_downloads" 4

"$PYTHON_BIN" "$SCRIPT_DIR/C040_prepare_sources.py" \
  --download-manifest "$SMOKE_ROOT/plan/download_manifest.tsv" \
  --data-root "$SMOKE_ROOT/source_downloads" \
  --out-dir "$SMOKE_ROOT/prepared_sources" \
  --format VRT \
  --workers 2

"$PYTHON_BIN" "$SCRIPT_DIR/C050_sample_nested_tiles.py" \
  --state-boundaries "$BOUNDARY_SHP" \
  --anchor-plan "$SMOKE_ROOT/plan/anchor_plan.csv" \
  --source-index "$SMOKE_ROOT/prepared_sources/source_index.csv" \
  --out-root "$SMOKE_ROOT/samples" \
  --states WI \
  --target-per-state 20 \
  --min-valid-ratio 1.0 \
  --rounds-per-stage 40 \
  --max-stall-rounds 8 \
  --candidate-attempts-per-source 10 \
  --output-mode VRT \
  --seed "$SEED" \
  --require-target

"$PYTHON_BIN" "$SCRIPT_DIR/C070_verify_sampling.py" \
  --sampling-root "$SMOKE_ROOT/samples" \
  --states WI \
  --target-per-state 20 \
  --report "$SMOKE_ROOT/qa_errors.csv"

echo "[C090] WI smoke PASS: $SMOKE_ROOT"
