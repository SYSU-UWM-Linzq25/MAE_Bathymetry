#!/usr/bin/env bash
# Download official 2025 Census TIGER/Line state boundaries used for CONUS sampling.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/CONUS_3DEP_NestedNative1m_20260815}
BOUNDARY_DIR=${BOUNDARY_DIR:-$DATA_ROOT/boundaries}
URL=${STATE_BOUNDARY_URL:-https://www2.census.gov/geo/tiger/TIGER2025/STATE/tl_2025_us_state.zip}

mkdir -p "$BOUNDARY_DIR"
ZIP="$BOUNDARY_DIR/tl_2025_us_state.zip"
wget -c --retry-connrefused --waitretry=5 --timeout=90 --tries=10 -O "${ZIP}.part" "$URL"
mv -f "${ZIP}.part" "$ZIP"
unzip -o "$ZIP" -d "$BOUNDARY_DIR"
test -s "$BOUNDARY_DIR/tl_2025_us_state.shp"
echo "[boundaries] $BOUNDARY_DIR/tl_2025_us_state.shp"
