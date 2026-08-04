#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
SOURCE_PACKAGE_DIR=${SOURCE_PACKAGE_DIR:-$(cd "$(dirname "$0")" && pwd)}
SOURCE_CODE=${SOURCE_CODE:-$ROOT/mae_Retrain}
SOURCE_CODE_OVERLAY=${SOURCE_CODE_OVERLAY:-$SOURCE_PACKAGE_DIR/code_relax}

mkdir -p "$RELAX_ROOT/script" "$RELAX_ROOT/matlab" "$RELAX_ROOT/docs" \
  "$RELAX_ROOT/results/D002c_D001c_Tile_QA" \
  "$RELAX_ROOT/results/NormOnly" "$RELAX_ROOT/results/MeterOnly" \
  "$RELAX_ROOT/results/NormThenMeter" "$RELAX_ROOT/results/MeterThenNorm" \
  "$RELAX_ROOT/results/EncoderUnfreeze_Preflight"
cp -pf "$SOURCE_PACKAGE_DIR"/script/* "$RELAX_ROOT/script/"
cp -pf "$SOURCE_PACKAGE_DIR"/matlab/* "$RELAX_ROOT/matlab/"
cp -pf "$SOURCE_PACKAGE_DIR"/docs/* "$RELAX_ROOT/docs/"
chmod +x "$RELAX_ROOT"/script/*.sh "$RELAX_ROOT"/script/*.py

if [[ ! -d "$SOURCE_CODE" ]]; then
  echo "[ERROR] Missing source model-code directory: $SOURCE_CODE" >&2
  exit 2
fi
mkdir -p "$RELAX_ROOT/mae_Retrain_relax"
rsync -a --delete \
  --exclude='__pycache__/' \
  --exclude='runs/' \
  --exclude='*.pth' \
  "$SOURCE_CODE/" "$RELAX_ROOT/mae_Retrain_relax/"

if [[ ! -f "$SOURCE_CODE_OVERLAY/main_pretrain_dem_unified_relax.py" || \
      ! -f "$SOURCE_CODE_OVERLAY/engine_pretrain_unified_relax.py" ]]; then
  echo "[ERROR] Missing unified RELAX Python overlay: $SOURCE_CODE_OVERLAY" >&2
  exit 2
fi
cp -pf "$SOURCE_CODE_OVERLAY/main_pretrain_dem_unified_relax.py" \
  "$RELAX_ROOT/mae_Retrain_relax/"
cp -pf "$SOURCE_CODE_OVERLAY/engine_pretrain_unified_relax.py" \
  "$RELAX_ROOT/mae_Retrain_relax/"

echo "Installed isolated relax project:"
echo "  root    = $RELAX_ROOT"
echo "  scripts = $RELAX_ROOT/script"
echo "  code    = $RELAX_ROOT/mae_Retrain_relax"
echo "  results = $RELAX_ROOT/results"
echo "  unified= $RELAX_ROOT/mae_Retrain_relax/main_pretrain_dem_unified_relax.py"
