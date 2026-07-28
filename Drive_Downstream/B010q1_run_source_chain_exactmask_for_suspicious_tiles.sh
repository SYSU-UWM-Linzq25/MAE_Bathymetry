#!/usr/bin/env bash
# NUMBER-ALIGNED NAME: B010q1_run_source_chain_exactmask_for_suspicious_tiles.sh
# ORIGINAL BACKUP NAME: B013_run_source_chain_exactmask_for_suspicious_tiles.sh
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
set -euo pipefail

# ============================================================
# B013: run A013 source-chain exact-mask inspector for suspicious
# tiles from cross-validation.
#
# Example:
#   TILE_LIST_TXT=/tank/.../cross_validation/evaluation/_summary/suspicious_tiles_for_A012.txt \
#   OUT_DIR=/tank/.../cross_validation/evaluation/source_chain_check/top300 \
#   bash B010q1_run_source_chain_exactmask_for_suspicious_tiles.sh
#
# Or pass filenames directly:
#   bash B010q1_run_source_chain_exactmask_for_suspicious_tiles.sh \
#     Select_tile_Basin_1m_CA_KlamathRiver_TopoBathy_2018_D18_ID777.tif
# ============================================================

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
WORK=$ROOT/Downstream_Task_Bathy
PROC=/tank/data/SFS/xinyis/data/bathymetry/Processed_Results
AUX=/tank/data/SFS/xinyis/data/bathymetry/Data_for_BetterMask/Auxiliary_ByRiver_1m

SCRIPT=${SCRIPT:-$WORK/script/A010q1_inspect_suspicious_tile_source_chain_exactmask.py}

# These are the model-used tile/mask directories after CanonicalND publication.
TILE_DIR=${TILE_DIR:-$ROOT/Data/Tiles_for_Training_1m/1m_Tiles}
MASK_DIR=${MASK_DIR:-$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles}

# Source-chain roots.
BATHY_ROOT=${BATHY_ROOT:-$PROC/Bathy_1m_CanonicalND}
DEP_ROOT=${DEP_ROOT:-$PROC/3DEP_1m_ResampleClip}
MERGED_ROOT=${MERGED_ROOT:-$PROC/Bathy3DEP_Merged_Tiff_1m_CanonicalND}
FINAL_MASK_ROOT=${FINAL_MASK_ROOT:-$PROC/PredictionMask_LCCBathyValid_1m_CanonicalND}
AUX_ROOT=${AUX_ROOT:-$AUX}

TILE_LIST_TXT=${TILE_LIST_TXT:-}
OUT_DIR=${OUT_DIR:-$WORK/cross_validation/evaluation/source_chain_exactmask_$(date +%Y%m%d_%H%M%S)}

module purge || true

# A013 calls GDAL command-line tools directly: gdalinfo and gdal_translate.
# Some GPU nodes do not expose GDAL after activating the conda env, so load a
# GDAL module first and verify that the commands are available.
GDAL_MODULE=${GDAL_MODULE:-gdal/2.3.0}
if module load "$GDAL_MODULE" 2>/dev/null; then
  echo "[GDAL] loaded module: $GDAL_MODULE"
else
  echo "[GDAL][WARN] Could not load module: $GDAL_MODULE"
  echo "[GDAL][WARN] Trying to continue; command -v check below will fail if GDAL is unavailable."
fi

source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing SCRIPT=$SCRIPT" >&2; exit 2; }

if ! command -v gdalinfo >/dev/null 2>&1; then
  echo "[ERROR] gdalinfo not found in PATH." >&2
  echo "        Try: module avail gdal" >&2
  echo "        Then rerun with: GDAL_MODULE=<module_name> bash $(basename "$0")" >&2
  exit 3
fi
if ! command -v gdal_translate >/dev/null 2>&1; then
  echo "[ERROR] gdal_translate not found in PATH." >&2
  echo "        Try: module avail gdal" >&2
  echo "        Then rerun with: GDAL_MODULE=<module_name> bash $(basename "$0")" >&2
  exit 4
fi

echo "[GDAL] gdalinfo=$(command -v gdalinfo)"
echo "[GDAL] gdal_translate=$(command -v gdal_translate)"
gdalinfo --version || true

ARGS=(
  --tile_dir "$TILE_DIR"
  --mask_dir "$MASK_DIR"
  --out_dir "$OUT_DIR"
  --bathy_root "$BATHY_ROOT"
  --dep_root "$DEP_ROOT"
  --merged_root "$MERGED_ROOT"
  --final_mask_root "$FINAL_MASK_ROOT"
  --aux_root "$AUX_ROOT"
)

if [[ -n "$TILE_LIST_TXT" ]]; then
  ARGS+=(--tile_list_txt "$TILE_LIST_TXT")
fi

if [[ $# -gt 0 ]]; then
  ARGS+=(--tile_names "$@")
fi

echo "=== B013 source-chain exact-mask inspector ==="
echo "SCRIPT=$SCRIPT"
echo "TILE_DIR=$TILE_DIR"
echo "MASK_DIR=$MASK_DIR"
echo "BATHY_ROOT=$BATHY_ROOT"
echo "DEP_ROOT=$DEP_ROOT"
echo "MERGED_ROOT=$MERGED_ROOT"
echo "FINAL_MASK_ROOT=$FINAL_MASK_ROOT"
echo "AUX_ROOT=$AUX_ROOT"
echo "TILE_LIST_TXT=$TILE_LIST_TXT"
echo "OUT_DIR=$OUT_DIR"
echo "TILE_NAMES=$*"

python "$SCRIPT" "${ARGS[@]}"

echo "=== DONE ==="
echo "$OUT_DIR"
