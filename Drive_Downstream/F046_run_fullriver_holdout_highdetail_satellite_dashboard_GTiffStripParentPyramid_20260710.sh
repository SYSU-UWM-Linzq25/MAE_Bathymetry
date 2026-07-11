#!/usr/bin/env bash
#SBATCH -J F046_holdout_satmap_gtiffstrip
#SBATCH -p HydroIntel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/F046_holdout_satmap_gtiffstrip_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/F046_holdout_satmap_gtiffstrip_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
SCRIPT=${SCRIPT:-$WORK/script/F045_build_fullriver_holdout_highdetail_satellite_dashboard_GTiffStripParentPyramid_20260710.py}

PRED_ROOT=${PRED_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Predictions_F010_TileAvgVRT_D001NoDataSafe}
ERROR_ROOT=${ERROR_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_GT_Error_F020_TileVRT_D001NoDataSafe}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_FullRiver_E001/Tiles_1m}
TILE_RES=${TILE_RES:-1m}

OUT_DIR=${OUT_DIR:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_WebMap_F045_HoldoutOnly_GTiffStripParentPyramid_D001NoDataSafe}
OUT_HTML=${OUT_HTML:-F045_HoldoutOnly_HighDetail_Satellite_Dashboard_GTiffStripParentPyramid.html}
ZIP_NAME=${ZIP_NAME:-F045_HoldoutOnly_HighDetail_Satellite_Dashboard_GTiffStripParentPyramid_Package.zip}

# DISPLAY ONLY:
# DETAIL_RES_M controls the intermediate EPSG:3857 display grid.
# It does not alter F010/F020 source rasters and does not recompute F020 metrics.
# The actual finest XYZ web-pixel size is derived from MAX_ZOOM and is written
# explicitly into the HTML, README, and manifest.
DETAIL_RES_M=${DETAIL_RES_M:-4}
MIN_ZOOM=${MIN_ZOOM:--1}
MAX_ZOOM=${MAX_ZOOM:--1}
TILE_PROCESSES=${TILE_PROCESSES:-${SLURM_CPUS_PER_TASK:-12}}
STATS_MAX_PX=${STATS_MAX_PX:-2200}
OVERLAY_OPACITY=${OVERLAY_OPACITY:-0.82}
OVERWRITE=${OVERWRITE:-0}
KEEP_INTERMEDIATE=${KEEP_INTERMEDIATE:-0}
MAKE_ZIP=${MAKE_ZIP:-1}

EXPERIMENTS=${EXPERIMENTS:-"holdout_CA_D001NoDataSafe holdout_CO_D001NoDataSafe holdout_Santiam_D001NoDataSafe"}

CONDA_SH=""
for candidate in \
  /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh \
  /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh
do
  if [[ -f "$candidate" ]]; then
    CONDA_SH="$candidate"
    break
  fi
done

if [[ -z "$CONDA_SH" ]]; then
  echo "[ERROR] Cannot find conda.sh." >&2
  exit 2
fi

source "$CONDA_SH"
conda activate "$ROOT/conda_envs/mae_zequn"

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export GDAL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.tiff,.vrt

mkdir -p "$WORK/cross_validation_v2/logs"

for path in "$SCRIPT" "$PRED_ROOT" "$ERROR_ROOT" "$TILE_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 2
  fi
done

# This revision requires only GDAL core executables. It intentionally does not
# depend on gdal2tiles.py, gdal_calc.py, or the osgeo_utils Python package.
for exe in python gdalinfo gdalwarp gdalbuildvrt gdal_translate gdaldem; do
  if ! command -v "$exe" >/dev/null 2>&1; then
    echo "[ERROR] Required core executable is not available: $exe" >&2
    exit 2
  fi
done

python - <<'PY'
import numpy
import tifffile
import matplotlib
from PIL import Image

print("[PYTHON] numpy      =", numpy.__version__)
print("[PYTHON] tifffile   =", tifffile.__version__)
print("[PYTHON] matplotlib =", matplotlib.__version__)
print("[PYTHON] Pillow     =", Image.__version__ if hasattr(Image, "__version__") else "available")
PY

read -r -a EXP_ARRAY <<< "$EXPERIMENTS"

ARGS=(
  --pred_root "$PRED_ROOT"
  --error_root "$ERROR_ROOT"
  --tile_root "$TILE_ROOT"
  --tile_res "$TILE_RES"
  --out_dir "$OUT_DIR"
  --out_html "$OUT_HTML"
  --zip_name "$ZIP_NAME"
  --detail_res_m "$DETAIL_RES_M"
  --min_zoom "$MIN_ZOOM"
  --max_zoom "$MAX_ZOOM"
  --tile_processes "$TILE_PROCESSES"
  --stats_max_px "$STATS_MAX_PX"
  --overlay_opacity "$OVERLAY_OPACITY"
  --experiments "${EXP_ARRAY[@]}"
)

if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" ]]; then
  ARGS+=(--overwrite)
fi

if [[ "$KEEP_INTERMEDIATE" == "1" || "$KEEP_INTERMEDIATE" == "true" || "$KEEP_INTERMEDIATE" == "TRUE" ]]; then
  ARGS+=(--keep_intermediate)
fi

if [[ "$MAKE_ZIP" != "1" && "$MAKE_ZIP" != "true" && "$MAKE_ZIP" != "TRUE" ]]; then
  ARGS+=(--no_zip)
fi

if [[ -n "${RIVERS:-}" ]]; then
  read -r -a RIVER_ARRAY <<< "$RIVERS"
  ARGS+=(--rivers "${RIVER_ARRAY[@]}")
fi

echo "============================================================"
echo "F046 holdout high-detail satellite dashboard"
echo "TILER=finest XYZ zoom via temporary RGBA GeoTIFF strips + lower parent PNG pyramid"
echo "LOW_ZOOM_DIRECT_GDAL_WARP=NO"
echo "GDALWARP_DIRECT_PNG=NO"
echo "GDAL2TILES_REQUIRED=NO"
echo "GDAL_CALC_REQUIRED=NO"
date
echo "HOST=$(hostname)"
echo "CONDA_SH=$CONDA_SH"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PYTHON=$(command -v python)"
echo "GDAL_VERSION=$(gdalinfo --version 2>&1)"
echo "SCRIPT=$SCRIPT"
echo "PRED_ROOT=$PRED_ROOT"
echo "ERROR_ROOT=$ERROR_ROOT"
echo "TILE_ROOT=$TILE_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "OUT_HTML=$OUT_HTML"
echo "ZIP_NAME=$ZIP_NAME"
echo "EXPERIMENTS=$EXPERIMENTS"
echo "DETAIL_RES_M=$DETAIL_RES_M"
echo "MIN_ZOOM=$MIN_ZOOM"
echo "MAX_ZOOM=$MAX_ZOOM"
echo "TILE_PROCESSES=$TILE_PROCESSES"
echo "OVERWRITE=$OVERWRITE"
echo "KEEP_INTERMEDIATE=$KEEP_INTERMEDIATE"
echo "MAKE_ZIP=$MAKE_ZIP"
echo "NOTICE=Display-only reprojection/resampling; original F010/F020 rasters unchanged"
echo "METRICS=Read from native-resolution F020 summary; not recomputed from display tiles"
echo "============================================================"

python -u "$SCRIPT" "${ARGS[@]}"

echo "============================================================"
echo "DONE F046"
echo "HTML=$OUT_DIR/$OUT_HTML"
echo "ZIP=$(dirname "$OUT_DIR")/$ZIP_NAME"
date
echo "============================================================"
