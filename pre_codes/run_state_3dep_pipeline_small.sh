#!/bin/bash
set -euo pipefail

# Usage:
#   OUT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Model_Evaluation
#   bash $OUT/pre_codes/run_state_3dep_pipeline.sh KY /tank/data/SFS/xinyis/data/topography/3DEP/KY $OUT
#
# What it does:
#   0) scan all tifs in SRC_DIR
#   1) epsg QA map
#   2) gdalwarp -> Warped VRTs to EPSG:5070 (1m, aligned grid)
#   3) build one state-level mosaic VRT
#   4) gdal_retile.py -> 336x336 GeoTIFF tiles (with overlap)

STATE="${1:?STATE code, e.g. KY}"
SRC_DIR="${2:?Source 3DEP dir, e.g. /.../3DEP/KY}"
OUT_ROOT="${3:?OUT root, e.g. /.../Model_Evaluation}"

# ---------------------------
# Conda / Python availability
# ---------------------------
source /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

echo "[ENV] python=$(which python)"
python -V || true
echo "[ENV] conda_prefix=$CONDA_PREFIX"

# ---------------------------
# Parameters
# ---------------------------
TGT_EPSG="EPSG:5070"
TR="1 1"
NODATA="-9999"

PATCH=336
OVERLAP_RATIO=0.10

# overlap pixels (avoid python dependency; use awk)
OVERLAP_PIX=$(awk -v p="$PATCH" -v r="$OVERLAP_RATIO" 'BEGIN{printf "%d", (p*r+0.5)}')

# parallelism for gdalwarp
P="${SLURM_CPUS_PER_TASK:-8}"

# ---------------------------
# Output structure (systematic)
# OUT_ROOT/3DEP_Samples/STATE/...
# ---------------------------
BASE="${OUT_ROOT}/3DEP_Samples/${STATE}"
LIST_DIR="${BASE}/01_lists"
WARP_DIR="${BASE}/02_warpvrt_5070"
MOS_DIR="${BASE}/03_mosaic"
TILE_DIR="${BASE}/04_tiles_tif_${PATCH}_small"
LOG_DIR="${BASE}/logs"

mkdir -p "$LIST_DIR" "$WARP_DIR" "$MOS_DIR" "$TILE_DIR" "$LOG_DIR"

RAW_LIST="${LIST_DIR}/${STATE}_raw_tifs.txt"
EPSG_MAP="${LIST_DIR}/${STATE}_epsg_map.txt"
MOSAIC_VRT="${MOS_DIR}/${STATE}_5070.vrt"
TILE_LIST="${LIST_DIR}/${STATE}_tiles_${PATCH}.txt"

echo "============================================================"
echo "STATE=$STATE"
echo "SRC_DIR=$SRC_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "BASE=$BASE"
echo "TGT_EPSG=$TGT_EPSG  TR=$TR  NODATA=$NODATA"
echo "PATCH=$PATCH  OVERLAP_RATIO=$OVERLAP_RATIO  OVERLAP_PIX=$OVERLAP_PIX"
echo "PARALLEL(P)=$P"
echo "============================================================"

echo "=== [${STATE}] Step4: RANDOM sampling ${PATCH}x${PATCH} GeoTIFF tiles (target=N_TILES) ==="
N_TILES=${N_TILES:-1000}     # 你可以在运行前 export N_TILES=50000
MAX_TRIES=${MAX_TRIES:-20000} # 尝试次数上限（避免死循环）
MIN_VALID=${MIN_VALID:-0.98}   # 有效像元比例阈值（先粗略用后面的 nodata check）

STEP=$((PATCH - OVERLAP_PIX))
if [ "$STEP" -le 0 ]; then
  echo "[ERROR] STEP<=0. PATCH=${PATCH}, OVERLAP_PIX=${OVERLAP_PIX}"
  exit 3
fi

read -r W H < <(gdalinfo "$MOSAIC_VRT" | awk -F'[ ,]+' '/Size is/ {print $3, $4; exit}')
echo "[INFO] mosaic size: W=$W H=$H"
echo "[INFO] target tiles: $N_TILES, max tries: $MAX_TRIES, step=$STEP"

mkdir -p "$TILE_DIR"
: > "$TILE_LIST"

# 简单随机：用 $RANDOM 生成 (x,y)；为了对齐到 step 网格，取整到 STEP
# 注意：bash 的 RANDOM 只有 0..32767，所以用两次拼一下扩大范围
rand_u32() { echo $(( (RANDOM<<16) ^ RANDOM )); }

saved=0
tries=0

while [ "$saved" -lt "$N_TILES" ] && [ "$tries" -lt "$MAX_TRIES" ]; do
  tries=$((tries+1))

  rx=$(rand_u32); ry=$(rand_u32)
  x=$(( (rx % (W - PATCH)) / STEP * STEP ))
  y=$(( (ry % (H - PATCH)) / STEP * STEP ))

  out="${TILE_DIR}/${STATE}_s$(printf %06d $saved)_x${x}_y${y}.tif"
  if [ -s "$out" ]; then
    continue
  fi

  # 先快速切出来
  gdal_translate -q \
    -of GTiff \
    -srcwin "$x" "$y" "$PATCH" "$PATCH" \
    -co TILED=YES -co COMPRESS=LZW \
    "$MOSAIC_VRT" "$out" || { rm -f "$out"; continue; }

  # 快速过滤：如果几乎全是 nodata（-9999），就删掉
  # 用 gdalinfo 的 STATISTICS_MINIMUM 判断是否全 nodata/NaN（没有统计就会慢，所以只做最简检查）
  # 这里用 sample 检查：取 20 个像元看是不是都等于 -9999
  # (gdal_translate 读出的 nodata 会写进去，所以这里直接 awk)
  check=$(gdal_translate -q -of XYZ -srcwin "$((x+PATCH/2))" "$((y+PATCH/2))" 1 1 "$MOSAIC_VRT" /vsistdout/ | awk '{print $3}')
  if [ "$check" = "$NODATA" ] || [ -z "$check" ]; then
    rm -f "$out"
    continue
  fi

  echo "$out" >> "$TILE_LIST"
  saved=$((saved+1))

  if (( saved % 500 == 0 )); then
    echo "[PROG] saved=$saved / $N_TILES  tries=$tries"
  fi
done

echo "[INFO] saved=$saved tries=$tries"
echo "[OUT ] tiles: $TILE_DIR"
echo "[OUT ] list : $TILE_LIST"

if [ "$saved" -lt "$N_TILES" ]; then
  echo "[WARN] only saved $saved tiles (target $N_TILES). Consider increasing MAX_TRIES or lowering filter."
fi

