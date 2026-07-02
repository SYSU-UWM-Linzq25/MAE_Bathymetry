#!/usr/bin/env bash
set -euo pipefail

ROOT="/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m"
OUTROOT="/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m_FixND"
DST_ND="-999999"

mkdir -p "$OUTROOT"

fix_one () {
  local river="$1"
  local srcnd="$2"
  local vrt="$ROOT/$river/Bathy_1m.vrt"

  local outdir="$OUTROOT/$river"
  mkdir -p "$outdir"

  local out_tif="$outdir/Bathy_1m.tif"
  local out_vrt="$outdir/Bathy_1m.vrt"

  echo "==== Fix ND ===="
  echo "RIVER : $river"
  echo "VRT   : $vrt"
  echo "SRC ND: $srcnd  ->  DST ND: $DST_ND"
  echo "OUT   : $out_tif"

  gdalwarp -overwrite -of GTiff -r near -multi -wo NUM_THREADS=ALL_CPUS \
    -srcnodata "$srcnd" -dstnodata "$DST_ND" \
    -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES \
    "$vrt" "$out_tif"

  gdalbuildvrt -overwrite -vrtnodata "$DST_ND" "$out_vrt" "$out_tif"

  echo "Check nodata(meta):"
  gdalinfo -json "$out_tif" | grep -m1 -i "noDataValue" || true
  echo
}

# 这几条都用同一个 srcnodata（float32 -FLT_MAX）
SRC_F32_MIN="-3.4028235e+38"

fix_one "Kletzch_Combined_UpMax3Null"        "$SRC_F32_MIN"
fix_one "BadgerFinNull"                     "$SRC_F32_MIN"
fix_one "Estabrook_Combined"                "$SRC_F32_MIN"
fix_one "KewaFix2Null"                      "$SRC_F32_MIN"
fix_one "CA_KlamathRiver_TopoBathy_2018_D18" "$SRC_F32_MIN"

echo "ALL DONE. Output root: $OUTROOT"

