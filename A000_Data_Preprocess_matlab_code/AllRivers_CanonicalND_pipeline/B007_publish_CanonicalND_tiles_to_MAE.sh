#!/usr/bin/env bash
# Publish staged CanonicalND tile/mask products only after all audits pass.
set -euo pipefail

ROOT_PR=/tank/data/SFS/xinyis/data/bathymetry/Processed_Results
ROOT_MAE=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
SRC=$ROOT_PR/Tiles_for_MAE_CanonicalND/Tiles_1m
DST_TRAIN=$ROOT_MAE/Data/Tiles_for_Training_1m/1m_Tiles
DST_MASK=$ROOT_MAE/Data/TilesMask_for_Training_1m/1m_Tiles
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP=$ROOT_MAE/Data/backup_before_AllRivers_CanonicalND_${STAMP}

RIVERS=(
  BadgerFinNull
  CA_KlamathRiver_TopoBathy_2018_D18
  CO_UpperColorado_Topobathy_1_2020
  Estabrook_Combined
  KewaFix2Null
  Kletzch_Combined_UpMax3Null
  MD_PotomacRiver_Bathy_2019
  NE_Niobrara_Topobathy_2018
  OR_MKRC_Topobathy_2021
  OR_SantiamRiverTB_Topobathy_1_D23
  WA_ChehalisRiverTB_Topobathy_1_D23
  WA_Nisqually_Bathymetric_2020
)

[[ -d "$SRC/Train_tile" && -d "$SRC/LCC_Mask" ]] || {
  echo "[ERROR] Missing staged source: $SRC" >&2
  exit 2
}

echo "=== Pre-publish filename/count checks ==="
for river in "${RIVERS[@]}"; do
  old_t=$(find "$DST_TRAIN" -maxdepth 1 -type f -name "*_${river}_ID*.tif" | wc -l)
  new_t=$(find "$SRC/Train_tile" -maxdepth 1 -type f -name "*_${river}_ID*.tif" | wc -l)
  old_m=$(find "$DST_MASK" -maxdepth 1 -type f -name "*_${river}_ID*_LCC_Mask.tif" | wc -l)
  new_m=$(find "$SRC/LCC_Mask" -maxdepth 1 -type f -name "*_${river}_ID*_LCC_Mask.tif" | wc -l)

  printf "%-48s old/new tiles=%d/%d masks=%d/%d\n" "$river" "$old_t" "$new_t" "$old_m" "$new_m"

  [[ "$new_t" -gt 0 && "$new_t" -eq "$new_m" ]] || {
    echo "[ERROR] Staged tile/mask mismatch for $river" >&2
    exit 3
  }
  [[ "$old_t" -eq "$new_t" && "$old_m" -eq "$new_m" ]] || {
    echo "[ERROR] Old/new count mismatch for $river" >&2
    exit 4
  }
done

mkdir -p "$BACKUP/Train_tile" "$BACKUP/LCC_Mask"

echo "=== Backup and publish ==="
for river in "${RIVERS[@]}"; do
  find "$DST_TRAIN" -maxdepth 1 -type f -name "*_${river}_ID*.tif" -print0 |
    while IFS= read -r -d '' f; do cp -p "$f" "$BACKUP/Train_tile/"; done

  find "$DST_MASK" -maxdepth 1 -type f -name "*_${river}_ID*_LCC_Mask.tif" -print0 |
    while IFS= read -r -d '' f; do cp -p "$f" "$BACKUP/LCC_Mask/"; done

  find "$SRC/Train_tile" -maxdepth 1 -type f -name "*_${river}_ID*.tif" -print0 |
    while IFS= read -r -d '' f; do cp -pf "$f" "$DST_TRAIN/"; done

  find "$SRC/LCC_Mask" -maxdepth 1 -type f -name "*_${river}_ID*_LCC_Mask.tif" -print0 |
    while IFS= read -r -d '' f; do cp -pf "$f" "$DST_MASK/"; done
done

echo "Published successfully."
echo "Backup: $BACKUP"
