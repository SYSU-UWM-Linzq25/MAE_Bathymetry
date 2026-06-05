#!/bin/bash
set -euo pipefail

# ============================================================
# Randomly sample 5 paired training tiles per river, and also
# copy a fixed set of previously failed/high-priority tile IDs
# for focused visual inspection.
#
# This script only COPIES files. It never deletes or moves source data.
# ============================================================

BASE=/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE/Tiles_1m
TRAIN_DIR=$BASE/Train_tile
MASK_DIR=$BASE/LCC_Mask

N_PER_RIVER=${N_PER_RIVER:-5}
SEED=${SEED:-20260604}
TIME_TAG=$(date +%Y%m%d_%H%M%S)

OUT_DIR=${OUT_DIR:-$BASE/VisualCheck_5pairs_per_river_plus_priority_${TIME_TAG}}
ARCHIVE=${ARCHIVE:-${OUT_DIR}.tar.gz}

echo "============================================================"
echo "Sample paired 1m tiles for visual inspection"
echo "BASE         = $BASE"
echo "TRAIN_DIR    = $TRAIN_DIR"
echo "MASK_DIR     = $MASK_DIR"
echo "N_PER_RIVER  = $N_PER_RIVER"
echo "SEED         = $SEED"
echo "OUT_DIR      = $OUT_DIR"
echo "ARCHIVE      = $ARCHIVE"
echo "============================================================"

[[ -d "$TRAIN_DIR" ]] || { echo "ERROR: missing $TRAIN_DIR"; exit 1; }
[[ -d "$MASK_DIR"  ]] || { echo "ERROR: missing $MASK_DIR"; exit 1; }

if [[ -e "$OUT_DIR" ]]; then
    echo "ERROR: output already exists:"
    echo "$OUT_DIR"
    echo "Set OUT_DIR to a new path or remove the old output manually."
    exit 1
fi

mkdir -p "$OUT_DIR"

python3 - "$TRAIN_DIR" "$MASK_DIR" "$OUT_DIR" "$N_PER_RIVER" "$SEED" <<'PY'
import csv
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

train_dir = Path(sys.argv[1])
mask_dir = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
n_per_river = int(sys.argv[4])
seed = int(sys.argv[5])

pattern = re.compile(
    r"^Select_tile_Basin_1m_(?P<river>.+)_ID(?P<tile_id>\d+)\.tif$",
    re.IGNORECASE,
)

# ------------------------------------------------------------
# Fixed tiles that previously showed severe failures.
# The script searches these exact IDs in the NEW extraction output:
#   Processed_Results/Tiles_for_MAE/Tiles_1m/Train_tile
# and copies their corresponding NEW masks from LCC_Mask.
# ------------------------------------------------------------
priority_train_names = [
    "Select_tile_Basin_1m_OR_MKRC_Topobathy_2021_ID881.tif",
    "Select_tile_Basin_1m_OR_MKRC_Topobathy_2021_ID391.tif",
    "Select_tile_Basin_1m_OR_MKRC_Topobathy_2021_ID409.tif",
    "Select_tile_Basin_1m_OR_MKRC_Topobathy_2021_ID361.tif",
    "Select_tile_Basin_1m_OR_MKRC_Topobathy_2021_ID505.tif",
    "Select_tile_Basin_1m_OR_MKRC_Topobathy_2021_ID990.tif",
    "Select_tile_Basin_1m_OR_MKRC_Topobathy_2021_ID1050.tif",
    "Select_tile_Basin_1m_OR_MKRC_Topobathy_2021_ID506.tif",
    "Select_tile_Basin_1m_OR_MKRC_Topobathy_2021_ID1074.tif",
    "Select_tile_Basin_1m_CA_KlamathRiver_TopoBathy_2018_D18_ID4670.tif",
    "Select_tile_Basin_1m_CA_KlamathRiver_TopoBathy_2018_D18_ID4674.tif",
    "Select_tile_Basin_1m_CA_KlamathRiver_TopoBathy_2018_D18_ID4667.tif",
    "Select_tile_Basin_1m_OR_SantiamRiverTB_Topobathy_1_D23_ID1092.tif",
    "Select_tile_Basin_1m_OR_SantiamRiverTB_Topobathy_1_D23_ID1115.tif",
    "Select_tile_Basin_1m_OR_SantiamRiverTB_Topobathy_1_D23_ID1095.tif",
]

# ============================================================
# Part A. Random 5 paired tiles per river
# ============================================================
pairs_by_river = defaultdict(list)
missing_masks = []
unmatched_names = []

for train_file in train_dir.iterdir():
    if not train_file.is_file() or train_file.suffix.lower() not in {".tif", ".tiff"}:
        continue

    match = pattern.match(train_file.name)
    if match is None:
        if len(unmatched_names) < 20:
            unmatched_names.append(train_file.name)
        continue

    river = match.group("river")
    tile_id = int(match.group("tile_id"))

    mask_name = f"Select_tile_1m_{river}_ID{tile_id}_LCC_Mask.tif"
    mask_file = mask_dir / mask_name

    if not mask_file.is_file():
        missing_masks.append((train_file.name, mask_name))
        continue

    pairs_by_river[river].append((tile_id, train_file, mask_file))

if not pairs_by_river:
    raise RuntimeError("No paired Train_tile/LCC_Mask files found.")

rng = random.Random(seed)
manifest = []
summary = []

random_root = out_dir / "Random_5_Per_River"

for river in sorted(pairs_by_river):
    available = sorted(pairs_by_river[river], key=lambda x: x[0])
    selected = rng.sample(available, min(n_per_river, len(available)))
    selected.sort(key=lambda x: x[0])

    river_dir = random_root / river
    train_out = river_dir / "Bathy3DEP"
    mask_out = river_dir / "Mask"

    train_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    for tile_id, train_file, mask_file in selected:
        dst_train = train_out / train_file.name
        dst_mask = mask_out / mask_file.name
        shutil.copy2(train_file, dst_train)
        shutil.copy2(mask_file, dst_mask)

        manifest.append({
            "river": river,
            "tile_id": tile_id,
            "train_tile": train_file.name,
            "mask_tile": mask_file.name,
            "train_source": str(train_file),
            "mask_source": str(mask_file),
            "train_copy": str(dst_train),
            "mask_copy": str(dst_mask),
        })

    summary.append({
        "river": river,
        "available_pairs": len(available),
        "selected_pairs": len(selected),
        "requested_pairs": n_per_river,
        "status": "PASS" if len(selected) == n_per_river else "FEWER_THAN_REQUESTED",
    })

    print(f"[RANDOM] {river}: available={len(available)}, selected={len(selected)}")

with (out_dir / "selected_tile_pairs_random.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "river", "tile_id", "train_tile", "mask_tile",
            "train_source", "mask_source", "train_copy", "mask_copy",
        ],
    )
    writer.writeheader()
    writer.writerows(manifest)

with (out_dir / "summary_by_river_random.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "river", "available_pairs", "selected_pairs",
            "requested_pairs", "status",
        ],
    )
    writer.writeheader()
    writer.writerows(summary)

# ============================================================
# Part B. Copy all previously failed/high-priority IDs
# ============================================================
priority_root = out_dir / "Priority_Previous_Failures"
priority_rows = []
priority_copied = 0
priority_missing = 0

print("")
print("============================================================")
print("Copy priority previous-failure cases")
print("============================================================")

for train_name in priority_train_names:
    match = pattern.match(train_name)
    if match is None:
        priority_rows.append({
            "requested_train_tile": train_name,
            "river": "",
            "tile_id": "",
            "train_exists": 0,
            "mask_exists": 0,
            "status": "INVALID_NAME_PATTERN",
            "train_source": "",
            "mask_source": "",
        })
        priority_missing += 1
        print(f"[PRIORITY][INVALID NAME] {train_name}")
        continue

    river = match.group("river")
    tile_id = int(match.group("tile_id"))
    train_file = train_dir / train_name
    mask_name = f"Select_tile_1m_{river}_ID{tile_id}_LCC_Mask.tif"
    mask_file = mask_dir / mask_name

    train_exists = train_file.is_file()
    mask_exists = mask_file.is_file()

    if train_exists and mask_exists:
        river_dir = priority_root / river
        train_out = river_dir / "Bathy3DEP"
        mask_out = river_dir / "Mask"
        train_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        shutil.copy2(train_file, train_out / train_file.name)
        shutil.copy2(mask_file, mask_out / mask_file.name)
        status = "COPIED"
        priority_copied += 1
        print(f"[PRIORITY][COPIED] {river} ID{tile_id}")
    else:
        missing_parts = []
        if not train_exists:
            missing_parts.append("TRAIN_MISSING")
        if not mask_exists:
            missing_parts.append("MASK_MISSING")
        status = "+".join(missing_parts)
        priority_missing += 1
        print(f"[PRIORITY][{status}] {river} ID{tile_id}")

    priority_rows.append({
        "requested_train_tile": train_name,
        "river": river,
        "tile_id": tile_id,
        "train_exists": int(train_exists),
        "mask_exists": int(mask_exists),
        "status": status,
        "train_source": str(train_file),
        "mask_source": str(mask_file),
    })

with (out_dir / "priority_previous_failures.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "requested_train_tile", "river", "tile_id",
            "train_exists", "mask_exists", "status",
            "train_source", "mask_source",
        ],
    )
    writer.writeheader()
    writer.writerows(priority_rows)

# Human-readable note
with (out_dir / "README.txt").open("w", encoding="utf-8") as f:
    f.write("Visual-check package for newly extracted 1m MAE training tiles.\n\n")
    f.write("Random_5_Per_River/: five randomly selected paired Bathy3DEP/Mask tiles per river.\n")
    f.write("Priority_Previous_Failures/: exact tile IDs that previously produced severe failures.\n")
    f.write("All copied files come from the NEW extraction outputs under Tiles_for_MAE/Tiles_1m.\n")
    f.write("The script only copies files; source files are not modified, moved, or deleted.\n")

print("")
print("============================================================")
print("Final summary")
print("============================================================")
print(f"Rivers with valid random pairs : {len(pairs_by_river)}")
print(f"Random paired tiles copied     : {len(manifest)}")
print(f"Priority cases requested       : {len(priority_train_names)}")
print(f"Priority pairs copied          : {priority_copied}")
print(f"Priority cases missing         : {priority_missing}")
print(f"Unmatched training names       : {len(unmatched_names)}")
print(f"Training tiles missing masks   : {len(missing_masks)}")

if missing_masks:
    print("First missing-mask examples:")
    for train_name, mask_name in missing_masks[:10]:
        print(f"  {train_name} -> {mask_name}")
PY

echo
echo "============================================================"
echo "Create archive"
echo "============================================================"

tar -czf "$ARCHIVE" -C "$BASE" "$(basename "$OUT_DIR")"

echo
echo "============================================================"
echo "DONE"
echo "Output folder:"
echo "$OUT_DIR"
echo
echo "Archive for download:"
echo "$ARCHIVE"
echo
echo "Random sample manifest:"
echo "$OUT_DIR/selected_tile_pairs_random.csv"
echo
echo "Priority-case manifest:"
echo "$OUT_DIR/priority_previous_failures.csv"
echo "============================================================"
