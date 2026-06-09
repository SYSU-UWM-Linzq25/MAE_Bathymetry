#!/usr/bin/env python3
"""Compare old and canonical-NoData MAE tiles without rasterio/GDAL Python."""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PATTERN = re.compile(r"^Select_tile_Basin_1m_(.+)_ID(\d+)\.tif$")


def read_array(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_base", required=True)
    ap.add_argument("--new_base", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--nodata_threshold", type=float, default=-9999.0)
    ap.add_argument("--zero_tol", type=float, default=1e-8)
    ap.add_argument("--value_tol", type=float, default=1e-5)
    args = ap.parse_args()

    old_base = Path(args.old_base)
    new_base = Path(args.new_base)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    old_train = old_base / "Train_tile"
    old_mask = old_base / "LCC_Mask"
    new_train = new_base / "Train_tile"
    new_mask = new_base / "LCC_Mask"

    rows = []
    for new_path in sorted(new_train.glob("Select_tile_Basin_1m_*_ID*.tif")):
        m = PATTERN.match(new_path.name)
        if not m:
            continue
        river, point_id = m.group(1), int(m.group(2))
        old_path = old_train / new_path.name
        old_mask_path = old_mask / f"Select_tile_1m_{river}_ID{point_id}_LCC_Mask.tif"
        new_mask_path = new_mask / f"Select_tile_1m_{river}_ID{point_id}_LCC_Mask.tif"

        if not old_path.is_file() or not old_mask_path.is_file() or not new_mask_path.is_file():
            rows.append({
                "river": river, "PointID": point_id, "status": "MISSING_PAIR",
                "old_tile": str(old_path), "new_tile": str(new_path),
            })
            continue

        old = read_array(old_path)
        new = read_array(new_path)
        om = read_array(old_mask_path) == 1
        nm = read_array(new_mask_path) == 1
        if old.shape != new.shape or om.shape != nm.shape:
            rows.append({
                "river": river, "PointID": point_id, "status": "SHAPE_MISMATCH",
                "old_tile": str(old_path), "new_tile": str(new_path),
            })
            continue

        old_nd = (~np.isfinite(old)) | (old <= args.nodata_threshold)
        new_nd = (~np.isfinite(new)) | (new <= args.nodata_threshold)
        old_zero = (~old_nd) & (np.abs(old) <= args.zero_tol)
        new_zero = (~new_nd) & (np.abs(new) <= args.zero_tol)
        common_valid = (~old_nd) & (~new_nd)
        diff = np.abs(new - old)
        changed = common_valid & (diff > args.value_tol)

        rows.append({
            "river": river,
            "PointID": point_id,
            "status": "PASS",
            "N_old_zero": int(old_zero.sum()),
            "N_new_zero": int(new_zero.sum()),
            "N_old_zero_to_new_nodata": int((old_zero & new_nd).sum()),
            "N_old_zero_to_new_nonzero": int((old_zero & (~new_nd) & (~new_zero)).sum()),
            "N_old_nodata": int(old_nd.sum()),
            "N_new_nodata": int(new_nd.sum()),
            "N_common_valid_changed": int(changed.sum()),
            "MaxAbsDiff_common_valid": float(diff[common_valid].max()) if common_valid.any() else np.nan,
            "N_mask_changed": int((om != nm).sum()),
            "N_new_zero_inside_mask": int((new_zero & nm).sum()),
            "N_new_zero_outside_mask": int((new_zero & ~nm).sum()),
            "old_tile": str(old_path),
            "new_tile": str(new_path),
        })

    per_tile = out / "canonicalND_old_new_per_tile.csv"
    if rows:
        fields = sorted({k for r in rows for k in r})
        with per_tile.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[r.get("river", "UNKNOWN")].append(r)

    summary_rows = []
    numeric_sum = [
        "N_old_zero", "N_new_zero", "N_old_zero_to_new_nodata",
        "N_old_zero_to_new_nonzero", "N_old_nodata", "N_new_nodata",
        "N_common_valid_changed", "N_mask_changed",
        "N_new_zero_inside_mask", "N_new_zero_outside_mask",
    ]
    for river, rs in sorted(groups.items()):
        good = [r for r in rs if r.get("status") == "PASS"]
        summary = {
            "river": river,
            "N_rows": len(rs),
            "N_pass": len(good),
            "N_failed": len(rs) - len(good),
            "N_tiles_with_new_zero": sum(int(r.get("N_new_zero", 0)) > 0 for r in good),
            "N_tiles_with_common_valid_change": sum(
                int(r.get("N_common_valid_changed", 0)) > 0 for r in good
            ),
            "MaxAbsDiff_common_valid": max(
                [float(r["MaxAbsDiff_common_valid"]) for r in good
                 if np.isfinite(float(r["MaxAbsDiff_common_valid"]))],
                default=np.nan,
            ),
        }
        for key in numeric_sum:
            summary[key] = sum(int(r.get(key, 0)) for r in good)
        summary_rows.append(summary)

    per_river = out / "canonicalND_old_new_per_river.csv"
    if summary_rows:
        with per_river.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            w.writeheader()
            w.writerows(summary_rows)

    print(f"Tiles compared : {len(rows)}")
    print(f"Per-tile CSV   : {per_tile}")
    print(f"Per-river CSV  : {per_river}")


if __name__ == "__main__":
    main()
