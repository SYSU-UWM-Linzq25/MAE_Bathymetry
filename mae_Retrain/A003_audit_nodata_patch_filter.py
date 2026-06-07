#!/usr/bin/env python3
"""Audit NoData-safe patch filtering before launching MAE training."""

import argparse
import csv
from pathlib import Path

import numpy as np

from util.dem_dataset import (
    _center_or_pad_triplet,
    _patch_status_from_masks,
    _read_dem_tiff,
    _read_lcc_mask_tiff,
    _valid_mask_from_values,
)


def read_list(path):
    return [
        line.strip() for line in Path(path).open()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem_list", required=True)
    ap.add_argument("--mask_list", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--input_size", type=int, default=336)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--nodata", type=float, default=-999999.0)
    ap.add_argument("--nodata_threshold", type=float, default=-9999.0)
    ap.add_argument("--min_prediction_patch_ratio", type=float, default=0.0001)
    ap.add_argument("--max_prediction_patch_ratio", type=float, default=0.80)
    ap.add_argument("--min_valid_visible_patch_ratio", type=float, default=0.70)
    args = ap.parse_args()

    dem_files = read_list(args.dem_list)
    mask_files = read_list(args.mask_list)
    if len(dem_files) != len(mask_files):
        raise RuntimeError(
            "List length mismatch: DEM={} mask={}".format(
                len(dem_files), len(mask_files)
            )
        )

    rows = []
    counts = {}
    for dem_path, mask_path in zip(dem_files, mask_files):
        row = {
            "dem_path": dem_path,
            "mask_path": mask_path,
            "valid_pixel_ratio": "",
            "candidate_patch_ratio": "",
            "prediction_patch_ratio": "",
            "valid_visible_patch_ratio": "",
            "ignored_patch_ratio": "",
            "status": "",
            "reason": "",
        }
        try:
            raw = _read_dem_tiff(dem_path)
            lcc = _read_lcc_mask_tiff(mask_path)
            if raw.shape != lcc.shape:
                raise ValueError(
                    "shape mismatch {} vs {}".format(raw.shape, lcc.shape)
                )
            valid = _valid_mask_from_values(
                raw, args.nodata, args.nodata_threshold
            ).astype(np.uint8)
            arr = np.where(valid > 0, raw, np.nan).astype(np.float32)
            _, lcc, valid = _center_or_pad_triplet(
                arr, lcc, valid, args.input_size, random_crop=False
            )
            st = _patch_status_from_masks(
                lcc, valid, patch_size=args.patch_size, threshold=0.5
            )
            n = int(st["valid"].size)
            pred_r = float(st["prediction"].sum() / n)
            vis_r = float(st["visible"].sum() / n)
            cand_r = float(st["candidate"].sum() / n)
            ign_r = float(st["ignored"].sum() / n)

            reason = "PASS"
            if int(st["prediction"].sum()) == 0:
                reason = "NO_USABLE_PREDICTION_PATCH"
            elif not (
                args.min_prediction_patch_ratio
                <= pred_r
                <= args.max_prediction_patch_ratio
            ):
                reason = "PREDICTION_PATCH_RATIO"
            elif vis_r < args.min_valid_visible_patch_ratio:
                reason = "VISIBLE_VALID_PATCH_RATIO"

            row.update({
                "valid_pixel_ratio": "{:.8f}".format(float(valid.mean())),
                "candidate_patch_ratio": "{:.8f}".format(cand_r),
                "prediction_patch_ratio": "{:.8f}".format(pred_r),
                "valid_visible_patch_ratio": "{:.8f}".format(vis_r),
                "ignored_patch_ratio": "{:.8f}".format(ign_r),
                "status": "PASS" if reason == "PASS" else "DROP",
                "reason": reason,
            })
        except Exception as exc:
            row["status"] = "DROP"
            row["reason"] = "READ_OR_GRID_ERROR: {!r}".format(exc)

        rows.append(row)
        counts[row["reason"]] = counts.get(row["reason"], 0) + 1

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Total pairs : {}".format(len(rows)))
    print("PASS        : {}".format(sum(r["status"] == "PASS" for r in rows)))
    print("DROP        : {}".format(sum(r["status"] == "DROP" for r in rows)))
    print("Reasons:")
    for key in sorted(counts):
        print("  {}: {}".format(key, counts[key]))
    print("CSV: {}".format(out))


if __name__ == "__main__":
    main()
