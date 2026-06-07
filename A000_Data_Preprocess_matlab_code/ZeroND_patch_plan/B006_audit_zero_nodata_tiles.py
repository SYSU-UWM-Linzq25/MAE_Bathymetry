#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tile_dir', required=True)
    ap.add_argument('--mask_dir', required=True)
    ap.add_argument('--river', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--nodata_threshold', type=float, default=-9999.0)
    ap.add_argument('--zero_tol', type=float, default=1e-8)
    ap.add_argument('--fail_on_zero', action='store_true')
    args = ap.parse_args()

    tile_dir = Path(args.tile_dir)
    mask_dir = Path(args.mask_dir)
    pattern = re.compile(rf'^Select_tile_Basin_1m_{re.escape(args.river)}_ID(\d+)\.tif$')

    rows = []
    n_zero_tiles = 0
    n_missing_masks = 0

    for tile_path in sorted(tile_dir.glob(f'Select_tile_Basin_1m_{args.river}_ID*.tif')):
        match = pattern.match(tile_path.name)
        if not match:
            continue
        point_id = int(match.group(1))
        mask_path = mask_dir / f'Select_tile_1m_{args.river}_ID{point_id}_LCC_Mask.tif'
        if not mask_path.is_file():
            n_missing_masks += 1
            continue

        with Image.open(tile_path) as im:
            tile = np.asarray(im, dtype=np.float64)
        with Image.open(mask_path) as im:
            mask = np.asarray(im) == 1

        finite = np.isfinite(tile)
        nodata = (~finite) | (tile <= args.nodata_threshold)
        zero = finite & (~nodata) & (np.abs(tile) <= args.zero_tol)
        valid = finite & (~nodata) & (~zero)

        n_zero = int(zero.sum())
        if n_zero:
            n_zero_tiles += 1

        values = tile[valid]
        rows.append({
            'PointID': point_id,
            'tile_file': tile_path.name,
            'mask_file': mask_path.name,
            'N_zero': n_zero,
            'N_zero_inside_mask': int((zero & mask).sum()),
            'N_zero_outside_mask': int((zero & ~mask).sum()),
            'N_nodata': int(nodata.sum()),
            'N_valid': int(valid.sum()),
            'valid_min': float(values.min()) if values.size else np.nan,
            'valid_max': float(values.max()) if values.size else np.nan,
            'valid_median': float(np.median(values)) if values.size else np.nan,
        })

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f'Tiles audited      : {len(rows)}')
    print(f'Tiles with zero    : {n_zero_tiles}')
    print(f'Missing masks      : {n_missing_masks}')
    print(f'CSV                : {out}')

    if n_missing_masks:
        raise SystemExit(2)
    if args.fail_on_zero and n_zero_tiles:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
