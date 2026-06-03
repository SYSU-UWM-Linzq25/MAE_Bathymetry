#!/usr/bin/env python3
"""
Extract best/worst Stage3 bathy+LCC tile pairs from evaluation CSVs for GIS review.

Inputs are evaluation CSVs such as:
  - top200_errors_val.csv
  - best200_nontrivial_tiles_val.csv

The script copies or symlinks the bathy GeoTIFF and its paired LCC mask into a review folder,
with stable prefixed names and a manifest CSV that preserves metrics from the source CSV.
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_stem_from_bath(path_or_name: str) -> str:
    name = Path(str(path_or_name)).name
    if name.endswith('.tif'):
        name = name[:-4]
    name = name.replace('Select_tile_Basin_1m_', '')
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', name)
    return name


def bath_to_lcc_name(bath_name: str) -> str:
    """Convert bath tile filename to paired LCC mask filename."""
    name = Path(str(bath_name)).name
    name = name.replace('Select_tile_Basin_1m_', 'Select_tile_1m_')
    if name.endswith('.tif'):
        name = name[:-4]
    if not name.endswith('_LCC_Mask'):
        name = f'{name}_LCC_Mask'
    return f'{name}.tif'


def resolve_file(value: str, base_dir: Optional[Path], fallback_name: Optional[str] = None) -> Optional[Path]:
    """Resolve either an absolute path or a filename under base_dir."""
    if value is not None and str(value) != 'nan':
        p = Path(str(value))
        if p.is_absolute() and p.exists():
            return p
        if base_dir is not None:
            q = base_dir / p.name
            if q.exists():
                return q
        if p.exists():
            return p
    if fallback_name and base_dir is not None:
        q = base_dir / fallback_name
        if q.exists():
            return q
    return None


def link_or_copy(src: Path, dst: Path, mode: str):
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'symlink':
        os.symlink(src, dst)
    else:
        raise ValueError(f'Unknown mode: {mode}')


def process_csv(args, csv_path: Path, label: str):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise RuntimeError(f'Empty CSV: {csv_path}')

    # Common columns from our evaluation scripts can be: path, file, bath_path, mask_path, mask_file.
    bath_col = pick_col(df, ['path', 'bath_path', 'dem_path', 'file'])
    mask_col = pick_col(df, ['mask_path', 'lcc_path', 'mask_file'])
    if bath_col is None:
        raise KeyError(f'Cannot identify bath file column in {csv_path}. Columns={list(df.columns)}')

    df = df.copy()
    # For top errors, CSV is usually already sorted descending by rmse_m_mask.
    # For best, already sorted ascending. We preserve CSV order unless --sort_metric is provided.
    if args.sort_metric:
        if args.sort_metric not in df.columns:
            raise KeyError(f'--sort_metric {args.sort_metric} not in {csv_path}. Columns={list(df.columns)}')
        df = df.sort_values(args.sort_metric, ascending=args.sort_ascending)

    df = df.head(args.n).reset_index(drop=True)

    out_root = Path(args.out_dir) / label
    out_bath = out_root / 'bath'
    out_lcc = out_root / 'lcc'
    ensure_dir(out_bath)
    ensure_dir(out_lcc)

    bath_dir = Path(args.bath_dir) if args.bath_dir else None
    lcc_dir = Path(args.lcc_dir) if args.lcc_dir else None

    records = []
    missing = []

    for i, row in df.iterrows():
        bath_val = str(row[bath_col])
        bath_src = resolve_file(bath_val, bath_dir)
        if bath_src is None:
            missing.append({'label': label, 'rank': i + 1, 'missing': 'bath', 'value': bath_val})
            continue

        if mask_col is not None and pd.notna(row.get(mask_col, None)):
            mask_val = str(row[mask_col])
            mask_src = resolve_file(mask_val, lcc_dir)
        else:
            mask_src = None

        if mask_src is None:
            expected_lcc = bath_to_lcc_name(bath_src.name)
            mask_src = resolve_file('', lcc_dir, fallback_name=expected_lcc)

        if mask_src is None:
            missing.append({'label': label, 'rank': i + 1, 'missing': 'lcc', 'value': bath_to_lcc_name(bath_src.name)})
            continue

        stem = safe_stem_from_bath(bath_src.name)
        metric_txt = ''
        if 'rmse_m_mask' in row and pd.notna(row['rmse_m_mask']):
            metric_txt = f'_rmse{float(row["rmse_m_mask"]):.3f}'
        prefix = f'{i+1:03d}_{label}{metric_txt}_{stem}'

        bath_dst = out_bath / f'{prefix}_BATH.tif'
        lcc_dst = out_lcc / f'{prefix}_LCC.tif'

        link_or_copy(bath_src, bath_dst, args.mode)
        link_or_copy(mask_src, lcc_dst, args.mode)

        rec = row.to_dict()
        rec.update({
            'review_label': label,
            'review_rank': i + 1,
            'bath_src': str(bath_src),
            'lcc_src': str(mask_src),
            'bath_review_file': str(bath_dst),
            'lcc_review_file': str(lcc_dst),
        })
        records.append(rec)

    manifest = pd.DataFrame(records)
    manifest.to_csv(out_root / f'manifest_{label}.csv', index=False)

    if missing:
        pd.DataFrame(missing).to_csv(out_root / f'missing_{label}.csv', index=False)

    print(f'[{label}] CSV: {csv_path}')
    print(f'[{label}] Requested: {len(df)}')
    print(f'[{label}] Extracted: {len(records)}')
    print(f'[{label}] Missing: {len(missing)}')
    print(f'[{label}] Output: {out_root}')

    return manifest


def main():
    parser = argparse.ArgumentParser(description='Extract evaluation-selected bathy/LCC tiles for GIS review.')
    parser.add_argument('--eval_dir', required=True, help='Evaluation result directory containing CSVs')
    parser.add_argument('--out_dir', required=True, help='Output review directory')
    parser.add_argument('--bath_dir', required=True, help='Original bath tile directory')
    parser.add_argument('--lcc_dir', required=True, help='Original LCC mask directory')
    parser.add_argument('--n', type=int, default=100, help='Number of rows to extract from each CSV')
    parser.add_argument('--mode', choices=['copy', 'symlink'], default='copy')
    parser.add_argument('--worst_csv', default='top200_errors_val.csv')
    parser.add_argument('--best_csv', default='best200_nontrivial_tiles_val.csv')
    parser.add_argument('--include_best_simple', action='store_true', help='Also extract best200_tiles_val.csv')
    parser.add_argument('--include_median', action='store_true', help='Also extract median040_tiles_val.csv')
    parser.add_argument('--sort_metric', default='', help='Optional metric to sort within each CSV before extraction')
    parser.add_argument('--sort_ascending', action='store_true', help='Use ascending sort if --sort_metric is set')
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    manifests = []

    worst_path = eval_dir / args.worst_csv
    if worst_path.exists():
        manifests.append(process_csv(args, worst_path, 'worst'))
    else:
        print(f'[WARN] Missing worst CSV: {worst_path}')

    best_path = eval_dir / args.best_csv
    if best_path.exists():
        manifests.append(process_csv(args, best_path, 'best_nontrivial'))
    else:
        print(f'[WARN] Missing best CSV: {best_path}')

    if args.include_best_simple:
        p = eval_dir / 'best200_tiles_val.csv'
        if p.exists():
            manifests.append(process_csv(args, p, 'best_simple'))
        else:
            print(f'[WARN] Missing best simple CSV: {p}')

    if args.include_median:
        p = eval_dir / 'median040_tiles_val.csv'
        if p.exists():
            manifests.append(process_csv(args, p, 'median'))
        else:
            print(f'[WARN] Missing median CSV: {p}')

    if manifests:
        all_manifest = pd.concat(manifests, ignore_index=True)
        all_manifest.to_csv(out_dir / 'manifest_all.csv', index=False)
        print(f'[ALL] Wrote: {out_dir / "manifest_all.csv"}')

    print('=== DONE ===')


if __name__ == '__main__':
    main()
