#!/usr/bin/env python3
"""Prepare a MAE v2 train/val split stratified by river-level tile_std_safe.

Purpose:
  Split whole rivers, not random tiles, while forcing both train and val
  to contain low / medium / high tile_std_safe river groups.

Inputs under tile_root:
  Train_tile/
  Hidden_Mask/
  Loss_Mask_Pixel/

Outputs:
  train_tiles.txt, val_tiles.txt, test_tiles.txt
  train_hidden.txt, val_hidden.txt, test_hidden.txt
  train_loss.txt, val_loss.txt, test_loss.txt
  split_manifest.csv
  river_std_summary.csv
  split_summary.txt

Default strategy:
  1) compute tile_std_safe = std(visible valid DEM pixels) * std_scale;
  2) summarize each river by median tile_std_safe;
  3) sort rivers and divide into low / medium / high tertiles;
  4) select val_per_bin rivers from each tertile for validation.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


TILE_RE = re.compile(
    r"^Select_tile_(?:Basin_)?(?P<res>\d+)m_(?P<river>.+)_ID(?P<id>\d+)(?:_(?P<suffix>[^.]+))?\.tif$",
    re.IGNORECASE,
)


def parse_tile(path: Path) -> Tuple[str, str, int]:
    m = TILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized tile name: {path.name}")
    res = f"{int(m.group('res'))}m"
    river = m.group("river")
    tile_id = int(m.group("id"))
    return res, river, tile_id


def key_from_path(path: Path) -> str:
    res, river, tile_id = parse_tile(path)
    return f"{res}_{river}_ID{tile_id}"


def safe_read(path: Path) -> np.ndarray:
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.read(1)
    except Exception:
        import tifffile
        return tifffile.imread(path)


def list_files(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(folder.glob("*.tif")):
        try:
            k = key_from_path(p)
        except ValueError:
            continue
        if k in out:
            raise RuntimeError(f"Duplicate key {k}: {out[k]} and {p}")
        out[k] = p.resolve()
    return out


def compute_std_safe(tile_path: Path, hidden_path: Path, nodata: float, nodata_threshold: float,
                     std_scale: float, eps: float, visible_only: bool) -> Tuple[float, float, int]:
    dem = safe_read(tile_path).astype(np.float64)
    hidden = safe_read(hidden_path).astype(np.float32)

    valid = np.isfinite(dem) & (dem > nodata_threshold) & (dem != nodata)
    if visible_only:
        use = valid & (hidden < 0.5)
    else:
        use = valid

    if int(use.sum()) < 2:
        use = valid

    vals = dem[use]
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan"), float("nan"), int(vals.size)

    mean = float(vals.mean())
    std = float(vals.std())
    std_safe = max(std * std_scale, eps)
    return mean, std_safe, int(vals.size)


def write_list(path: Path, items: List[Path]) -> None:
    path.write_text("\n".join(str(p) for p in items) + ("\n" if items else ""))


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--std_scale", type=float, default=1.5)
    ap.add_argument("--tile_norm_eps", type=float, default=1e-3)
    ap.add_argument("--nodata", type=float, default=-999999.0)
    ap.add_argument("--nodata_threshold", type=float, default=-9999.0)
    ap.add_argument("--visible_only", action="store_true", default=True)
    ap.add_argument("--bin_stat", choices=["median", "mean"], default="median")
    ap.add_argument("--val_per_bin", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--val_rivers",
        nargs="*",
        default=None,
        help=(
            "Optional manual validation river list. If provided, these whole rivers "
            "are used as validation and val_per_bin random selection is skipped."
        ),
    )
    args = ap.parse_args()

    tile_root = Path(args.tile_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = tile_root / "Train_tile"
    hidden_dir = tile_root / "Hidden_Mask"
    loss_dir = tile_root / "Loss_Mask_Pixel"

    tiles = list_files(train_dir)
    hidden = list_files(hidden_dir)
    loss = list_files(loss_dir)

    keys = sorted(set(tiles) & set(hidden) & set(loss))
    if not keys:
        raise RuntimeError("No matched Train_tile / Hidden_Mask / Loss_Mask_Pixel triples found.")

    rows = []
    by_river: Dict[str, List[dict]] = {}
    for i, k in enumerate(keys, start=1):
        res, river, tile_id = parse_tile(tiles[k])
        mean, std_safe, n_used = compute_std_safe(
            tiles[k], hidden[k],
            nodata=args.nodata,
            nodata_threshold=args.nodata_threshold,
            std_scale=args.std_scale,
            eps=args.tile_norm_eps,
            visible_only=args.visible_only,
        )
        row = {
            "key": k,
            "res": res,
            "river": river,
            "tile_id": tile_id,
            "tile_path": str(tiles[k]),
            "hidden_path": str(hidden[k]),
            "loss_path": str(loss[k]),
            "tile_mean_visible_m": mean,
            "tile_std_safe": std_safe,
            "n_visible_pixels_for_std": n_used,
        }
        rows.append(row)
        by_river.setdefault(river, []).append(row)
        if i % 500 == 0:
            print(f"[STD] processed {i}/{len(keys)} tiles")

    river_rows = []
    for river, rr in sorted(by_river.items()):
        vals = np.asarray([r["tile_std_safe"] for r in rr], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            stat = float("nan")
            mean = float("nan")
            median = float("nan")
            p90 = float("nan")
        else:
            mean = float(vals.mean())
            median = float(np.median(vals))
            p90 = float(np.percentile(vals, 90))
            stat = median if args.bin_stat == "median" else mean
        river_rows.append({
            "river": river,
            "n_tiles": len(rr),
            "mean_tile_std_safe": mean,
            "median_tile_std_safe": median,
            "p90_tile_std_safe": p90,
            "bin_sort_value": stat,
        })

    river_rows.sort(key=lambda r: (math.inf if not math.isfinite(float(r["bin_sort_value"])) else float(r["bin_sort_value"])))
    n = len(river_rows)
    for idx, r in enumerate(river_rows):
        if idx < n / 3:
            r["std_bin"] = "low"
        elif idx < 2 * n / 3:
            r["std_bin"] = "mid"
        else:
            r["std_bin"] = "high"

    all_rivers = {r["river"] for r in river_rows}
    if args.val_rivers:
        val_rivers = set(args.val_rivers)
        missing = sorted(val_rivers - all_rivers)
        if missing:
            raise RuntimeError(f"Manual --val_rivers not found in tile set: {missing}")
        split_mode = "manual_val_rivers"
    else:
        rng = random.Random(args.seed)
        val_rivers = set()
        for bin_name in ["low", "mid", "high"]:
            group = [r["river"] for r in river_rows if r["std_bin"] == bin_name]
            group_sorted = sorted(group)
            rng.shuffle(group_sorted)
            take = min(max(args.val_per_bin, 1), max(len(group_sorted) - 1, 1))
            val_rivers.update(group_sorted[:take])
        split_mode = "random_val_per_bin"

    for r in river_rows:
        r["split"] = "val" if r["river"] in val_rivers else "train"

    for row in rows:
        rr = next(r for r in river_rows if r["river"] == row["river"])
        row["std_bin"] = rr["std_bin"]
        row["split"] = rr["split"]

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    write_list(out_dir / "train_tiles.txt", [Path(r["tile_path"]) for r in train_rows])
    write_list(out_dir / "val_tiles.txt", [Path(r["tile_path"]) for r in val_rows])
    write_list(out_dir / "test_tiles.txt", [])

    write_list(out_dir / "train_hidden.txt", [Path(r["hidden_path"]) for r in train_rows])
    write_list(out_dir / "val_hidden.txt", [Path(r["hidden_path"]) for r in val_rows])
    write_list(out_dir / "test_hidden.txt", [])

    write_list(out_dir / "train_loss.txt", [Path(r["loss_path"]) for r in train_rows])
    write_list(out_dir / "val_loss.txt", [Path(r["loss_path"]) for r in val_rows])
    write_list(out_dir / "test_loss.txt", [])

    write_csv(out_dir / "split_manifest.csv", rows)
    write_csv(out_dir / "river_std_summary.csv", river_rows)

    summary = []
    summary.append("A017 std-stratified river split")
    summary.append(f"tile_root={tile_root}")
    summary.append(f"out_dir={out_dir}")
    summary.append(f"n_total_tiles={len(rows)}")
    summary.append(f"n_train_tiles={len(train_rows)}")
    summary.append(f"n_val_tiles={len(val_rows)}")
    summary.append(f"std_scale={args.std_scale}")
    summary.append(f"bin_stat={args.bin_stat}")
    summary.append(f"split_mode={split_mode}")
    summary.append(f"val_per_bin={args.val_per_bin}")
    summary.append(f"seed={args.seed}")
    summary.append(f"manual_val_rivers={' '.join(args.val_rivers) if args.val_rivers else ''}")
    summary.append("")
    summary.append("Rivers:")
    for r in river_rows:
        summary.append(
            f"  {r['split']:5s} {r['std_bin']:4s} {r['river']:45s} "
            f"n={r['n_tiles']:5d} median_std={r['median_tile_std_safe']:.6g} "
            f"mean_std={r['mean_tile_std_safe']:.6g}"
        )
    (out_dir / "split_summary.txt").write_text("\n".join(summary) + "\n")

    print("\n".join(summary))


if __name__ == "__main__":
    main()
