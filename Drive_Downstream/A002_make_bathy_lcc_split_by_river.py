#!/usr/bin/env python3
"""
Create paired train/val splits for 1m bathymetry tiles and mask tiles,
with river-level holdout validation.

Filename examples:
  bath:
    Select_tile_Basin_1m_BadgerFinNull_ID10.tif
  mask:
    Select_tile_1m_BadgerFinNull_ID10_LCC_Mask.tif

Key:
  BadgerFinNull_ID10

River:
  BadgerFinNull
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

TIFF_EXTS = {".tif", ".tiff", ".TIF", ".TIFF"}


def bath_key(path: Path) -> str:
    s = path.stem
    for prefix in (
        "Select_tile_Basin_1m_",
        "Select_tile_Basin_",
        "Select_tile_1m_",
        "Select_tile_",
    ):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


def mask_key(path: Path) -> str:
    s = path.stem
    for prefix in (
        "Select_tile_1m_",
        "Select_tile_Basin_1m_",
        "Select_tile_Basin_",
        "Select_tile_",
    ):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break

    for suffix in ("_LCC_Mask", "_LCCMASK", "_mask", "_Mask"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def river_from_key(key: str) -> str:
    """
    Convert:
      BadgerFinNull_ID10 -> BadgerFinNull
      OR_MKRC_Topobathy_2021_ID123 -> OR_MKRC_Topobathy_2021
    """
    m = re.match(r"^(.*)_ID\d+$", key)
    if not m:
        raise ValueError(f"Cannot parse river name from key: {key}")
    return m.group(1)


def list_tiffs(root: Path) -> List[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix in TIFF_EXTS])


def index_by_key(paths: Iterable[Path], key_func) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    dup: Dict[str, List[Path]] = {}

    for p in paths:
        k = key_func(p)
        if k in out:
            dup.setdefault(k, [out[k]]).append(p)
        else:
            out[k] = p

    if dup:
        msg = "\n".join(
            [f"{k}: " + "; ".join(str(x) for x in v[:5])
             for k, v in list(dup.items())[:10]]
        )
        raise RuntimeError(f"Duplicate keys detected. Please inspect naming.\n{msg}")

    return out


def write_list(path: Path, items: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in items:
            f.write(str(p) + "\n")


def link_items(dst_dir: Path, items: Iterable[Path]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in items:
        dst = dst_dir / p.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(str(p), str(dst))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--bath_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--val_river", default="", help="River name used as validation set.")
    ap.add_argument("--seed", type=int, default=20260428)

    ap.add_argument("--smoke_train_n", type=int, default=1000)
    ap.add_argument("--smoke_val_n", type=int, default=200)

    ap.add_argument("--exclude_regex", default="", help="Optional regex to exclude pairs by key.")
    ap.add_argument("--make_symlinks", action="store_true")

    args = ap.parse_args()

    bath_dir = Path(args.bath_dir).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baths = index_by_key(list_tiffs(bath_dir), bath_key)
    masks = index_by_key(list_tiffs(mask_dir), mask_key)

    common = sorted(set(baths) & set(masks))
    missing_mask = sorted(set(baths) - set(masks))
    missing_bath = sorted(set(masks) - set(baths))

    if args.exclude_regex:
        rx = re.compile(args.exclude_regex)
        common = [k for k in common if not rx.search(k)]

    if not common:
        raise RuntimeError("No paired bath/mask files found. Check naming rules.")

    # Group by river
    river_to_keys: Dict[str, List[str]] = {}
    for k in common:
        river = river_from_key(k)
        river_to_keys.setdefault(river, []).append(k)

    # Write river count table
    count_csv = out_dir / "tile_count_by_river.csv"
    with count_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["river", "n_tiles"])
        for river in sorted(river_to_keys):
            w.writerow([river, len(river_to_keys[river])])

    # If no val river is given, only report count and stop before split.
    if not args.val_river:
        print("[PAIR] bath files:", len(baths))
        print("[PAIR] mask files:", len(masks))
        print("[PAIR] paired files:", len(common))
        print("[PAIR] missing masks:", len(missing_mask))
        print("[PAIR] missing baths:", len(missing_bath))
        print("[COUNT]", count_csv)
        print("[NOTE] No --val_river provided, so only tile counts were generated.")
        print("[RIVER COUNTS]")
        for river in sorted(river_to_keys):
            print(f"  {river}: {len(river_to_keys[river])}")
        return

    val_river = args.val_river

    if val_river not in river_to_keys:
        available = "\n".join([f"  {r}: {len(river_to_keys[r])}" for r in sorted(river_to_keys)])
        raise RuntimeError(
            f"--val_river not found: {val_river}\n"
            f"Available rivers:\n{available}"
        )

    val_keys = sorted(river_to_keys[val_river])
    train_keys = sorted([k for k in common if river_from_key(k) != val_river])

    # Shuffle train only, val remains deterministic sorted
    rng = random.Random(args.seed)
    rng.shuffle(train_keys)

    splits = {
        "train": train_keys,
        "val": val_keys,
    }

    # Write all pairs
    with (out_dir / "all_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "river", "split", "bath_path", "mask_path"])
        for split, ks in splits.items():
            for k in ks:
                w.writerow([k, river_from_key(k), split, baths[k], masks[k]])

    # Write list files
    for split, ks in splits.items():
        write_list(out_dir / f"{split}.txt", [baths[k] for k in ks])
        write_list(out_dir / f"{split}_masks.txt", [masks[k] for k in ks])

    # Smoke files
    smoke_spec = {"train": args.smoke_train_n, "val": args.smoke_val_n}
    for split, n in smoke_spec.items():
        ks = splits[split][: max(0, n)]
        write_list(out_dir / f"smoke_{split}.txt", [baths[k] for k in ks])
        write_list(out_dir / f"smoke_{split}_masks.txt", [masks[k] for k in ks])

    # Optional symlinks
    if args.make_symlinks:
        for split, ks in splits.items():
            link_items(out_dir / "symlink" / split / "bath", [baths[k] for k in ks])
            link_items(out_dir / "symlink" / split / "mask", [masks[k] for k in ks])

        for split, n in smoke_spec.items():
            ks = splits[split][: max(0, n)]
            link_items(out_dir / "symlink" / f"smoke_{split}" / "bath", [baths[k] for k in ks])
            link_items(out_dir / "symlink" / f"smoke_{split}" / "mask", [masks[k] for k in ks])

    print("[PAIR] bath files:", len(baths))
    print("[PAIR] mask files:", len(masks))
    print("[PAIR] paired files:", len(common))
    print("[PAIR] missing masks:", len(missing_mask))
    print("[PAIR] missing baths:", len(missing_bath))
    print("[COUNT]", count_csv)
    print("[VAL_RIVER]", val_river)
    print("[SPLIT] train/val:", len(train_keys), len(val_keys))
    print("[SMOKE] train/val:", min(args.smoke_train_n, len(train_keys)), min(args.smoke_val_n, len(val_keys)))
    print("[OUT]", out_dir)

    if missing_mask[:10]:
        print("[WARN] example missing masks:", missing_mask[:10])
    if missing_bath[:10]:
        print("[WARN] example missing bath:", missing_bath[:10])


if __name__ == "__main__":
    main()