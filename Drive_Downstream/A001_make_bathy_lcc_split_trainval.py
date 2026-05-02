#!/usr/bin/env python3
"""
Create paired train/val splits for 1 m bathymetry GeoTIFF tiles and LCC masks.

Outputs:
  all_pairs.csv
  train.txt / val.txt
  train_masks.txt / val_masks.txt
  smoke_train.txt / smoke_val.txt
  smoke_train_masks.txt / smoke_val_masks.txt

By default this script DOES NOT create symlinks. Use --make_symlinks only for manual inspection.
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
    for prefix in ("Select_tile_Basin_1m_", "Select_tile_Basin_", "Select_tile_1m_", "Select_tile_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


def mask_key(path: Path) -> str:
    s = path.stem
    for prefix in ("Select_tile_1m_", "Select_tile_Basin_1m_", "Select_tile_Basin_", "Select_tile_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    for suffix in ("_LCC_Mask", "_LCCMASK", "_mask", "_Mask"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


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
        msg = "\n".join([f"{k}: " + "; ".join(str(x) for x in v[:5]) for k, v in list(dup.items())[:10]])
        raise RuntimeError(f"Duplicate keys detected. Please inspect naming.\n{msg}")
    return out


def split_keys_train_val(keys: List[str], seed: int, train_ratio: float) -> Tuple[List[str], List[str]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--train_ratio must be between 0 and 1 for train/val split.")
    rng = random.Random(seed)
    keys = list(keys)
    rng.shuffle(keys)
    n = len(keys)
    if n < 2:
        raise RuntimeError("Need at least 2 paired files to create train/val split.")
    n_train = int(round(n * train_ratio))
    n_train = min(max(n_train, 1), n - 1)
    return keys[:n_train], keys[n_train:]


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
    ap.add_argument("--seed", type=int, default=20260428)
    ap.add_argument("--train_ratio", type=float, default=0.80)
    ap.add_argument("--smoke_train_n", type=int, default=1000)
    ap.add_argument("--smoke_val_n", type=int, default=200)
    ap.add_argument("--exclude_regex", default="", help="Optional regex to exclude pairs by key, e.g. 'Coast|coast|Coastal'.")
    ap.add_argument("--make_symlinks", action="store_true", help="Optional only; not needed for training.")
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
        raise RuntimeError("No paired bath/LCC mask files found. Check naming rules in this script.")

    train_keys, val_keys = split_keys_train_val(common, args.seed, args.train_ratio)
    splits = {"train": train_keys, "val": val_keys}

    with (out_dir / "all_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "split", "bath_path", "mask_path"])
        for split, ks in splits.items():
            for k in ks:
                w.writerow([k, split, baths[k], masks[k]])

    for split, ks in splits.items():
        write_list(out_dir / f"{split}.txt", [baths[k] for k in ks])
        write_list(out_dir / f"{split}_masks.txt", [masks[k] for k in ks])

    smoke_spec = {"train": args.smoke_train_n, "val": args.smoke_val_n}
    for split, n in smoke_spec.items():
        ks = splits[split][: max(0, n)]
        write_list(out_dir / f"smoke_{split}.txt", [baths[k] for k in ks])
        write_list(out_dir / f"smoke_{split}_masks.txt", [masks[k] for k in ks])

    if args.make_symlinks:
        for split, ks in splits.items():
            link_items(out_dir / "symlink" / split / "bath", [baths[k] for k in ks])
            link_items(out_dir / "symlink" / split / "lcc", [masks[k] for k in ks])
        for split, n in smoke_spec.items():
            ks = splits[split][: max(0, n)]
            link_items(out_dir / "symlink" / f"smoke_{split}" / "bath", [baths[k] for k in ks])
            link_items(out_dir / "symlink" / f"smoke_{split}" / "lcc", [masks[k] for k in ks])

    print("[PAIR] bath files:", len(baths))
    print("[PAIR] mask files:", len(masks))
    print("[PAIR] paired files:", len(common))
    print("[PAIR] missing masks:", len(missing_mask))
    print("[PAIR] missing baths:", len(missing_bath))
    print("[SPLIT] train/val:", len(train_keys), len(val_keys))
    print("[SMOKE] train/val:", min(args.smoke_train_n, len(train_keys)), min(args.smoke_val_n, len(val_keys)))
    print("[OUT]", out_dir)
    print("[NOTE] No symlinks created unless --make_symlinks is provided.")
    if missing_mask[:10]:
        print("[WARN] example missing masks:", missing_mask[:10])
    if missing_bath[:10]:
        print("[WARN] example missing bath:", missing_bath[:10])


if __name__ == "__main__":
    main()
