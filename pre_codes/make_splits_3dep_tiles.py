#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import json
from pathlib import Path
from collections import defaultdict
import random

RE_XY = re.compile(r"_x(\d+)_y(\d+)\.tif$", re.IGNORECASE)
RE_RC = re.compile(r"_r(\d+)_c(\d+)\.tif$", re.IGNORECASE)

def parse_xy_rc(p: Path):
    """Return ('xy', x, y) or ('rc', r, c) or (None, None, None)"""
    name = p.name
    m = RE_XY.search(name)
    if m:
        return ("xy", int(m.group(1)), int(m.group(2)))
    m = RE_RC.search(name)
    if m:
        return ("rc", int(m.group(1)), int(m.group(2)))
    return (None, None, None)

def list_tiles(state_dir: Path, patch: int):
    """
    Prefer list file if exists:
      01_lists/{STATE}_tiles_{patch}.txt
    else fallback to scanning 04_tiles_tif_{patch}/*.tif
    """
    lists = state_dir / "01_lists"
    # try to infer state code
    state = state_dir.name
    list_file = lists / f"{state}_tiles_{patch}.txt"
    if list_file.exists():
        tiles = []
        for line in list_file.read_text().splitlines():
            line = line.strip()
            if line:
                tiles.append(Path(line))
        return tiles

    tile_dir = state_dir / f"04_tiles_tif_{patch}"
    if tile_dir.exists():
        return sorted(tile_dir.glob("*.tif"))
    return []

def block_id_for_tile(p: Path, mode: str, a: int, b: int, block_tiles: int, block_pixels: int):
    """
    mode:
      - 'xy': a=x, b=y -> block by pixels
      - 'rc': a=r, b=c -> block by tile index
      - None: fallback: hash bucket
    """
    if mode == "xy":
        bx = a // block_pixels
        by = b // block_pixels
        return f"xy_{bx}_{by}"
    if mode == "rc":
        br = a // block_tiles
        bc = b // block_tiles
        return f"rc_{br}_{bc}"
    # fallback: group by parent folder + hash of filename prefix
    return f"misc_{hash(p.name) % 100000}"

def split_blocks(block_keys, seed, frac_train, frac_val, frac_test):
    """
    Deterministically split blocks into train/val/test by shuffling.
    Ensures non-empty val/test when possible.
    """
    rng = random.Random(seed)
    keys = list(block_keys)
    rng.shuffle(keys)

    n = len(keys)
    if n == 0:
        return [], [], []

    n_train = int(round(frac_train * n))
    n_val = int(round(frac_val * n))
    # adjust to ensure sum <= n
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    n_test = n - n_train - n_val

    # if very small, enforce at least 1 block in val/test when possible
    if n >= 3:
        if n_val == 0:
            n_val = 1
            if n_train > 1:
                n_train -= 1
            else:
                # steal from test if needed
                n_test = max(0, n - n_train - n_val)
        if n_test == 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            else:
                # steal from val if needed
                n_val = max(0, n - n_train - n_test)

    train_keys = keys[:n_train]
    val_keys = keys[n_train:n_train + n_val]
    test_keys = keys[n_train + n_val:]
    return train_keys, val_keys, test_keys

def write_list(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(str(it) + "\n")

def main():
    ap = argparse.ArgumentParser("Make train/val/test splits across multi-state 3DEP tiles (block-based).")
    ap.add_argument("--root", required=True, help="Upstream_Model_ReTrain root (contains 3DEP_Samples/)")
    ap.add_argument("--patch", type=int, default=336, help="Patch size, used to locate 04_tiles_tif_{patch}")
    ap.add_argument("--train", type=float, default=0.90, help="Train fraction")
    ap.add_argument("--val", type=float, default=0.05, help="Val fraction")
    ap.add_argument("--test", type=float, default=0.05, help="Test fraction")
    ap.add_argument("--block_tiles", type=int, default=10, help="Block size in tiles (for rc naming)")
    ap.add_argument("--step", type=int, default=302, help="Step in pixels (patch-overlap), for xy naming")
    ap.add_argument("--seed", type=int, default=1, help="Random seed (reproducible)")
    ap.add_argument("--holdout_state", default="KY", help="State code to hold out entirely as test-state (e.g., KY)")
    ap.add_argument("--out_dir", default="splits", help="Output folder name under root")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    samples = root / "3DEP_Samples"
    out_dir = root / args.out_dir

    if not samples.exists():
        raise SystemExit(f"[ERROR] not found: {samples}")

    # block_pixels for xy mode: block_tiles * step (step is in pixels)
    block_pixels = args.block_tiles * args.step

    # discover states
    states = sorted([p for p in samples.iterdir() if p.is_dir()])
    if not states:
        raise SystemExit(f"[ERROR] no state dirs under: {samples}")

    summary = {
        "root": str(root),
        "samples_dir": str(samples),
        "patch": args.patch,
        "fractions_default": {"train": args.train, "val": args.val, "test": args.test},
        "block_tiles": args.block_tiles,
        "step_pixels": args.step,
        "block_pixels_xy": block_pixels,
        "seed": args.seed,
        "states": {},
        "totals": {},
        "holdout_state": args.holdout_state,
    }

    # per-state split (90/5/5) with blocks
    global_train, global_val, global_test = [], [], []
    by_state_dir = out_dir / "by_state_default_90_5_5"
    by_state_dir.mkdir(parents=True, exist_ok=True)

    for sdir in states:
        state = sdir.name
        tiles = list_tiles(sdir, args.patch)
        tiles = [t for t in tiles if t.exists()]
        if not tiles:
            summary["states"][state] = {"tiles": 0, "blocks": 0}
            continue

        # group tiles by block id
        blocks = defaultdict(list)
        for t in tiles:
            mode, a, b = parse_xy_rc(t)
            bid = block_id_for_tile(t, mode, a, b, args.block_tiles, block_pixels)
            blocks[bid].append(t)

        block_keys = sorted(blocks.keys())
        # state-specific seed (stable but different)
        s_seed = args.seed + (abs(hash(state)) % 100000)

        # If too few blocks, fallback to tile-level split to avoid empty sets
        if len(block_keys) < 6:
            rng = random.Random(s_seed)
            tlist = tiles[:]
            rng.shuffle(tlist)
            n = len(tlist)
            n_train = int(round(args.train * n))
            n_val = int(round(args.val * n))
            if n_train + n_val > n:
                n_val = max(0, n - n_train)
            train_tiles = tlist[:n_train]
            val_tiles = tlist[n_train:n_train + n_val]
            test_tiles = tlist[n_train + n_val:]
        else:
            tr_keys, va_keys, te_keys = split_blocks(block_keys, s_seed, args.train, args.val, args.test)
            train_tiles = [t for k in tr_keys for t in blocks[k]]
            val_tiles   = [t for k in va_keys for t in blocks[k]]
            test_tiles  = [t for k in te_keys for t in blocks[k]]

        # save per-state
        write_list(by_state_dir / f"{state}_train.txt", train_tiles)
        write_list(by_state_dir / f"{state}_val.txt", val_tiles)
        write_list(by_state_dir / f"{state}_test.txt", test_tiles)

        global_train.extend(train_tiles)
        global_val.extend(val_tiles)
        global_test.extend(test_tiles)

        summary["states"][state] = {
            "tiles": len(tiles),
            "blocks": len(block_keys),
            "default_split": {
                "train": len(train_tiles),
                "val": len(val_tiles),
                "test": len(test_tiles),
            }
        }

    # write global default
    write_list(out_dir / "global_default_train.txt", global_train)
    write_list(out_dir / "global_default_val.txt", global_val)
    write_list(out_dir / "global_default_test.txt", global_test)

    summary["totals"]["default_90_5_5"] = {
        "train": len(global_train),
        "val": len(global_val),
        "test": len(global_test),
        "all": len(global_train) + len(global_val) + len(global_test),
    }

    # Holdout-state split: holdout_state all to test; remaining states train/val only (95/5 by default)
    hold = args.holdout_state
    holdout_train, holdout_val, holdout_test = [], [], []
    by_state_hold_dir = out_dir / f"by_state_holdout_{hold}"
    by_state_hold_dir.mkdir(parents=True, exist_ok=True)

    # collect holdout tiles
    hold_dir = samples / hold
    if hold_dir.exists():
        hold_tiles = list_tiles(hold_dir, args.patch)
        hold_tiles = [t for t in hold_tiles if t.exists()]
        holdout_test = hold_tiles[:]  # all holdout as test
    else:
        holdout_test = []

    # remaining states: 95/5 (train/val)
    frac_train2, frac_val2 = 0.95, 0.05
    for sdir in states:
        state = sdir.name
        if state == hold:
            continue
        tiles = list_tiles(sdir, args.patch)
        tiles = [t for t in tiles if t.exists()]
        if not tiles:
            continue

        blocks = defaultdict(list)
        for t in tiles:
            mode, a, b = parse_xy_rc(t)
            bid = block_id_for_tile(t, mode, a, b, args.block_tiles, block_pixels)
            blocks[bid].append(t)
        block_keys = sorted(blocks.keys())
        s_seed = args.seed + 777 + (abs(hash("HOLDOUT_"+state)) % 100000)

        if len(block_keys) < 6:
            rng = random.Random(s_seed)
            tlist = tiles[:]
            rng.shuffle(tlist)
            n = len(tlist)
            n_train = int(round(frac_train2 * n))
            train_tiles = tlist[:n_train]
            val_tiles = tlist[n_train:]
        else:
            tr_keys, va_keys, _ = split_blocks(block_keys, s_seed, frac_train2, frac_val2, 0.0)
            train_tiles = [t for k in tr_keys for t in blocks[k]]
            val_tiles   = [t for k in va_keys for t in blocks[k]]

        write_list(by_state_hold_dir / f"{state}_train.txt", train_tiles)
        write_list(by_state_hold_dir / f"{state}_val.txt", val_tiles)

        holdout_train.extend(train_tiles)
        holdout_val.extend(val_tiles)

    # write holdout global
    write_list(out_dir / f"holdout_{hold}_train.txt", holdout_train)
    write_list(out_dir / f"holdout_{hold}_val.txt", holdout_val)
    write_list(out_dir / f"holdout_{hold}_test.txt", holdout_test)

    summary["totals"][f"holdout_{hold}"] = {
        "train": len(holdout_train),
        "val": len(holdout_val),
        "test": len(holdout_test),
        "train_states": [p.name for p in states if p.name != hold],
    }

    # write summary json
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[DONE] Wrote splits to:", out_dir)
    print("  - global_default_train/val/test.txt")
    print(f"  - holdout_{hold}_train/val/test.txt")
    print("  - summary.json")

if __name__ == "__main__":
    main()

