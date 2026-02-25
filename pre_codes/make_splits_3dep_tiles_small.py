#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Make train/val splits for 3DEP tiles across multiple states (SMALL version).

Differences vs original make_splits_3dep_tiles.py:
  - Supports tiles stored in a custom tiles subdir, e.g. 04_tiles_tif_336_small
  - Outputs ONLY train/val (no test)
  - Optional holdout_state will be written to a separate list (not called "test")
  - Per-state split: each non-holdout state is split independently with the SAME ratios
    (e.g., 95% train / 5% val), then concatenated into global train/val.
  - Uses block-based splitting to reduce spatial leakage, but will (optionally) split
    at most ONE "boundary block" per state to hit the exact per-state tile counts.

Example:
  python pre_codes/make_splits_3dep_tiles_small.py \
    --root /tank/.../Upstream_Model_ReTrain \
    --samples_dir 3DEP_Samples \
    --patch 336 \
    --tiles_subdir 04_tiles_tif_336_small \
    --train 0.95 --val 0.05 \
    --block_tiles 10 --step 302 --seed 1 \
    --out_dir splits/smoke_small_1000
"""

import argparse
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

def list_tiles(state_dir: Path, patch: int, tiles_subdir: str):
    """
    Prefer list file if exists:
      01_lists/{STATE}_tiles_{patch}_small.txt  (when tiles_subdir contains '_small')
      01_lists/{STATE}_tiles_{patch}.txt        (fallback)
    else fallback to scanning {tiles_subdir}/*.tif
    """
    state = state_dir.name
    lists = state_dir / "01_lists"

    candidates = []
    if "_small" in tiles_subdir.lower():
        candidates.append(lists / f"{state}_tiles_{patch}_small.txt")
    candidates.append(lists / f"{state}_tiles_{patch}.txt")

    for list_file in candidates:
        if list_file.exists():
            tiles = []
            for line in list_file.read_text().splitlines():
                line = line.strip()
                if line:
                    tiles.append(Path(line))
            return tiles

    tile_dir = state_dir / tiles_subdir
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
    return f"misc_{hash(p.name) % 100000}"

def split_blocks_train_val(block_keys, seed, frac_train, frac_val):
    """Deterministically split blocks into train/val by shuffling."""
    rng = random.Random(seed)
    keys = list(block_keys)
    rng.shuffle(keys)

    n = len(keys)
    if n == 0:
        return [], []

    n_train = int(round(frac_train * n))
    n_val = n - n_train

    # enforce non-empty val when possible
    if n >= 2 and n_val == 0:
        n_val = 1
        n_train = max(0, n - n_val)

    train_keys = keys[:n_train]
    val_keys = keys[n_train:n_train + n_val]
    return train_keys, val_keys

def split_state_tiles_with_blocks(blocks: dict, block_keys: list, seed: int,
                                 n_train_target: int, n_val_target: int,
                                 allow_partial_block: bool = True):
    """Split tiles into train/val aiming at exact target counts.

    Strategy:
      - Shuffle blocks.
      - Greedily assign blocks to VAL until reaching n_val_target.
      - If overshoot and allow_partial_block: split the last (boundary) block so that
        val count matches exactly. This limits leakage to within one block.
      - Remaining blocks go to TRAIN.
    """
    rng = random.Random(seed)
    keys = list(block_keys)
    rng.shuffle(keys)

    val_tiles = []
    train_tiles = []
    boundary_block = None
    boundary_mode = "none"

    # 1) Build VAL from shuffled blocks
    for k in keys:
        if len(val_tiles) >= n_val_target:
            break
        block_tiles = blocks[k]
        if len(val_tiles) + len(block_tiles) <= n_val_target:
            val_tiles.extend(block_tiles)
        else:
            # overshoot
            boundary_block = k
            if allow_partial_block:
                need = max(0, n_val_target - len(val_tiles))
                # deterministic shuffle inside boundary block
                t = block_tiles[:]
                rng2 = random.Random(seed + 99991)
                rng2.shuffle(t)
                val_tiles.extend(t[:need])
                train_tiles.extend(t[need:])
                boundary_mode = "partial_block"
            else:
                val_tiles.extend(block_tiles)
                boundary_mode = "overshoot_whole_block"
            break

    # 2) Remaining blocks go to TRAIN
    used_in_val = set(val_tiles)
    used_in_train = set(train_tiles)
    for k in keys:
        for t in blocks[k]:
            if t in used_in_val or t in used_in_train:
                continue
            train_tiles.append(t)

    # 3) If still missing VAL (rare), steal from TRAIN deterministically
    if len(val_tiles) < n_val_target and len(train_tiles) > 0:
        need = n_val_target - len(val_tiles)
        rng3 = random.Random(seed + 123457)
        rng3.shuffle(train_tiles)
        move = train_tiles[:need]
        val_tiles.extend(move)
        train_tiles = train_tiles[need:]
        boundary_mode = "tile_steal"

    # 4) Finally, enforce exact train count if requested and possible
    # (keep val fixed, adjust train by trimming if train too large)
    if n_train_target >= 0 and len(train_tiles) > n_train_target:
        train_tiles = train_tiles[:n_train_target]
        boundary_mode = boundary_mode + "+trim_train"

    return train_tiles, val_tiles, boundary_block, boundary_mode

def write_list(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(str(it) + "\n")

def main():
    ap = argparse.ArgumentParser("Make train/val splits across multi-state 3DEP tiles (SMALL).")

    ap.add_argument("--root", required=True, help="Upstream_Model_ReTrain root")
    ap.add_argument("--samples_dir", default="3DEP_Samples",
                    help="Samples folder under root (e.g., 3DEP_Samples or 3DEP_Samples_small)")
    ap.add_argument("--patch", type=int, default=336, help="Patch size (e.g., 336)")
    ap.add_argument("--tiles_subdir", default="",
                    help="Tiles folder name under each state. If empty, use 04_tiles_tif_{patch}. "
                         "For small sampling, set to 04_tiles_tif_{patch}_small")
    ap.add_argument("--train", type=float, default=0.95, help="Train fraction")
    ap.add_argument("--val", type=float, default=0.05, help="Val fraction")
    ap.add_argument("--block_tiles", type=int, default=10, help="Block size in tiles (for rc naming)")
    ap.add_argument("--step", type=int, default=302, help="Step in pixels (patch overlap), for xy naming")
    ap.add_argument("--seed", type=int, default=1, help="Random seed")
    ap.add_argument("--holdout_state", default="",
                    help="Optional state code to hold out completely (written to holdout_<STATE>.txt). "
                         "Leave empty to disable.")
    ap.add_argument("--allow_partial_block", action="store_true",
                    help="When using block splitting, allow splitting ONE boundary block per state to hit exact "
                         "tile counts for train/val. Recommended for keeping per-state ratios exact.")
    ap.add_argument("--out_dir", default="splits/smoke_small", help="Output folder under root")
    args = ap.parse_args()

    if args.train <= 0 or args.val <= 0:
        raise SystemExit("[ERROR] --train and --val must be > 0")
    if abs((args.train + args.val) - 1.0) > 1e-6:
        raise SystemExit("[ERROR] require --train + --val = 1.0 (no test).")

    root = Path(args.root).resolve()
    samples = root / args.samples_dir
    out_dir = root / args.out_dir
    if not samples.exists():
        raise SystemExit(f"[ERROR] not found: {samples}")

    tiles_subdir = args.tiles_subdir.strip() or f"04_tiles_tif_{args.patch}"
    block_pixels = args.block_tiles * args.step

    states = sorted([p for p in samples.iterdir() if p.is_dir()])
    if not states:
        raise SystemExit(f"[ERROR] no state dirs under: {samples}")

    hold = args.holdout_state.strip()
    global_train, global_val = [], []
    holdout_tiles = []

    by_state_dir = out_dir / "by_state"
    by_state_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "root": str(root),
        "samples_dir": str(samples),
        "tiles_subdir": tiles_subdir,
        "patch": args.patch,
        "fractions": {"train": args.train, "val": args.val},
        "block_tiles": args.block_tiles,
        "step_pixels": args.step,
        "block_pixels_xy": block_pixels,
        "seed": args.seed,
        "holdout_state": hold,
        "states": {},
        "totals": {},
    }

    for sdir in states:
        state = sdir.name
        tiles = list_tiles(sdir, args.patch, tiles_subdir)
        tiles = [t.resolve() for t in tiles if t.exists()]

        if not tiles:
            summary["states"][state] = {"tiles": 0, "blocks": 0}
            continue

        if hold and state.upper() == hold.upper():
            holdout_tiles.extend(tiles)
            # also write by-state holdout list
            write_list(by_state_dir / f"{state}_holdout.txt", tiles)
            summary["states"][state] = {"tiles": len(tiles), "blocks": None, "split": {"holdout": len(tiles)}}
            continue

        blocks = defaultdict(list)
        for t in tiles:
            mode, a, b = parse_xy_rc(t)
            bid = block_id_for_tile(t, mode, a, b, args.block_tiles, block_pixels)
            blocks[bid].append(t)
        block_keys = sorted(blocks.keys())

        s_seed = args.seed + (abs(hash(state)) % 100000)

        # ---- Per-state target counts (exact ratios within each state) ----
        n = len(tiles)
        n_val = int(round(args.val * n))
        if n >= 2 and n_val == 0:
            n_val = 1
        n_train = n - n_val

        boundary_block = None
        boundary_mode = ""

        if len(block_keys) < 4:
            # too few blocks -> fall back to per-tile shuffle (exact ratio)
            rng = random.Random(s_seed)
            tlist = tiles[:]
            rng.shuffle(tlist)
            train_tiles = tlist[:n_train]
            val_tiles = tlist[n_train:]
            boundary_mode = "tile_shuffle"
        else:
            train_tiles, val_tiles, boundary_block, boundary_mode = split_state_tiles_with_blocks(
                blocks, block_keys, s_seed, n_train, n_val, allow_partial_block=args.allow_partial_block
            )

        write_list(by_state_dir / f"{state}_train.txt", train_tiles)
        write_list(by_state_dir / f"{state}_val.txt", val_tiles)

        global_train.extend(train_tiles)
        global_val.extend(val_tiles)

        summary["states"][state] = {
            "tiles": len(tiles),
            "blocks": len(block_keys),
            "split": {"train": len(train_tiles), "val": len(val_tiles)},
            "split_mode": boundary_mode,
            "boundary_block": boundary_block,
            "targets": {"train": n_train, "val": n_val}
        }

    out_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: To avoid duplicated "global" vs top-level lists, we ONLY write the
    # final aggregated lists into out_dir/global/.
    global_dir = out_dir / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    write_list(global_dir / "train.txt", global_train)
    write_list(global_dir / "val.txt", global_val)

    if hold and holdout_tiles:
        write_list(global_dir / f"holdout_{hold}.txt", holdout_tiles)

    summary["totals"] = {
        "train": len(global_train),
        "val": len(global_val),
        "all": len(global_train) + len(global_val),
        "holdout": len(holdout_tiles) if hold else 0
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("[DONE] Wrote splits to:", out_dir)
    print("  - global/train.txt / global/val.txt")
    if hold:
        print(f"  - global/holdout_{hold}.txt (optional)")
    print("  - by_state/*.txt")
    print("  - summary.json")

if __name__ == "__main__":
    main()
