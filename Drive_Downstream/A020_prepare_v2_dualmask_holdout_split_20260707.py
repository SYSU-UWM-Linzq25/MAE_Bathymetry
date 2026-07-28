#!/usr/bin/env python3
# NUMBER-ALIGNED NAME: A020_prepare_v2_dualmask_holdout_split_20260707.py
# ORIGINAL BACKUP NAME: A020_prepare_v2_dualmask_holdout_split_20260707.py
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
"""Prepare MAE v2 holdout train/val lists for Train_tile + Hidden_Mask + Loss_Mask_Pixel.

Holdout-only split:
  train = all rivers except holdout river(s)
  val   = all tiles from holdout river(s)
  test  = empty

Supported D001 output filename patterns:
  Train_tile/Select_tile_Basin_1m_<river>_ID123.tif
  Hidden_Mask/Select_tile_1m_<river>_ID123_HiddenMask.tif
  Loss_Mask_Pixel/Select_tile_1m_<river>_ID123_LossMask*.tif

Matching key:
  <resolution>m_<river>_ID<id>
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


TILE_RE = re.compile(
    r"^Select_tile_(?:Basin_)?(?P<res>\d+)m_(?P<river>.+)_ID(?P<id>\d+)(?:_(?P<suffix>[^.]+))?\.tif$",
    re.IGNORECASE,
)

DEFAULT_RIVERS = [
    "BadgerFinNull",
    "Estabrook_Combined",
    "KewaFix2Null",
    "Kletzch_Combined_UpMax3Null",
    "CA_KlamathRiver_TopoBathy_2018_D18",
    "CO_UpperColorado_Topobathy_1_2020",
    "MD_PotomacRiver_Bathy_2019",
    "NE_Niobrara_Topobathy_2018",
    "OR_MKRC_Topobathy_2021",
    "OR_SantiamRiverTB_Topobathy_1_D23",
    "WA_ChehalisRiverTB_Topobathy_1_D23",
    "WA_Nisqually_Bathymetric_2020",
]

PRESET_HOLDOUTS = {
    "CO": ("CO_UpperColorado_Topobathy_1_2020", ["CO_UpperColorado_Topobathy_1_2020"]),
    "CA": ("CA_KlamathRiver_TopoBathy_2018_D18", ["CA_KlamathRiver_TopoBathy_2018_D18"]),
    "Santiam": ("OR_SantiamRiverTB_Topobathy_1_D23", ["OR_SantiamRiverTB_Topobathy_1_D23"]),
    "NE": ("NE_Niobrara_Topobathy_2018", ["NE_Niobrara_Topobathy_2018"]),
    "OR_MKRC": ("OR_MKRC_Topobathy_2021", ["OR_MKRC_Topobathy_2021"]),
    "Nisqually": ("WA_Nisqually_Bathymetric_2020", ["WA_Nisqually_Bathymetric_2020"]),
    "MD": ("MD_PotomacRiver_Bathy_2019", ["MD_PotomacRiver_Bathy_2019"]),
    "Chehalis": ("WA_ChehalisRiverTB_Topobathy_1_D23", ["WA_ChehalisRiverTB_Topobathy_1_D23"]),
    "MilwaukeeGroup": (
        "MilwaukeeRiverGroup",
        ["BadgerFinNull", "Estabrook_Combined", "KewaFix2Null", "Kletzch_Combined_UpMax3Null"],
    ),
}


def _match(path: Path) -> re.Match:
    m = TILE_RE.match(path.name)
    if not m:
        raise ValueError(
            "Cannot parse tile/mask filename. Expected patterns like:\n"
            "  Select_tile_Basin_1m_<river>_ID123.tif\n"
            "  Select_tile_1m_<river>_ID123_HiddenMask.tif\n"
            "  Select_tile_1m_<river>_ID123_LossMask*.tif\n"
            f"Got: {path.name}"
        )
    return m


def parse_tile(path: Path) -> Tuple[str, int]:
    m = _match(path)
    return m.group("river"), int(m.group("id"))


def key(path: Path) -> str:
    m = _match(path)
    return f"{m.group('res')}m_{m.group('river')}_ID{m.group('id')}"


def collect(tile_root: Path, river_allowlist: List[str] | None = None) -> List[dict]:
    train_dir = tile_root / "Train_tile"
    hidden_dir = tile_root / "Hidden_Mask"
    loss_dir = tile_root / "Loss_Mask_Pixel"

    for d in (train_dir, hidden_dir, loss_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Missing required directory: {d}")

    allow = set(river_allowlist or [])

    train_files = sorted(train_dir.glob("*.tif"))
    hidden_files = sorted(hidden_dir.glob("*.tif"))
    loss_files = sorted(loss_dir.glob("*.tif"))

    print(f"[INFO] Train tiles : {len(train_files)}")
    print(f"[INFO] Hidden masks: {len(hidden_files)}")
    print(f"[INFO] Loss masks  : {len(loss_files)}")

    hidden_map = {}
    duplicated_hidden = defaultdict(list)
    for p in hidden_files:
        k = key(p)
        if k in hidden_map:
            duplicated_hidden[k].append(str(p))
        else:
            hidden_map[k] = p

    loss_map = {}
    duplicated_loss = defaultdict(list)
    for p in loss_files:
        k = key(p)
        if k in loss_map:
            duplicated_loss[k].append(str(p))
        else:
            loss_map[k] = p

    if duplicated_hidden:
        print(f"[WARN] duplicated hidden-mask keys: {len(duplicated_hidden)}. First 5:", file=sys.stderr)
        for k, vals in list(duplicated_hidden.items())[:5]:
            print(f"  {k}: {vals}", file=sys.stderr)

    if duplicated_loss:
        print(f"[WARN] duplicated loss-mask keys: {len(duplicated_loss)}. First 5:", file=sys.stderr)
        for k, vals in list(duplicated_loss.items())[:5]:
            print(f"  {k}: {vals}", file=sys.stderr)

    rows: List[dict] = []
    missing = []
    skipped_by_allowlist = defaultdict(int)

    for tp in train_files:
        river, tid = parse_tile(tp)
        if allow and river not in allow:
            skipped_by_allowlist[river] += 1
            continue

        k = key(tp)
        hp = hidden_map.get(k)
        lp = loss_map.get(k)
        if hp is None or lp is None:
            missing.append((tp.name, hp is None, lp is None))
            continue

        rows.append({
            "key": k,
            "river": river,
            "tile_id": tid,
            "tile": str(tp),
            "hidden": str(hp),
            "loss": str(lp),
        })

    if skipped_by_allowlist:
        print("[INFO] skipped files outside river_allowlist:", file=sys.stderr)
        for river, n in sorted(skipped_by_allowlist.items()):
            print(f"  {river}: {n}", file=sys.stderr)

    if missing:
        print(f"[WARN] missing hidden/loss masks for {len(missing)} train tiles. First 20:", file=sys.stderr)
        for item in missing[:20]:
            print("  ", item, file=sys.stderr)

    if not rows:
        raise RuntimeError("No matched Train_tile/Hidden_Mask/Loss_Mask_Pixel triples found.")

    return rows


def write_list(path: Path, values: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""))


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "key", "river", "tile_id", "tile", "hidden", "loss"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--holdout_preset",
        default="",
        choices=[""] + sorted(PRESET_HOLDOUTS.keys()),
        help="Optional preset. Example: CO, CA, Santiam, MilwaukeeGroup.",
    )
    ap.add_argument("--holdout_name", default="")
    ap.add_argument("--holdout_rivers", nargs="*", default=[])
    ap.add_argument("--river_allowlist", nargs="*", default=DEFAULT_RIVERS)
    args = ap.parse_args()

    holdout_name = args.holdout_name
    holdout_rivers = list(args.holdout_rivers)

    if args.holdout_preset:
        preset_name, preset_rivers = PRESET_HOLDOUTS[args.holdout_preset]
        if not holdout_name:
            holdout_name = preset_name
        if not holdout_rivers:
            holdout_rivers = preset_rivers

    if not holdout_rivers:
        raise ValueError("Holdout split requires --holdout_rivers or --holdout_preset.")
    if not holdout_name:
        holdout_name = "_".join(holdout_rivers)

    rows = collect(Path(args.tile_root), river_allowlist=args.river_allowlist)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    holdout = set(holdout_rivers)
    all_rivers = sorted({row["river"] for row in rows})
    unknown_holdout = sorted(holdout - set(all_rivers))
    if unknown_holdout:
        raise ValueError(
            "Holdout river(s) not found in matched tile triples: "
            + ", ".join(unknown_holdout)
            + "\nAvailable rivers: "
            + ", ".join(all_rivers)
        )

    train = [row.copy() for row in rows if row["river"] not in holdout]
    val = [row.copy() for row in rows if row["river"] in holdout]
    test: List[dict] = []

    if not train:
        raise RuntimeError("Training split is empty. Check holdout_rivers.")
    if not val:
        raise RuntimeError("Validation split is empty. Check holdout_rivers.")

    for split, subset in [("train", train), ("val", val), ("test", test)]:
        for row in subset:
            row["split"] = split
        write_list(out / f"{split}_tiles.txt", [row["tile"] for row in subset])
        write_list(out / f"{split}_hidden.txt", [row["hidden"] for row in subset])
        write_list(out / f"{split}_loss.txt", [row["loss"] for row in subset])

    write_csv(out / "split_manifest.csv", train + val + test)

    by_river_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in train + val + test:
        by_river_count[row["river"]][row["split"]] += 1

    summary = [
        f"split_name=holdout_{holdout_name}",
        "mode=holdout_only",
        f"holdout_name={holdout_name}",
        f"holdout_rivers={' '.join(holdout_rivers)}",
        f"n_total={len(rows)}",
        f"n_train={len(train)}",
        f"n_val={len(val)}",
        f"n_test={len(test)}",
        "",
        "by_river:",
    ]

    for river in all_rivers:
        d = by_river_count[river]
        summary.append(f"  {river}: train={d['train']} val={d['val']} test={d['test']}")

    (out / "split_summary.txt").write_text("\n".join(summary) + "\n")

    print("\n".join(summary))
    print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
