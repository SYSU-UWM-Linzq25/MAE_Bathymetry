#!/usr/bin/env python3
"""Create national EPSG:5070 spatial-block train/val splits shared by all four scales."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
from collections import Counter
from pathlib import Path

from conus_common import SCALES, parse_states, read_csv, write_csv


LOG = logging.getLogger("spatial-split")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sampling-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--states", nargs="*", default=None)
    p.add_argument("--block-size-m", type=float, default=33600.0)
    p.add_argument("--guard-m", type=float, default=1680.0)
    p.add_argument("--train-fraction", type=float, default=0.80)
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--log-level", default="INFO")
    return p


def label_for_block(bx: int, by: int, seed: int, train_fraction: float) -> str:
    raw = hashlib.sha256(f"{seed}|{bx}|{by}".encode("utf-8")).digest()
    value = int.from_bytes(raw[:8], "big") / float(2**64)
    if value < train_fraction:
        return "train"
    return "val"


def conflicting_neighbor_labels(x: float, y: float, bx: int, by: int, args, own: str) -> bool:
    size = args.block_size_m
    guard = args.guard_m
    x_local = x - bx * size
    y_local = y - by * size
    dx: list[int] = [0]
    dy: list[int] = [0]
    if x_local < guard:
        dx.append(-1)
    if size - x_local < guard:
        dx.append(1)
    if y_local < guard:
        dy.append(-1)
    if size - y_local < guard:
        dy.append(1)
    for ox in dx:
        for oy in dy:
            if ox == 0 and oy == 0:
                continue
            if label_for_block(bx + ox, by + oy, args.seed, args.train_fraction) != own:
                return True
    return False


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0,1); validation receives the remainder")
    if args.guard_m < SCALES["S10"] / 2:
        raise ValueError("guard must be at least half the S10 width (1680 m)")
    states = parse_states(args.states)
    sampling_root = Path(args.sampling_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    centers: list[dict] = []
    nested_by_center: dict[str, list[dict]] = {}
    for state in states:
        center_path = sampling_root / "manifests" / f"centers_{state}.csv"
        nested_path = sampling_root / "manifests" / f"nested_{state}.csv"
        if not center_path.exists() or not nested_path.exists():
            raise FileNotFoundError(f"Missing sampling manifests for {state}")
        centers.extend(read_csv(center_path))
        for row in read_csv(nested_path):
            nested_by_center.setdefault(row["center_id"], []).append(row)

    split_rows: list[dict] = []
    kept: dict[str, str] = {}
    for row in centers:
        x = float(row["center_x_5070"])
        y = float(row["center_y_5070"])
        bx = math.floor(x / args.block_size_m)
        by = math.floor(y / args.block_size_m)
        split = label_for_block(bx, by, args.seed, args.train_fraction)
        excluded = conflicting_neighbor_labels(x, y, bx, by, args, split)
        reason = "cross_split_guard" if excluded else ""
        split_rows.append(
            {
                "center_id": row["center_id"],
                "state": row["state"],
                "center_x_5070": row["center_x_5070"],
                "center_y_5070": row["center_y_5070"],
                "block_x": bx,
                "block_y": by,
                "split": split if not excluded else "excluded",
                "exclude_reason": reason,
            }
        )
        if not excluded:
            kept[row["center_id"]] = split

    write_csv(
        out_dir / "center_splits.csv",
        split_rows,
        ("center_id", "state", "center_x_5070", "center_y_5070", "block_x", "block_y", "split", "exclude_reason"),
    )

    list_counts = Counter()
    for scale in SCALES:
        for split in ("train", "val"):
            path = out_dir / "lists" / scale / f"{split}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            values: list[str] = []
            for center_id, center_split in kept.items():
                if center_split != split:
                    continue
                rows = [r for r in nested_by_center.get(center_id, []) if r["scale"] == scale]
                if len(rows) != 1:
                    raise RuntimeError(f"{center_id}: expected one {scale} row, found {len(rows)}")
                sample_path = rows[0].get("sample_path", "")
                if sample_path:
                    values.append(sample_path)
            values.sort()
            path.write_text("".join(v + "\n" for v in values), encoding="utf-8")
            list_counts[(scale, split)] = len(values)

    counts = Counter(row["split"] for row in split_rows)
    summary = [
        {"category": key, "centers": counts[key]}
        for key in ("train", "val", "excluded")
    ]
    write_csv(out_dir / "split_summary.csv", summary, ("category", "centers"))
    LOG.info("split counts: %s", dict(counts))
    if not all(counts[name] > 0 for name in ("train", "val")):
        LOG.error("One or more splits are empty")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
