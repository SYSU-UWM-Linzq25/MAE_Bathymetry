#!/usr/bin/env python3
"""Randomly sample valid S10 centers and create concentric S1/S3/S5/S10 windows for one or more states."""

from __future__ import annotations

import argparse
import logging
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from osgeo import gdal, ogr
from rasterio.windows import Window

from conus_common import (
    NODATA,
    SCALES,
    TARGET_CRS,
    bbox_geometry,
    load_state_geometries,
    parse_states,
    read_csv,
    stable_id,
    stable_int,
    write_csv,
)


LOG = logging.getLogger("nested-sampling")
gdal.UseExceptions()

CENTER_FIELDS = (
    "center_id", "state", "center_rank", "center_x_5070", "center_y_5070",
    "s10_valid_ratio", "distance_stage_m", "download_key", "prepared_path",
    "s10_col_off", "s10_row_off", "seed",
)
NESTED_FIELDS = (
    "center_id", "state", "scale", "size_pixels", "resolution_m", "center_x_5070",
    "center_y_5070", "download_key", "prepared_path", "col_off", "row_off",
    "valid_ratio_from_s10", "sample_relpath", "sample_path",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-boundaries", required=True)
    p.add_argument("--anchor-plan", required=True)
    p.add_argument("--source-index", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--states", nargs="*", default=None)
    p.add_argument("--target-per-state", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--min-valid-ratio", type=float, default=1.0)
    p.add_argument("--distance-stages-m", default="3024,1512,756,0")
    p.add_argument("--rounds-per-stage", type=int, default=20)
    p.add_argument("--max-stall-rounds", type=int, default=5)
    p.add_argument("--candidate-attempts-per-source", type=int, default=5)
    p.add_argument("--output-mode", choices=("VRT", "GTiff", "manifest"), default="VRT")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--require-target", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


class DistanceIndex:
    def __init__(self, distance: float, existing: list[tuple[float, float]]):
        self.distance = float(distance)
        self.cell = max(self.distance, 1.0)
        self.cells: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
        self.exact: set[tuple[int, int]] = set()
        for x, y in existing:
            self.add(x, y)

    def cell_key(self, x: float, y: float) -> tuple[int, int]:
        return math.floor(x / self.cell), math.floor(y / self.cell)

    def too_close(self, x: float, y: float) -> bool:
        exact_key = (round(x), round(y))
        if exact_key in self.exact:
            return True
        if self.distance <= 0:
            return False
        gx, gy = self.cell_key(x, y)
        radius2 = self.distance * self.distance
        for ix in range(gx - 1, gx + 2):
            for iy in range(gy - 1, gy + 2):
                for ox, oy in self.cells.get((ix, iy), ()):
                    if (x - ox) ** 2 + (y - oy) ** 2 < radius2:
                        return True
        return False

    def add(self, x: float, y: float) -> None:
        self.exact.add((round(x), round(y)))
        self.cells[self.cell_key(x, y)].append((x, y))


def random_point_in_geometry(geom, rng: random.Random, attempts: int = 200):
    if geom is None or geom.IsEmpty():
        return None
    minx, maxx, miny, maxy = geom.GetEnvelope()
    if minx >= maxx or miny >= maxy:
        return None
    for _ in range(attempts):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint_2D(x, y)
        if geom.Intersects(point):
            return float(x), float(y)
    representative = geom.PointOnSurface()
    if representative is not None and not representative.IsEmpty():
        return float(representative.GetX()), float(representative.GetY())
    return None


def open_source(row: dict):
    path = row["prepared_path"]
    src = rasterio.open(path)
    if src.crs is None or src.crs.to_string().upper() not in ("EPSG:5070", TARGET_CRS):
        src.close()
        raise RuntimeError(f"Prepared raster is not EPSG:5070: {path} ({src.crs})")
    if abs(src.transform.a - 1.0) > 1e-6 or abs(src.transform.e + 1.0) > 1e-6:
        src.close()
        raise RuntimeError(f"Prepared raster is not north-up 1 m: {path} ({src.transform})")
    return src


def candidate_from_source(src, state_geom, rng: random.Random):
    half = SCALES["S10"] // 2
    safety = 2
    inner = bbox_geometry(
        (
            src.bounds.left + half + safety,
            src.bounds.bottom + half + safety,
            src.bounds.right - half - safety,
            src.bounds.top - half - safety,
        )
    )
    feasible = state_geom.Intersection(inner)
    point_xy = random_point_in_geometry(feasible, rng)
    if point_xy is None:
        return None
    point_x, point_y = point_xy
    row, col = src.index(point_x, point_y)
    col0 = int(col - half)
    row0 = int(row - half)
    if col0 < 0 or row0 < 0 or col0 + SCALES["S10"] > src.width or row0 + SCALES["S10"] > src.height:
        return None
    center_x, center_y = src.transform * (col0 + half, row0 + half)
    center_point = ogr.Geometry(ogr.wkbPoint)
    center_point.AddPoint_2D(float(center_x), float(center_y))
    if not state_geom.Intersects(center_point):
        return None
    return col0, row0, float(center_x), float(center_y)


def valid_ratio(src, col0: int, row0: int) -> float:
    size = SCALES["S10"]
    mask = src.read_masks(1, window=Window(col0, row0, size, size), boundless=False)
    if mask.shape != (size, size):
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def create_sample(src_path: str, dst_path: Path, col: int, row: int, size: int, mode: str) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return
    options = gdal.TranslateOptions(
        format=mode,
        srcWin=[col, row, size, size],
        noData=NODATA,
        creationOptions=(
            ["TILED=YES", "COMPRESS=ZSTD", "PREDICTOR=3", "BIGTIFF=IF_SAFER"]
            if mode == "GTiff" else []
        ),
    )
    ds = gdal.Translate(str(dst_path), src_path, options=options)
    if ds is None:
        raise RuntimeError(f"gdal.Translate failed: {dst_path}")
    ds.FlushCache()
    ds = None


def rows_for_state(plan_rows: list[dict], index_rows: list[dict], state: str) -> list[dict]:
    keys = {row["download_key"] for row in plan_rows if row["state"] == state}
    seen: set[str] = set()
    rows: list[dict] = []
    for row in index_rows:
        if row["download_key"] in keys and row["prepared_path"] not in seen:
            rows.append(row)
            seen.add(row["prepared_path"])
    return rows


def sample_state(args, state: str, state_geom, sources: list[dict], out_root: Path) -> tuple[int, list[dict], list[dict]]:
    state_center_path = out_root / "manifests" / f"centers_{state}.csv"
    state_nested_path = out_root / "manifests" / f"nested_{state}.csv"
    if state_center_path.exists() and state_nested_path.exists() and not args.overwrite:
        existing = read_csv(state_center_path)
        if len(existing) >= args.target_per_state:
            LOG.info("%s: existing complete manifest (%d), skip", state, len(existing))
            return len(existing), existing, read_csv(state_nested_path)

    if not sources:
        LOG.error("%s: no prepared sources from anchor plan", state)
        return 0, [], []

    distance_stages = [float(v) for v in args.distance_stages_m.split(",")]
    rng = random.Random(stable_int(args.seed, state, "center-sampling"))
    accepted: list[dict] = []
    accepted_xy: list[tuple[float, float]] = []
    source_failures = defaultdict(int)
    source_accepts = defaultdict(int)

    opened: dict[str, object] = {}
    try:
        for distance in distance_stages:
            if len(accepted) >= args.target_per_state:
                break
            index = DistanceIndex(distance, accepted_xy)
            stall_rounds = 0
            for round_index in range(args.rounds_per_stage):
                if len(accepted) >= args.target_per_state:
                    break
                order = sources[:]
                rng.shuffle(order)
                progress = 0
                for source_row in order:
                    path = source_row["prepared_path"]
                    if path not in opened:
                        try:
                            opened[path] = open_source(source_row)
                        except Exception as exc:
                            LOG.warning("%s: reject source %s: %s", state, path, exc)
                            opened[path] = None
                    src = opened[path]
                    if src is None:
                        continue
                    for _ in range(args.candidate_attempts_per_source):
                        candidate = candidate_from_source(src, state_geom, rng)
                        if candidate is None:
                            source_failures[path] += 1
                            continue
                        col0, row0, x, y = candidate
                        if index.too_close(x, y):
                            continue
                        ratio = valid_ratio(src, col0, row0)
                        if ratio + 1e-12 < args.min_valid_ratio:
                            source_failures[path] += 1
                            continue
                        center_id = stable_id(state, round(x, 3), round(y, 3), n=16)
                        accepted.append(
                            {
                                "center_id": center_id,
                                "state": state,
                                "center_rank": len(accepted) + 1,
                                "center_x_5070": f"{x:.3f}",
                                "center_y_5070": f"{y:.3f}",
                                "s10_valid_ratio": f"{ratio:.9f}",
                                "distance_stage_m": f"{distance:.1f}",
                                "download_key": source_row["download_key"],
                                "prepared_path": str(Path(path).resolve()),
                                "s10_col_off": col0,
                                "s10_row_off": row0,
                                "seed": args.seed,
                            }
                        )
                        accepted_xy.append((x, y))
                        index.add(x, y)
                        source_accepts[path] += 1
                        progress += 1
                        break
                    if len(accepted) >= args.target_per_state:
                        break
                LOG.info(
                    "%s: distance=%.0f round=%d accepted=%d (+%d)",
                    state, distance, round_index + 1, len(accepted), progress,
                )
                if progress == 0:
                    stall_rounds += 1
                    if stall_rounds >= args.max_stall_rounds:
                        LOG.info(
                            "%s: distance=%.0f stop after %d consecutive no-progress rounds",
                            state, distance, stall_rounds,
                        )
                        break
                else:
                    stall_rounds = 0
    finally:
        for src in opened.values():
            if src is not None:
                src.close()

    accepted = accepted[: args.target_per_state]
    nested: list[dict] = []
    for center in accepted:
        max_col = int(center["s10_col_off"])
        max_row = int(center["s10_row_off"])
        for scale, size in SCALES.items():
            inset = (SCALES["S10"] - size) // 2
            col = max_col + inset
            row = max_row + inset
            suffix = ".vrt" if args.output_mode == "VRT" else ".tif"
            relpath = Path("tiles") / scale / state / f"{center['center_id']}_{scale}{suffix}"
            sample_path = out_root / relpath
            if args.output_mode != "manifest":
                create_sample(center["prepared_path"], sample_path, col, row, size, args.output_mode)
            nested.append(
                {
                    "center_id": center["center_id"],
                    "state": state,
                    "scale": scale,
                    "size_pixels": size,
                    "resolution_m": 1,
                    "center_x_5070": center["center_x_5070"],
                    "center_y_5070": center["center_y_5070"],
                    "download_key": center["download_key"],
                    "prepared_path": center["prepared_path"],
                    "col_off": col,
                    "row_off": row,
                    "valid_ratio_from_s10": center["s10_valid_ratio"],
                    "sample_relpath": str(relpath) if args.output_mode != "manifest" else "",
                    "sample_path": str(sample_path.resolve()) if args.output_mode != "manifest" else "",
                }
            )

    write_csv(state_center_path, accepted, CENTER_FIELDS)
    write_csv(state_nested_path, nested, NESTED_FIELDS)
    diagnostics = [
        {
            "prepared_path": path,
            "accepted_centers": source_accepts[path],
            "failed_candidates": source_failures[path],
        }
        for path in sorted(set(source_accepts) | set(source_failures))
    ]
    write_csv(
        out_root / "diagnostics" / f"sources_{state}.csv",
        diagnostics,
        ("prepared_path", "accepted_centers", "failed_candidates"),
    )
    return len(accepted), accepted, nested


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    if not (0.0 < args.min_valid_ratio <= 1.0):
        raise ValueError("--min-valid-ratio must be in (0,1]")
    if args.rounds_per_stage < 1 or args.max_stall_rounds < 1:
        raise ValueError("--rounds-per-stage and --max-stall-rounds must be positive")
    if args.candidate_attempts_per_source < 1:
        raise ValueError("--candidate-attempts-per-source must be positive")
    states = parse_states(args.states)
    state_geoms = load_state_geometries(args.state_boundaries, states, TARGET_CRS)
    plan_rows = read_csv(args.anchor_plan)
    index_rows = read_csv(args.source_index)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    short: list[tuple[str, int]] = []
    for state in states:
        sources = rows_for_state(plan_rows, index_rows, state)
        count, _, _ = sample_state(args, state, state_geoms[state], sources, out_root)
        LOG.info("%s: final centers=%d target=%d sources=%d", state, count, args.target_per_state, len(sources))
        if count < args.target_per_state:
            short.append((state, count))
    if short:
        LOG.error("States below target: %s", ", ".join(f"{s}={n}" for s, n in short))
        if args.require_target:
            return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
