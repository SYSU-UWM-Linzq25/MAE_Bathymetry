#!/usr/bin/env python3
"""Choose spatially distributed random 1 m DEM anchor products and make a deduplicated download plan."""

from __future__ import annotations

import argparse
import csv
import logging
import random
from collections import defaultdict
from pathlib import Path

from osgeo import osr

from conus_common import (
    TARGET_CRS,
    download_url,
    item_bbox,
    parse_states,
    read_json,
    remote_basename,
    stable_id,
    stable_int,
    write_csv,
)


LOG = logging.getLogger("anchor-plan")


FIELDS = (
    "state", "state_anchor_rank", "anchor_key", "download_key", "url", "local_relpath",
    "title", "publication_date", "min_lon", "min_lat", "max_lon", "max_lat",
    "centroid_x_5070", "centroid_y_5070", "selection_grid_x", "selection_grid_y",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--states", nargs="*", default=None)
    p.add_argument("--anchors-per-state", type=int, default=150)
    p.add_argument("--selection-grid-m", type=float, default=20000.0)
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--log-level", default="INFO")
    return p


def date_key(item: dict) -> str:
    return str(item.get("lastUpdated") or item.get("publicationDate") or "")


def spatial_key(item: dict, url: str) -> str:
    bounds = item_bbox(item)
    if bounds is None:
        return stable_id("url", url)
    vals = tuple(round(v, 5) for v in bounds)
    return "bbox_" + "_".join(f"{v:.5f}" for v in vals)


def unique_latest_items(items: list[dict]) -> list[dict]:
    by_url: dict[str, dict] = {}
    for item in items:
        url = download_url(item)
        if not url:
            continue
        old = by_url.get(url)
        if old is None or date_key(item) > date_key(old):
            by_url[url] = item

    # TNMAccess can expose an older and newer copy of the same 10 km footprint.
    by_spatial: dict[str, tuple[str, dict]] = {}
    for url, item in by_url.items():
        key = spatial_key(item, url)
        old = by_spatial.get(key)
        if old is None or date_key(item) > date_key(old[1]):
            by_spatial[key] = (url, item)
    return [item for _, item in by_spatial.values()]


def choose_balanced(items: list[dict], n: int, grid_m: float, seed: int, state: str) -> list[dict]:
    source_crs = osr.SpatialReference()
    target_crs = osr.SpatialReference()
    source_crs.SetFromUserInput("EPSG:4326")
    target_crs.SetFromUserInput(TARGET_CRS)
    if hasattr(source_crs, "SetAxisMappingStrategy"):
        source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source_crs, target_crs)
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    missing_bbox: list[dict] = []
    for item in items:
        bounds = item_bbox(item)
        if bounds is None:
            missing_bbox.append(item)
            continue
        minx, miny, maxx, maxy = bounds
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        x, y, _ = transform.TransformPoint(cx, cy)
        item = dict(item)
        item["_centroid_x_5070"] = x
        item["_centroid_y_5070"] = y
        item["_selection_grid_x"] = int(x // grid_m)
        item["_selection_grid_y"] = int(y // grid_m)
        buckets[(item["_selection_grid_x"], item["_selection_grid_y"])].append(item)

    rng = random.Random(stable_int(seed, state, "anchor-selection"))
    for values in buckets.values():
        rng.shuffle(values)
    bucket_keys = list(buckets)
    rng.shuffle(bucket_keys)

    chosen: list[dict] = []
    layer = 0
    while len(chosen) < n:
        progressed = False
        order = bucket_keys[:]
        rng.shuffle(order)
        for key in order:
            values = buckets[key]
            if layer < len(values):
                chosen.append(values[layer])
                progressed = True
                if len(chosen) >= n:
                    break
        if not progressed:
            break
        layer += 1

    if len(chosen) < n and missing_bbox:
        rng.shuffle(missing_bbox)
        chosen.extend(missing_bbox[: n - len(chosen)])
    return chosen[:n]


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    states = parse_states(args.states)
    inventory_dir = Path(args.inventory_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_rows: list[dict] = []
    for state in states:
        data = read_json(inventory_dir / f"{state}.json")
        items = unique_latest_items(list(data.get("items", [])))
        chosen = choose_balanced(items, args.anchors_per_state, args.selection_grid_m, args.seed, state)
        if len(chosen) < args.anchors_per_state:
            LOG.warning("%s: requested %d anchors but inventory supports %d", state, args.anchors_per_state, len(chosen))
        state_rows: list[dict] = []
        for rank, item in enumerate(chosen, start=1):
            url = download_url(item)
            if not url:
                continue
            download_key = stable_id("src", url)
            bounds = item_bbox(item) or ("", "", "", "")
            x = item.get("_centroid_x_5070", "")
            y = item.get("_centroid_y_5070", "")
            local_name = f"{download_key}_{remote_basename(url)}"
            row = {
                "state": state,
                "state_anchor_rank": rank,
                "anchor_key": stable_id("anchor", state, url),
                "download_key": download_key,
                "url": url,
                "local_relpath": f"raw/{download_key[:7]}/{local_name}",
                "title": item.get("title", ""),
                "publication_date": item.get("publicationDate") or item.get("lastUpdated") or "",
                "min_lon": bounds[0], "min_lat": bounds[1], "max_lon": bounds[2], "max_lat": bounds[3],
                "centroid_x_5070": x, "centroid_y_5070": y,
                "selection_grid_x": item.get("_selection_grid_x", ""),
                "selection_grid_y": item.get("_selection_grid_y", ""),
            }
            state_rows.append(row)
            plan_rows.append(row)
        write_csv(out_dir / "by_state" / f"anchors_{state}.csv", state_rows, FIELDS)
        LOG.info("%s: planned %d anchor products", state, len(state_rows))

    write_csv(out_dir / "anchor_plan.csv", plan_rows, FIELDS)

    by_url: dict[str, dict] = {}
    for row in plan_rows:
        by_url.setdefault(row["url"], row)
    download_rows = [
        {
            "download_key": row["download_key"],
            "url": row["url"],
            "local_relpath": row["local_relpath"],
        }
        for row in sorted(by_url.values(), key=lambda x: x["download_key"])
    ]
    manifest = out_dir / "download_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("download_key", "url", "local_relpath"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(download_rows)

    state_counts = defaultdict(int)
    for row in plan_rows:
        state_counts[row["state"]] += 1
    summary = [
        {"state": state, "planned_anchors": state_counts[state]}
        for state in states
    ]
    write_csv(out_dir / "anchor_summary.csv", summary, ("state", "planned_anchors"))
    LOG.info("Plan complete: %d state-anchor rows; %d unique downloads", len(plan_rows), len(download_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
