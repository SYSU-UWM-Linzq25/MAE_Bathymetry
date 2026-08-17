#!/usr/bin/env python3
"""Query and cache TNMAccess 1 m DEM product inventory for each CONUS state."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import requests

from conus_common import (
    download_url,
    geometry_bounds,
    geometry_intersects_bbox,
    item_bbox,
    load_state_geometries,
    parse_states,
    write_csv,
    write_json,
)


LOG = logging.getLogger("tnm-inventory")
DEFAULT_ENDPOINT = "https://tnmaccess.nationalmap.gov/api/v1/products"
DEFAULT_DATASET = "Digital Elevation Model (DEM) 1 meter"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-boundaries", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--states", nargs="*", default=None)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--product-format", default="GeoTIFF")
    p.add_argument("--page-size", type=int, default=500)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--request-delay", type=float, default=0.25)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


def request_page(session: requests.Session, endpoint: str, params: dict, timeout: int, retries: int):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(endpoint, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("items", []), list):
                raise RuntimeError(f"Unexpected TNMAccess response keys: {list(data) if isinstance(data, dict) else type(data)}")
            return data
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            delay = min(60.0, 2.0 ** attempt)
            LOG.warning("Request failed (%s/%s): %s; retry in %.1fs", attempt + 1, retries, exc, delay)
            time.sleep(delay)
    raise RuntimeError(f"TNMAccess failed after {retries} attempts: {last_error}")


def query_state(session, args, state: str, geom_wgs84) -> tuple[list[dict], int | None]:
    minx, miny, maxx, maxy = geometry_bounds(geom_wgs84)
    offset = 0
    raw_total: int | None = None
    found: list[dict] = []
    seen: set[str] = set()

    while True:
        params = {
            "datasets": args.dataset,
            "bbox": f"{minx:.8f},{miny:.8f},{maxx:.8f},{maxy:.8f}",
            "prodFormats": args.product_format,
            "max": args.page_size,
            "offset": offset,
        }
        data = request_page(session, args.endpoint, params, args.timeout, args.retries)
        items = data.get("items", [])
        if raw_total is None:
            try:
                raw_total = int(data.get("total"))
            except (TypeError, ValueError):
                raw_total = None

        for item in items:
            if not isinstance(item, dict):
                continue
            url = download_url(item)
            if not url or url in seen:
                continue
            footprint = item_bbox(item)
            if footprint is not None and not geometry_intersects_bbox(geom_wgs84, footprint):
                continue
            seen.add(url)
            found.append(item)

        LOG.info("%s: offset=%d page=%d kept=%d total=%s", state, offset, len(items), len(found), raw_total)
        if not items:
            break
        offset += len(items)
        if raw_total is not None and offset >= raw_total:
            break
        if len(items) < args.page_size and raw_total is None:
            break
        time.sleep(args.request_delay)

    return found, raw_total


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    states = parse_states(args.states)
    geometries = load_state_geometries(args.state_boundaries, states, "EPSG:4326")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "UWM-MAE-CONUS-sampler/20260815"})

    summary: list[dict] = []
    for state in states:
        state_path = out_dir / f"{state}.json"
        if state_path.exists() and not args.overwrite:
            cached = __import__("json").loads(state_path.read_text(encoding="utf-8"))
            items = cached.get("items", [])
            raw_total = cached.get("raw_total")
            LOG.info("%s: using cached inventory with %d items", state, len(items))
        else:
            items, raw_total = query_state(session, args, state, geometries[state])
            write_json(
                state_path,
                {
                    "state": state,
                    "endpoint": args.endpoint,
                    "dataset": args.dataset,
                    "product_format": args.product_format,
                    "raw_total": raw_total,
                    "items": items,
                },
            )
        total_bytes = 0
        for item in items:
            try:
                total_bytes += int(item.get("sizeInBytes", 0) or 0)
            except (TypeError, ValueError):
                pass
        summary.append(
            {
                "state": state,
                "kept_products": len(items),
                "raw_query_total": raw_total if raw_total is not None else "",
                "reported_size_bytes": total_bytes,
                "inventory_json": str(state_path),
            }
        )

    write_csv(
        out_dir / "inventory_summary.csv",
        summary,
        ("state", "kept_products", "raw_query_total", "reported_size_bytes", "inventory_json"),
    )
    empty = [row["state"] for row in summary if int(row["kept_products"]) == 0]
    if empty:
        LOG.error("No 1 m DEM products returned for: %s", ",".join(empty))
        return 2
    LOG.info("Inventory complete: %d states", len(states))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
