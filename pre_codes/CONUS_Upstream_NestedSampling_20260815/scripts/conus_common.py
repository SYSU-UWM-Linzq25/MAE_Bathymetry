#!/usr/bin/env python3
"""Shared helpers for the CONUS nested 3DEP sampling workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence
from urllib.parse import unquote, urlparse

from osgeo import ogr, osr

ogr.UseExceptions()
osr.UseExceptions()


CONUS48 = (
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
)

SCALES = {"S1": 336, "S3": 1008, "S5": 1680, "S10": 3360}
TARGET_CRS = "EPSG:5070"
NODATA = -999999.0


def stable_int(*parts: object, bits: int = 64) -> int:
    text = "|".join(str(p) for p in parts).encode("utf-8")
    raw = hashlib.sha256(text).digest()
    return int.from_bytes(raw[: bits // 8], "big", signed=False)


def stable_id(prefix: str, *parts: object, n: int = 12) -> str:
    text = "|".join(str(p) for p in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(text).hexdigest()[:n]}"


def parse_states(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(CONUS48)
    out: list[str] = []
    for value in values:
        for token in re.split(r"[\s,]+", value.strip().upper()):
            if token:
                out.append(token)
    invalid = sorted(set(out) - set(CONUS48))
    if invalid:
        raise ValueError(f"Not CONUS state abbreviations: {invalid}")
    return list(dict.fromkeys(out))


def load_state_geometries(path: str | Path, states: Sequence[str], crs: str):
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise ValueError(f"Cannot open state boundary file: {path}")
    layer = dataset.GetLayer(0)
    if layer is None:
        raise ValueError(f"State boundary file has no readable layer: {path}")

    definition = layer.GetLayerDefn()
    fields = [definition.GetFieldDefn(i).GetName() for i in range(definition.GetFieldCount())]
    field_lookup = {name.upper(): name for name in fields}
    state_field = next(
        (field_lookup[name] for name in ("STUSPS", "STATE_ABBR", "STATE", "POSTAL") if name in field_lookup),
        None,
    )
    if state_field is None:
        raise ValueError(f"Cannot find state abbreviation field in {path}; columns={fields}")

    source_crs = layer.GetSpatialRef()
    if source_crs is None:
        raise ValueError(f"State boundary file has no CRS: {path}")
    source_crs = source_crs.Clone()
    target_crs = osr.SpatialReference()
    if target_crs.SetFromUserInput(crs) != 0:
        raise ValueError(f"Cannot parse target CRS: {crs}")
    if hasattr(source_crs, "SetAxisMappingStrategy"):
        source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = None if source_crs.IsSame(target_crs) else osr.CoordinateTransformation(source_crs, target_crs)

    requested = set(states)
    grouped = {state: [] for state in states}
    layer.ResetReading()
    for feature in layer:
        state = str(feature.GetField(state_field) or "").upper()
        if state not in requested:
            continue
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            continue
        geometry = geometry.Clone()
        if transform is not None and geometry.Transform(transform) != 0:
            raise ValueError(f"Failed to transform {state} geometry to {crs}")
        grouped[state].append(geometry)

    missing = sorted(state for state, geometries in grouped.items() if not geometries)
    if missing:
        raise ValueError(f"State boundary file is missing: {missing}")

    result = {}
    for state in states:
        geometries = grouped[state]
        merged = geometries[0].Clone()
        for geometry in geometries[1:]:
            unioned = merged.Union(geometry)
            if unioned is None:
                raise ValueError(f"Failed to merge multipart state geometry: {state}")
            merged = unioned
        result[state] = merged
    return result


def item_bbox(item: Mapping[str, object]):
    value = item.get("boundingBox")
    if not isinstance(value, Mapping):
        return None
    aliases = (
        ("minX", "minY", "maxX", "maxY"),
        ("west", "south", "east", "north"),
    )
    for names in aliases:
        try:
            vals = [float(value[name]) for name in names]
        except (KeyError, TypeError, ValueError):
            continue
        if vals[0] < vals[2] and vals[1] < vals[3]:
            return tuple(vals)
    return None


def bbox_geometry(bounds: Sequence[float]):
    minx, miny, maxx, maxy = (float(value) for value in bounds)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint_2D(minx, miny)
    ring.AddPoint_2D(maxx, miny)
    ring.AddPoint_2D(maxx, maxy)
    ring.AddPoint_2D(minx, maxy)
    ring.AddPoint_2D(minx, miny)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon


def geometry_bounds(geometry) -> tuple[float, float, float, float]:
    minx, maxx, miny, maxy = geometry.GetEnvelope()
    return float(minx), float(miny), float(maxx), float(maxy)


def geometry_intersects_bbox(geometry, bounds: Sequence[float]) -> bool:
    return bool(geometry.Intersects(bbox_geometry(bounds)))


def download_url(item: Mapping[str, object]) -> str | None:
    for key in ("downloadURL", "downloadUrl", "url"):
        value = item.get(key)
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            return value
    return None


def remote_basename(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    if not name:
        name = "source.tif"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:180]


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temp.replace(path)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_rows(paths: Iterable[str | Path]) -> Iterator[dict[str, str]]:
    for path in paths:
        yield from read_csv(path)


def point_block(value: float, size: float) -> int:
    return math.floor(value / size)
