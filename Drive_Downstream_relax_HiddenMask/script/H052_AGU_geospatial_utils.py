#!/usr/bin/env python3
"""Minimal geospatial utilities for the AGU representative-reach figures.

This helper intentionally contains no project-specific prediction or result
paths. It can be copied independently into either the strict or relaxed
project's script directory.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tifffile

NODATA = -999999.0
NODATA_THRESHOLD = -9999.0


@dataclass(frozen=True)
class Affine:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


def tag_value(tags, key, default=None):
    tag = tags.get(key)
    return default if tag is None else tag.value


def transform_from_tags(tags) -> Affine:
    scale = (
        tag_value(tags, 33550, None)
        or tag_value(tags, "ModelPixelScaleTag", None)
    )
    tie = (
        tag_value(tags, 33922, None)
        or tag_value(tags, "ModelTiepointTag", None)
    )
    matrix = (
        tag_value(tags, 34264, None)
        or tag_value(tags, "ModelTransformationTag", None)
    )
    if scale is not None and tie is not None:
        scale = tuple(float(value) for value in scale)
        tie = tuple(float(value) for value in tie)
        sx, sy = abs(scale[0]), abs(scale[1])
        return Affine(
            sx,
            0.0,
            tie[3] - tie[0] * sx,
            0.0,
            -sy,
            tie[4] + tie[1] * sy,
        )
    if matrix is not None:
        matrix = tuple(float(value) for value in matrix)
        return Affine(
            matrix[0],
            matrix[1],
            matrix[3],
            matrix[4],
            matrix[5],
            matrix[7],
        )
    raise RuntimeError("Missing GeoTIFF transform tags")


@lru_cache(maxsize=256)
def read_tif(path_text: str) -> Tuple[np.ndarray, Affine]:
    path = Path(path_text)
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        array = np.asarray(page.asarray()).squeeze()
        transform = transform_from_tags(page.tags)
    if array.ndim != 2:
        raise RuntimeError(f"Expected 2D TIFF, got {array.shape}: {path}")
    return array, transform


def valid_gt(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float64, copy=False)
    return (
        np.isfinite(values)
        & (values > NODATA_THRESHOLD)
        & (values != NODATA)
    )


def valid_pred(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float64, copy=False)
    return (
        np.isfinite(values)
        & (values > NODATA_THRESHOLD)
        & (values != NODATA)
    )


def divide(sum_array: np.ndarray, count_array: np.ndarray) -> np.ndarray:
    output = np.full(sum_array.shape, np.nan, dtype=np.float32)
    valid = count_array > 0
    output[valid] = (
        sum_array[valid] / count_array[valid]
    ).astype(np.float32)
    return output


def affine_compatible(
    first: Affine,
    second: Affine,
    tolerance: float = 1e-6,
) -> bool:
    return all(
        abs(a - b) <= tolerance
        for a, b in zip(
            (first.a, first.b, first.c, first.d, first.e, first.f),
            (second.a, second.b, second.c, second.d, second.e, second.f),
        )
    )


def resolve_river_dir(
    root: Path,
    expected_experiment: str,
    preset: str,
    river: str,
) -> Path:
    direct = root / expected_experiment / river
    if (
        direct.is_dir()
        and list(direct.glob("*tileavg_prediction_manifest.csv"))
    ):
        return direct

    candidates: List[Path] = []
    for manifest in root.rglob("*tileavg_prediction_manifest.csv"):
        parent = manifest.parent
        text = str(parent).lower()
        if river.lower() not in text:
            continue
        if (
            preset.lower() not in text
            and f"holdout_{preset.lower()}" not in text
        ):
            continue
        candidates.append(parent)

    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(
            f"No prediction manifest for preset={preset}, river={river} "
            f"below {root}"
        )

    preferred = [
        path
        for path in unique
        if expected_experiment.lower() in str(path).lower()
    ]
    if len(preferred) == 1:
        return preferred[0]

    raise RuntimeError(
        "Ambiguous per-river prediction directories for "
        f"preset={preset}, river={river}, root={root}:\n"
        + "\n".join(str(path) for path in unique)
    )


def locate_manifest(
    river_dir: Path,
) -> Tuple[Path, Optional[Path]]:
    manifests = sorted(
        river_dir.glob("*tileavg_prediction_manifest.csv")
    )
    if not manifests:
        raise FileNotFoundError(f"No prediction manifest in {river_dir}")
    summaries = sorted(river_dir.glob("*summary.json"))
    return manifests[0], summaries[0] if summaries else None


MANIFEST_PATH_FIELDS = (
    "tile_path",
    "avg_pred_tile_path",
    "hidden_path",
    "loss_path",
    "core_loss_path",
)


def resolve_manifest_file(
    raw_value: str,
    river_dir: Path,
    field: str,
    manifest_path: Path,
) -> Tuple[Path, str]:
    raw = Path(str(raw_value))
    if raw.is_file():
        return raw, "original"

    candidates: List[Tuple[Path, str]] = []
    parts = raw.parts
    indices = [
        index
        for index, part in enumerate(parts)
        if part == river_dir.name
    ]
    for index in reversed(indices):
        suffix = parts[index + 1 :]
        if suffix:
            candidates.append(
                (
                    river_dir.joinpath(*suffix),
                    "rebase_after_river",
                )
            )

    if raw.parent.name:
        candidates.append(
            (
                river_dir / raw.parent.name / raw.name,
                "rebase_parent_and_name",
            )
        )
    candidates.append((river_dir / raw.name, "rebase_filename"))

    seen = set()
    valid: List[Tuple[Path, str]] = []
    for candidate, method in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            valid.append((candidate, method))

    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        preferred = [
            item
            for item in valid
            if item[1] == "rebase_after_river"
        ]
        if len(preferred) == 1:
            return preferred[0]
        raise RuntimeError(
            f"Ambiguous moved manifest path for field={field}: "
            f"{raw_value}\n"
            + "\n".join(
                f"  {method}: {path}"
                for path, method in valid
            )
        )

    matches = sorted(river_dir.rglob(raw.name))
    if len(matches) == 1:
        return matches[0], "basename_search_in_current_river"
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous basename while resolving field={field}: "
            f"{raw_value}\n"
            + "\n".join(str(path) for path in matches[:20])
        )

    raise FileNotFoundError(
        "Manifest path does not exist and could not be rebased.\n"
        f"  manifest={manifest_path}\n"
        f"  current_river_dir={river_dir}\n"
        f"  field={field}\n"
        f"  stored_path={raw_value}"
    )


def read_manifest_with_rebased_paths(
    manifest_path: Path,
    river_dir: Path,
    config_label: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    audit: List[Dict[str, str]] = []
    method_counts: Dict[str, int] = {}

    for row_index, row in enumerate(rows, start=2):
        for field in MANIFEST_PATH_FIELDS:
            raw_value = row.get(field, "")
            if not raw_value:
                continue
            resolved, method = resolve_manifest_file(
                raw_value,
                river_dir,
                field,
                manifest_path,
            )
            row[field] = str(resolved)
            method_counts[method] = (
                method_counts.get(method, 0) + 1
            )
            if method != "original":
                audit.append(
                    {
                        "configuration": config_label,
                        "manifest": str(manifest_path),
                        "csv_row": str(row_index),
                        "field": field,
                        "stored_path": str(raw_value),
                        "resolved_path": str(resolved),
                        "resolution_method": method,
                    }
                )

    print(
        f"[PATH-QA] {config_label}: rows={len(rows)} "
        + " ".join(
            f"{method}={count}"
            for method, count in sorted(method_counts.items())
        )
    )
    return rows, audit


PROCESSED_TILE_SUBDIRS = {
    "tile_path": "FullRiver_tile",
    "hidden_path": "Hidden_Mask",
    "loss_path": "Loss_Mask_Pixel",
    "core_loss_path": "Core_Loss_Mask_Pixel",
}


def resolve_processed_tile_root(base: Path) -> Path:
    candidates = (base / "Tiles_1m", base)
    required = tuple(PROCESSED_TILE_SUBDIRS.values())
    valid = [
        candidate
        for candidate in candidates
        if all(
            (candidate / subdir).is_dir()
            for subdir in required
        )
    ]
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        for candidate in valid:
            if candidate.name == "Tiles_1m":
                return candidate
    raise FileNotFoundError(
        "Could not resolve processed full-river tile root. "
        "Expected these directories under either the supplied "
        "path or its Tiles_1m child:\n"
        + "\n".join(
            f"  {name}" for name in required
        )
        + f"\nSupplied base: {base}"
    )


def derived_processed_tile_name(field: str, key: str) -> str:
    if field == "tile_path":
        return f"E001_FullRiver_tile_{key}.tif"
    if field == "hidden_path":
        return f"E001_tile_{key}_HiddenMask.tif"
    if field == "loss_path":
        return f"E001_tile_{key}_LossMaskPixel.tif"
    if field == "core_loss_path":
        return f"E001_tile_{key}_CoreLossMaskPixel.tif"
    raise KeyError(field)


@lru_cache(maxsize=65536)
def resolve_processed_tile_paths_cached(
    tile_root_text: str,
    key: str,
    stored_tile_path: str,
    stored_hidden_path: str,
    stored_loss_path: str,
    stored_core_loss_path: str,
) -> Tuple[Tuple[str, str], ...]:
    tile_root = Path(tile_root_text)
    stored = {
        "tile_path": stored_tile_path,
        "hidden_path": stored_hidden_path,
        "loss_path": stored_loss_path,
        "core_loss_path": stored_core_loss_path,
    }

    resolved: Dict[str, str] = {}
    failures: List[str] = []

    for field, subdir in PROCESSED_TILE_SUBDIRS.items():
        candidates: List[Path] = []
        raw = stored.get(field, "")
        if raw:
            candidates.append(
                tile_root / subdir / Path(raw).name
            )
        candidates.append(
            tile_root
            / subdir
            / derived_processed_tile_name(field, key)
        )

        unique: List[Path] = []
        seen = set()
        for candidate in candidates:
            candidate_text = str(candidate)
            if candidate_text not in seen:
                seen.add(candidate_text)
                unique.append(candidate)

        existing = [
            candidate
            for candidate in unique
            if candidate.is_file()
        ]
        if len(existing) == 1:
            resolved[field] = str(existing[0])
        elif len(existing) > 1:
            if len(
                {path.resolve() for path in existing}
            ) == 1:
                resolved[field] = str(existing[0])
            else:
                failures.append(
                    f"{field}: ambiguous candidates: "
                    + ", ".join(
                        str(path) for path in existing
                    )
                )
        else:
            failures.append(
                f"{field}: none of these files exists: "
                + ", ".join(
                    str(path) for path in unique
                )
            )

    if failures:
        raise FileNotFoundError(
            f"Processed tile files are incomplete for key={key} "
            f"under {tile_root}:\n  "
            + "\n  ".join(failures)
        )

    return tuple(sorted(resolved.items()))


def processed_tile_paths(
    tile_root: Path,
    row: Mapping[str, str],
) -> Dict[str, str]:
    key = str(row.get("key", ""))
    if not key:
        raise RuntimeError("Manifest row has no key.")
    return dict(
        resolve_processed_tile_paths_cached(
            str(tile_root),
            key,
            str(row.get("tile_path", "")),
            str(row.get("hidden_path", "")),
            str(row.get("loss_path", "")),
            str(row.get("core_loss_path", "")),
        )
    )
