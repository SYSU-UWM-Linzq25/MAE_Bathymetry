#!/usr/bin/env python3
"""H046: continuous-reach visualization using aligned processed tile branches.

For every continuous reach, four separate six-panel figures are generated:

1. Strict mask + normalized objective
2. Strict mask + meter objective
3. Relaxed mask + normalized objective
4. Relaxed mask + meter objective

Strict figures read GT and masks from:
    Tiles_for_MAE_FullRiver_E001

Relaxed figures read GT and masks from:
    Tiles_for_MAE_FullRiver_E001c_AnyVisiblePatch

Formal overlap-averaged full-river predictions remain the prediction source.

Each figure contains:

1. Processed full GT with sampling centers
2. Patch-processed Hidden Mask (0/1; blank outside valid processed GT)
3. Final prediction/loss mask (0/1; blank outside valid processed GT)
4. GT inside the final mask
5. Prediction inside the final mask
6. Signed error inside the final mask

Each configuration calculates its own display ranges. Reach metrics are
evaluated on exact four-way common final pixels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import html
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

NODATA = -999999.0
NODATA_THRESHOLD = -9999.0

CASES = (
    {
        "preset": "CA",
        "label": "CA Klamath",
        "river": "CA_KlamathRiver_TopoBathy_2018_D18",
        "strict_norm_exp": "holdout_CA_D001NoDataSafe",
        "strict_meter_exp": "holdout_CA_D003MeterMAE_BaselineEval_D001NoDataSafe",
        "relax_norm_exp": "holdout_CA_D001cAnyVisiblePatch_D001NoDataSafe",
        "relax_meter_exp": "holdout_CA_D044MeterOnly_D001cAnyVisiblePatch_D001NoDataSafe",
    },
    {
        "preset": "CO",
        "label": "CO Upper Colorado",
        "river": "CO_UpperColorado_Topobathy_1_2020",
        "strict_norm_exp": "holdout_CO_D001NoDataSafe",
        "strict_meter_exp": "holdout_CO_D003MeterMAE_BaselineEval_D001NoDataSafe",
        "relax_norm_exp": "holdout_CO_D001cAnyVisiblePatch_D001NoDataSafe",
        "relax_meter_exp": "holdout_CO_D044MeterOnly_D001cAnyVisiblePatch_D001NoDataSafe",
    },
    {
        "preset": "Santiam",
        "label": "OR Santiam",
        "river": "OR_SantiamRiverTB_Topobathy_1_D23",
        "strict_norm_exp": "holdout_Santiam_D001NoDataSafe",
        "strict_meter_exp": "holdout_Santiam_D003MeterMAE_BaselineEval_D001NoDataSafe",
        "relax_norm_exp": "holdout_Santiam_D001cAnyVisiblePatch_D001NoDataSafe",
        "relax_meter_exp": "holdout_Santiam_D044MeterOnly_D001cAnyVisiblePatch_D001NoDataSafe",
    },
)

CONFIG_ORDER = (
    "strict_normalized",
    "strict_meter",
    "relaxed_normalized",
    "relaxed_meter",
)
CONFIG_LABELS = {
    "strict_normalized": "Strict mask + normalized objective",
    "strict_meter": "Strict mask + meter objective",
    "relaxed_normalized": "Relaxed mask + normalized objective",
    "relaxed_meter": "Relaxed mask + meter objective",
}


@dataclass(frozen=True)
class Affine:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


@dataclass(frozen=True)
class Segment:
    preset: str
    river: str
    river_label: str
    line_id: str
    segment_index: int
    reference_rows: Tuple[Dict[str, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create four separate six-panel local-reach figures.",
    )
    strict_results = Path(
        "/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/"
        "Downstream_Task_Bathy/Results"
    )
    relax_results = Path(
        "/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/"
        "Downstream_Task_Bathy_relax_HiddenMask/results"
    )
    parser.add_argument(
        "--strict_normalized_pred_root",
        type=Path,
        default=strict_results / "FullRiver_Predictions_F010_TileAvgVRT_D001NoDataSafe",
    )
    parser.add_argument(
        "--strict_meter_pred_root",
        type=Path,
        default=strict_results
        / "FullRiver_Predictions_F060_TileAvgVRT_D003MeterMAE_BaselineEval_D001NoDataSafe",
    )
    parser.add_argument(
        "--relaxed_normalized_pred_root",
        type=Path,
        default=relax_results
        / "FullRiver_Predictions_F049_NormalizedObjective_D001cAnyVisiblePatch",
    )
    parser.add_argument(
        "--relaxed_meter_pred_root",
        type=Path,
        default=relax_results
        / "FullRiver_Predictions_F044_MeterOnly_D001cAnyVisiblePatch",
    )
    parser.add_argument(
        "--strict_tile_base",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "Tiles_for_MAE_FullRiver_E001"
        ),
        help=(
            "Processed strict full-river tile branch. GT, Hidden Mask, "
            "Loss Mask and Core Loss Mask for strict figures are all read "
            "from this branch."
        ),
    )
    parser.add_argument(
        "--relax_tile_base",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "Tiles_for_MAE_FullRiver_E001c_AnyVisiblePatch"
        ),
        help=(
            "Processed relaxed full-river tile branch. GT, Hidden Mask, "
            "Loss Mask and Core Loss Mask for relaxed figures are all read "
            "from this branch."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--segment_size", type=int, default=10)
    parser.add_argument("--segment_stride", type=int, default=10)
    parser.add_argument("--min_points", type=int, default=5)
    parser.add_argument("--min_common_pixels", type=int, default=1000)
    parser.add_argument("--n_best", type=int, default=3)
    parser.add_argument("--n_median", type=int, default=3)
    parser.add_argument("--n_worst", type=int, default=3)
    parser.add_argument("--n_meter_advantage", type=int, default=3)
    parser.add_argument("--n_relaxed_advantage", type=int, default=3)
    parser.add_argument("--crop_padding", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--max_segment_dense_pixels", type=int, default=25000000)
    parser.add_argument(
        "--render_all_reaches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render four six-panel figures for every successfully assembled "
            "continuous reach, not only the selected examples."
        ),
    )
    parser.add_argument(
        "--resume_visuals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete reach figure folders from an interrupted run.",
    )
    parser.add_argument(
        "--render_overview_for_all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also render the optional 2x2 prediction overview for every reach. "
            "The four required six-panel figures are always rendered."
        ),
    )
    parser.add_argument(
        "--max_render_reaches",
        type=int,
        default=0,
        help="Debug limit per river; 0 means render every reach.",
    )
    parser.add_argument(
        "--render_progress_every",
        type=int,
        default=10,
        help="Print visualization progress every N reaches.",
    )
    return parser.parse_args()


def tag_value(tags, key, default=None):
    tag = tags.get(key)
    return default if tag is None else tag.value


def transform_from_tags(tags) -> Affine:
    scale = tag_value(tags, 33550, None) or tag_value(tags, "ModelPixelScaleTag", None)
    tie = tag_value(tags, 33922, None) or tag_value(tags, "ModelTiepointTag", None)
    matrix = tag_value(tags, 34264, None) or tag_value(tags, "ModelTransformationTag", None)
    if scale is not None and tie is not None:
        scale = tuple(float(value) for value in scale)
        tie = tuple(float(value) for value in tie)
        sx, sy = abs(scale[0]), abs(scale[1])
        return Affine(sx, 0.0, tie[3] - tie[0] * sx, 0.0, -sy, tie[4] + tie[1] * sy)
    if matrix is not None:
        matrix = tuple(float(value) for value in matrix)
        return Affine(matrix[0], matrix[1], matrix[3], matrix[4], matrix[5], matrix[7])
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


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))



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
    """Resolve a path stored in a manifest after result folders were moved.

    Historical strict prediction manifests contain absolute paths below
    ``Processed_Results/FullRiver_Predictions_*``. The prediction folders were
    later moved to ``Downstream_Task_Bathy/Results`` without rewriting the CSV.
    For moved prediction files, the suffix below the river directory is stable,
    so it can be rebased onto the current manifest directory.
    """
    raw = Path(str(raw_value))
    if raw.is_file():
        return raw, "original"

    candidates: List[Tuple[Path, str]] = []

    # Most reliable rule: keep the path suffix after the river folder.
    parts = raw.parts
    indices = [index for index, part in enumerate(parts) if part == river_dir.name]
    for index in reversed(indices):
        suffix = parts[index + 1 :]
        if suffix:
            candidates.append((river_dir.joinpath(*suffix), "rebase_after_river"))

    # Common moved-result layout: river_dir/<tile-output-folder>/<filename>.
    if raw.parent.name:
        candidates.append(
            (river_dir / raw.parent.name / raw.name, "rebase_parent_and_name")
        )

    # Defensive direct-file fallback.
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
        # Prefer the suffix-preserving rule.
        preferred = [
            item for item in valid if item[1] == "rebase_after_river"
        ]
        if len(preferred) == 1:
            return preferred[0]
        raise RuntimeError(
            f"Ambiguous moved manifest path for field={field}: {raw_value}\n"
            + "\n".join(f"  {method}: {path}" for path, method in valid)
        )

    # Last-resort basename search only inside this branch's current river folder.
    matches = sorted(river_dir.rglob(raw.name))
    if len(matches) == 1:
        return matches[0], "basename_search_in_current_river"
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous basename while resolving field={field}: {raw_value}\n"
            + "\n".join(f"  {path}" for path in matches[:20])
        )

    raise FileNotFoundError(
        "Manifest path does not exist and could not be rebased.\n"
        f"  manifest={manifest_path}\n"
        f"  current_river_dir={river_dir}\n"
        f"  field={field}\n"
        f"  stored_path={raw_value}\n"
        "The prediction directory may have been moved again, or the file is "
        "actually missing."
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
            method_counts[method] = method_counts.get(method, 0) + 1
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

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def locate_manifest(river_dir: Path) -> Tuple[Path, Optional[Path]]:
    manifests = sorted(river_dir.glob("*tileavg_prediction_manifest.csv"))
    if not manifests:
        raise FileNotFoundError(f"No prediction manifest in {river_dir}")
    summaries = sorted(river_dir.glob("*summary.json"))
    return manifests[0], summaries[0] if summaries else None


def validate_relaxed_summary(summary_path: Optional[Path], label: str) -> None:
    if summary_path is None:
        raise RuntimeError(f"Missing relaxed prediction summary for {label}")
    data = json.loads(summary_path.read_text())
    if data.get("prediction_patch_filter_applied") is not True:
        raise RuntimeError(
            f"Refusing legacy relaxed prediction without prediction_patch_filter_applied=true: {summary_path}"
        )


def valid_gt(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float64, copy=False)
    return np.isfinite(values) & (values > NODATA_THRESHOLD) & (values != NODATA)


def valid_pred(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float64, copy=False)
    return np.isfinite(values) & (values > NODATA_THRESHOLD) & (values != NODATA)


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def resolve_river_dir(root: Path, expected_experiment: str, preset: str, river: str) -> Path:
    """Resolve a per-river prediction directory without assuming one exact root layout."""
    direct = root / expected_experiment / river
    if direct.is_dir() and list(direct.glob("*tileavg_prediction_manifest.csv")):
        return direct

    candidates: List[Path] = []
    for manifest in root.rglob("*tileavg_prediction_manifest.csv"):
        parent = manifest.parent
        text = str(parent).lower()
        if river.lower() not in text:
            continue
        if preset.lower() not in text and f"holdout_{preset.lower()}" not in text:
            continue
        candidates.append(parent)

    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(
            f"No prediction manifest for preset={preset}, river={river} below {root}"
        )

    # Prefer the exact historical experiment tag when multiple copies exist.
    preferred = [path for path in unique if expected_experiment.lower() in str(path).lower()]
    if len(preferred) == 1:
        return preferred[0]

    raise RuntimeError(
        "Ambiguous per-river prediction directories for "
        f"preset={preset}, river={river}, root={root}:\n"
        + "\n".join(str(path) for path in unique)
    )


def branch_dirs(args: argparse.Namespace, case: Mapping[str, str]) -> Dict[str, Path]:
    return {
        "strict_normalized": resolve_river_dir(
            args.strict_normalized_pred_root, case["strict_norm_exp"], case["preset"], case["river"]
        ),
        "strict_meter": resolve_river_dir(
            args.strict_meter_pred_root, case["strict_meter_exp"], case["preset"], case["river"]
        ),
        "relaxed_normalized": resolve_river_dir(
            args.relaxed_normalized_pred_root, case["relax_norm_exp"], case["preset"], case["river"]
        ),
        "relaxed_meter": resolve_river_dir(
            args.relaxed_meter_pred_root, case["relax_meter_exp"], case["preset"], case["river"]
        ),
    }



PROCESSED_TILE_SUBDIRS = {
    "tile_path": "FullRiver_tile",
    "hidden_path": "Hidden_Mask",
    "loss_path": "Loss_Mask_Pixel",
    "core_loss_path": "Core_Loss_Mask_Pixel",
}


def resolve_processed_tile_root(base: Path) -> Path:
    """Resolve either <base> or <base>/Tiles_1m to the actual tile root."""
    candidates = (base / "Tiles_1m", base)
    required = tuple(PROCESSED_TILE_SUBDIRS.values())

    valid = [
        candidate
        for candidate in candidates
        if all((candidate / subdir).is_dir() for subdir in required)
    ]
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        # Prefer the explicit Tiles_1m level when both happen to exist.
        for candidate in valid:
            if candidate.name == "Tiles_1m":
                return candidate

    raise FileNotFoundError(
        "Could not resolve processed full-river tile root. Expected these "
        "directories under either the supplied path or its Tiles_1m child:\n"
        + "\n".join(f"  {name}" for name in required)
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
    """Resolve one manifest row against a specified processed tile branch.

    Basenames from the manifest are preferred because the strict and relaxed
    branches use the same E001 naming convention. If a basename is unavailable,
    the filename is derived from the manifest key.
    """
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
            candidates.append(tile_root / subdir / Path(raw).name)
        candidates.append(
            tile_root / subdir / derived_processed_tile_name(field, key)
        )

        unique: List[Path] = []
        seen = set()
        for candidate in candidates:
            candidate_text = str(candidate)
            if candidate_text not in seen:
                seen.add(candidate_text)
                unique.append(candidate)

        existing = [candidate for candidate in unique if candidate.is_file()]
        if len(existing) == 1:
            resolved[field] = str(existing[0])
        elif len(existing) > 1:
            # Both names normally point to the same filename. If not, fail
            # rather than silently mixing two tile generations.
            if len({path.resolve() for path in existing}) == 1:
                resolved[field] = str(existing[0])
            else:
                failures.append(
                    f"{field}: ambiguous candidates: "
                    + ", ".join(str(path) for path in existing)
                )
        else:
            failures.append(
                f"{field}: none of these files exists: "
                + ", ".join(str(path) for path in unique)
            )

    if failures:
        raise FileNotFoundError(
            f"Processed tile files are incomplete for key={key} under "
            f"{tile_root}:\n  " + "\n  ".join(failures)
        )

    return tuple(sorted(resolved.items()))


def processed_tile_paths(
    tile_root: Path,
    row: Mapping[str, str],
) -> Dict[str, str]:
    key = str(row.get("key", ""))
    if not key:
        raise RuntimeError("Manifest row has no key.")
    pairs = resolve_processed_tile_paths_cached(
        str(tile_root),
        key,
        str(row.get("tile_path", "")),
        str(row.get("hidden_path", "")),
        str(row.get("loss_path", "")),
        str(row.get("core_loss_path", "")),
    )
    return dict(pairs)


def affine_compatible(first: Affine, second: Affine, tolerance: float = 1e-6) -> bool:
    return all(
        abs(a - b) <= tolerance
        for a, b in zip(
            (first.a, first.b, first.c, first.d, first.e, first.f),
            (second.a, second.b, second.c, second.d, second.e, second.f),
        )
    )

def load_line_mapping(relax_tile_base: Path, river: str) -> Tuple[Dict[int, str], str]:
    qa = relax_tile_base / "QA" / river / f"E001c_candidate_QA_1m_{river}.csv"
    if not qa.is_file():
        return {}, "fallback_sequential_tile_id"
    frame = pd.read_csv(qa)
    lower = {column.lower(): column for column in frame.columns}
    point_column = lower.get("point_id")
    line_column = lower.get("line_id")
    kept_column = lower.get("kept")
    if point_column is None or line_column is None:
        return {}, f"fallback_missing_columns_in_{qa.name}"
    if kept_column is not None:
        frame = frame[pd.to_numeric(frame[kept_column], errors="coerce") > 0]
    mapping: Dict[int, str] = {}
    for point_id, line_id in zip(pd.to_numeric(frame[point_column], errors="coerce"), frame[line_column]):
        if np.isfinite(point_id):
            mapping[int(round(float(point_id)))] = str(line_id)
    return mapping, str(qa)


def build_segments(
    preset: str,
    river: str,
    label: str,
    reference_rows: Sequence[Dict[str, str]],
    all_maps: Mapping[str, Mapping[str, Dict[str, str]]],
    line_mapping: Mapping[int, str],
    size: int,
    stride: int,
    min_points: int,
) -> List[Segment]:
    by_line: Dict[str, List[Dict[str, str]]] = {}
    for row in reference_rows:
        key = row.get("key", "")
        if not key or any(key not in all_maps[config] for config in CONFIG_ORDER):
            continue
        tile_id = int(float(row["tile_id"]))
        line_id = line_mapping.get(tile_id, "SEQUENTIAL")
        by_line.setdefault(str(line_id), []).append(row)
    segments: List[Segment] = []
    segment_index = 0
    for line_id, rows in sorted(by_line.items()):
        rows = sorted(rows, key=lambda item: int(float(item["tile_id"])))
        for start in range(0, len(rows), stride):
            chunk = rows[start : start + size]
            if len(chunk) < min_points:
                continue
            segment_index += 1
            segments.append(
                Segment(
                    preset=preset,
                    river=river,
                    river_label=label,
                    line_id=line_id,
                    segment_index=segment_index,
                    reference_rows=tuple(chunk),
                )
            )
    return segments


def divide(sum_array: np.ndarray, count_array: np.ndarray) -> np.ndarray:
    output = np.full(sum_array.shape, np.nan, dtype=np.float32)
    valid = count_array > 0
    output[valid] = (sum_array[valid] / count_array[valid]).astype(np.float32)
    return output


def occurrence(sum_array: np.ndarray, count_array: np.ndarray) -> np.ndarray:
    output = np.full(sum_array.shape, np.nan, dtype=np.float32)
    valid = count_array > 0
    output[valid] = sum_array[valid] / count_array[valid]
    return output


def segment_data(
    segment: Segment,
    maps: Mapping[str, Mapping[str, Dict[str, str]]],
    tile_roots: Mapping[str, Path],
    max_dense_pixels: int,
) -> Dict[str, Any]:
    """Assemble a reach from branch-specific processed E001/E001c tiles.

    Strict configurations use the strict E001 branch for GT and masks.
    Relaxed configurations use the E001c AnyVisiblePatch branch.

    Predictions are still read from the formal overlap-averaged full-river
    prediction manifests. This keeps the model output unchanged while making
    GT, Hidden Mask and final mask spatially consistent with the exact
    processed branch used for inference.
    """
    bounds: List[Tuple[float, float, float, float]] = []

    # Use the relaxed processed GT tiles only to define the union grid. Strict
    # and relaxed tiles are required to have compatible georeferencing below.
    for reference_row in segment.reference_rows:
        key = reference_row["key"]
        row = maps["relaxed_meter"][key]
        source = processed_tile_paths(tile_roots["relaxed_meter"], row)
        gt, transform = read_tif(source["tile_path"])
        tile_height, tile_width = gt.shape
        bounds.append(
            (
                transform.c,
                transform.f,
                transform.c + tile_width * abs(transform.a),
                transform.f - tile_height * abs(transform.e),
            )
        )

    left = min(item[0] for item in bounds)
    top = max(item[1] for item in bounds)
    right = max(item[2] for item in bounds)
    bottom = min(item[3] for item in bounds)

    first_source = processed_tile_paths(
        tile_roots["relaxed_meter"],
        maps["relaxed_meter"][segment.reference_rows[0]["key"]],
    )
    _, first_transform = read_tif(first_source["tile_path"])
    resolution = abs(first_transform.a)

    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    if width * height > max_dense_pixels:
        raise RuntimeError(f"Reach grid too large: {height}x{width}")
    shape = (height, width)

    branch_data: Dict[str, Dict[str, Any]] = {}
    for config in CONFIG_ORDER:
        branch_data[config] = {
            "gt_sum": np.zeros(shape, dtype=np.float64),
            "gt_count": np.zeros(shape, dtype=np.uint16),
            "pred_sum": np.zeros(shape, dtype=np.float64),
            "pred_count": np.zeros(shape, dtype=np.uint16),
            "hidden_any": np.zeros(shape, dtype=bool),
            "raw_loss_any": np.zeros(shape, dtype=bool),
            "final_mask": np.zeros(shape, dtype=bool),
            "core": np.zeros(shape, dtype=bool),
            "valid_gt_footprint": np.zeros(shape, dtype=bool),
            "centers": [],
        }

    for reference_row in segment.reference_rows:
        key = reference_row["key"]

        for config in CONFIG_ORDER:
            row = maps[config][key]
            source = processed_tile_paths(tile_roots[config], row)

            branch_gt, transform = read_tif(source["tile_path"])
            hidden, hidden_transform = read_tif(source["hidden_path"])
            raw_loss, loss_transform = read_tif(source["loss_path"])
            core, core_transform = read_tif(source["core_loss_path"])
            pred, pred_transform = read_tif(row["avg_pred_tile_path"])

            for label, other_transform in (
                ("hidden", hidden_transform),
                ("loss", loss_transform),
                ("core_loss", core_transform),
                ("prediction", pred_transform),
            ):
                if not affine_compatible(transform, other_transform):
                    raise RuntimeError(
                        f"Georeference mismatch for key={key}, config={config}, "
                        f"layer={label}."
                    )

            if not (
                branch_gt.shape
                == hidden.shape
                == raw_loss.shape
                == core.shape
                == pred.shape
            ):
                raise RuntimeError(
                    f"Shape mismatch for key={key}, config={config}: "
                    f"GT={branch_gt.shape}, hidden={hidden.shape}, "
                    f"loss={raw_loss.shape}, core={core.shape}, pred={pred.shape}"
                )

            branch_gt = branch_gt.astype(np.float32, copy=False)
            pred = pred.astype(np.float32, copy=False)

            tile_height, tile_width = branch_gt.shape
            row0 = int(round((top - transform.f) / resolution))
            col0 = int(round((transform.c - left) / resolution))
            target_slice = np.s_[
                row0 : row0 + tile_height,
                col0 : col0 + tile_width,
            ]

            valid_branch_gt = valid_gt(branch_gt)
            hidden_mask = (
                np.isfinite(hidden)
                & (hidden.astype(np.float32) > 0.5)
                & valid_branch_gt
            )
            raw_loss_mask = (
                np.isfinite(raw_loss)
                & (raw_loss.astype(np.float32) > 0.5)
                & valid_branch_gt
            )
            core_mask = (
                np.isfinite(core)
                & (core.astype(np.float32) > 0.5)
                & valid_branch_gt
            )

            # The formal full-river prediction TIFF already contains only
            # genuine predicted pixels after overlap averaging. Therefore the
            # final local mask is the exact processed core-loss mask intersected
            # with valid processed GT and valid formal prediction.
            final_mask = core_mask & valid_pred(pred)

            branch = branch_data[config]

            target_gt_sum = branch["gt_sum"][target_slice]
            target_gt_count = branch["gt_count"][target_slice]
            target_gt_sum[valid_branch_gt] += branch_gt[valid_branch_gt]
            target_gt_count[valid_branch_gt] += 1

            target_pred_sum = branch["pred_sum"][target_slice]
            target_pred_count = branch["pred_count"][target_slice]
            target_pred_sum[final_mask] += pred[final_mask]
            target_pred_count[final_mask] += 1

            branch["hidden_any"][target_slice] |= hidden_mask
            branch["raw_loss_any"][target_slice] |= raw_loss_mask
            branch["final_mask"][target_slice] |= final_mask
            branch["core"][target_slice] |= core_mask
            branch["valid_gt_footprint"][target_slice] |= valid_branch_gt

            center_x = transform.c + tile_width * abs(transform.a) / 2.0
            center_y = transform.f - tile_height * abs(transform.e) / 2.0
            center_col = (center_x - left) / resolution
            center_row = (top - center_y) / resolution
            branch["centers"].append((center_col, center_row))

    output: Dict[str, Any] = {"resolution": resolution}
    common = np.ones(shape, dtype=bool)

    for config in CONFIG_ORDER:
        branch = branch_data[config]
        branch_gt = divide(branch["gt_sum"], branch["gt_count"])
        pred = divide(branch["pred_sum"], branch["pred_count"])
        final_mask = (
            branch["final_mask"]
            & np.isfinite(branch_gt)
            & np.isfinite(pred)
        )

        output[config] = {
            "gt": branch_gt,
            "pred": pred,
            "footprint": final_mask,
            "hidden": branch["hidden_any"],
            "raw_loss": branch["raw_loss_any"],
            "final_mask": final_mask,
            "core": branch["core"],
            "valid_gt_footprint": branch["valid_gt_footprint"],
            "centers": branch["centers"],
            "processed_tile_root": str(tile_roots[config]),
        }
        common &= final_mask

    output["common"] = common
    return output

def error_stats(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    error = pred[mask].astype(np.float64) - gt[mask].astype(np.float64)
    if error.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "p90": float("nan")}
    absolute = np.abs(error)
    return {
        "n": int(error.size),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "p90": float(np.percentile(absolute, 90)),
    }


def segment_metrics(segment: Segment, data: Mapping[str, Any]) -> Dict[str, Any]:
    common = data["common"]
    point_ids = [int(float(row["tile_id"])) for row in segment.reference_rows]
    row: Dict[str, Any] = {
        "preset": segment.preset,
        "river": segment.river,
        "river_label": segment.river_label,
        "line_id": segment.line_id,
        "segment_index": segment.segment_index,
        "segment_id": (
            f"{segment.preset}_L{sanitize(segment.line_id)}_"
            f"P{min(point_ids):06d}-{max(point_ids):06d}"
        ),
        "first_point_id": min(point_ids),
        "last_point_id": max(point_ids),
        "n_sampling_points": len(point_ids),
        "n_fourway_common_loss_pixels": int(common.sum()),
    }
    for config in CONFIG_ORDER:
        values = error_stats(data[config]["pred"], data[config]["gt"], common)
        row[f"{config}_mae_m"] = values["mae"]
        row[f"{config}_rmse_m"] = values["rmse"]
        row[f"{config}_bias_m"] = values["bias"]
        row[f"{config}_p90_abs_error_m"] = values["p90"]
        row[f"{config}_own_prediction_pixels"] = int(data[config]["footprint"].sum())
    row["relaxed_meter_advantage_over_relaxed_normalized_m"] = (
        row["relaxed_normalized_mae_m"] - row["relaxed_meter_mae_m"]
    )
    row["strict_meter_advantage_over_strict_normalized_m"] = (
        row["strict_normalized_mae_m"] - row["strict_meter_mae_m"]
    )
    row["relaxed_mask_advantage_for_meter_m"] = (
        row["strict_meter_mae_m"] - row["relaxed_meter_mae_m"]
    )
    row["relaxed_mask_advantage_for_normalized_m"] = (
        row["strict_normalized_mae_m"] - row["relaxed_normalized_mae_m"]
    )
    return row


def crop_slice(mask: np.ndarray, padding: int):
    rows, columns = np.where(mask)
    if rows.size == 0:
        return np.s_[:, :]
    return np.s_[
        max(0, int(rows.min()) - padding) : min(mask.shape[0], int(rows.max()) + padding + 1),
        max(0, int(columns.min()) - padding) : min(mask.shape[1], int(columns.max()) + padding + 1),
    ]


def add_scale_and_north(ax, width_pixels: int, resolution: float) -> None:
    length_m = max(10, int(round((width_pixels * resolution / 5.0) / 10.0) * 10))
    length_px = length_m / resolution
    x1 = 0.05
    x2 = min(0.45, x1 + length_px / max(width_pixels, 1))
    y = 0.93
    ax.plot([x1, x2], [y, y], transform=ax.transAxes, linewidth=3)
    ax.text((x1 + x2) / 2, y - 0.04, f"{length_m} m", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    ax.annotate(
        "N",
        xy=(0.94, 0.94),
        xytext=(0.94, 0.78),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        arrowprops={"arrowstyle": "-|>", "linewidth": 1.5},
    )


def finite_limits(
    arrays: Sequence[np.ndarray],
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    fallback: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[float, float]:
    values = [
        np.asarray(array, dtype=np.float64)[
            np.isfinite(np.asarray(array, dtype=np.float64))
        ]
        for array in arrays
    ]
    values = [array for array in values if array.size]
    if not values:
        return fallback

    combined = np.concatenate(values)
    lower = float(np.percentile(combined, lower_percentile))
    upper = float(np.percentile(combined, upper_percentile))

    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower = float(np.nanmin(combined))
        upper = float(np.nanmax(combined))
        if lower == upper:
            upper = lower + 1.0
    return lower, upper


def render_one(
    config: str,
    metrics: Mapping[str, Any],
    data: Mapping[str, Any],
    crop,
    output: Path,
    dpi: int,
) -> Dict[str, float]:
    """Render one configuration with its own independent display ranges.

    The second-row GT and prediction share one elevation range only inside this
    configuration's six-panel figure. No second-row range is shared with the
    strict/relaxed or normalized/meter figures.

    Signed-error limits are also calculated independently for this
    configuration and are symmetric around zero.
    """
    branch = data[config]
    gt = branch["gt"][crop]
    pred = branch["pred"][crop]
    valid_gt_footprint = branch["valid_gt_footprint"][crop].astype(bool)
    hidden_bool = branch["hidden"][crop].astype(bool)
    final_mask = branch.get("final_mask", branch["core"])[crop].astype(bool)
    hidden = np.where(valid_gt_footprint, hidden_bool.astype(float), np.nan)
    error = pred - gt

    row_slice, col_slice = crop
    row_offset = row_slice.start or 0
    col_offset = col_slice.start or 0
    centers = [
        (col - col_offset, row - row_offset)
        for col, row in branch["centers"]
        if (col_slice.start or 0) <= col < (col_slice.stop or gt.shape[1])
        and (row_slice.start or 0) <= row < (row_slice.stop or gt.shape[0])
    ]

    gt_final = np.where(final_mask, gt, np.nan)
    pred_final = np.where(final_mask, pred, np.nan)
    error_final = np.where(final_mask, error, np.nan)
    final_loss_mask = np.where(
        valid_gt_footprint,
        final_mask.astype(float),
        np.nan,
    )

    # First-row complete GT has its own range.
    full_gt_lower, full_gt_upper = finite_limits([gt])

    # Second-row GT and prediction are unified only within this configuration.
    second_lower, second_upper = finite_limits(
        [gt_final, pred_final],
        fallback=(full_gt_lower, full_gt_upper),
    )

    # Error scale is independent for this configuration.
    finite_error = error_final[np.isfinite(error_final)]
    if finite_error.size:
        error_limit = max(
            float(np.percentile(np.abs(finite_error), 98)),
            0.05,
        )
    else:
        error_limit = 0.5

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Row 1.
    image = axes[0, 0].imshow(
        gt,
        vmin=full_gt_lower,
        vmax=full_gt_upper,
    )
    if centers:
        axes[0, 0].scatter(
            [item[0] for item in centers],
            [item[1] for item in centers],
            s=18,
            marker="x",
            label="Sampling centers",
        )
        axes[0, 0].legend(loc="lower right", fontsize=7)
    axes[0, 0].set_title("Ground truth with sampling centers")
    axes[0, 0].axis("off")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.03)

    image = axes[0, 1].imshow(hidden, vmin=0, vmax=1)
    axes[0, 1].set_title("Patch-processed Hidden Mask (0/1)")
    axes[0, 1].axis("off")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.03)

    image = axes[0, 2].imshow(final_loss_mask, vmin=0, vmax=1)
    axes[0, 2].set_title("Final prediction / loss mask (0/1)")
    axes[0, 2].axis("off")
    fig.colorbar(image, ax=axes[0, 2], fraction=0.046, pad=0.03)

    # Row 2: ranges belong only to this configuration.
    panels = (
        (
            axes[1, 0],
            gt_final,
            "GT inside final prediction / loss mask",
            second_lower,
            second_upper,
            None,
        ),
        (
            axes[1, 1],
            pred_final,
            "Prediction inside final prediction / loss mask",
            second_lower,
            second_upper,
            None,
        ),
        (
            axes[1, 2],
            error_final,
            "Signed error inside final prediction / loss mask",
            -error_limit,
            error_limit,
            "coolwarm",
        ),
    )
    for ax, array, title, lower, upper, cmap in panels:
        image = ax.imshow(array, vmin=lower, vmax=upper, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    add_scale_and_north(axes[0, 0], gt.shape[1], data["resolution"])
    mae = float(metrics[f"{config}_mae_m"])
    rmse = float(metrics[f"{config}_rmse_m"])
    fig.suptitle(
        f"{metrics['river_label']} | {metrics['segment_id']}\n"
        f"{CONFIG_LABELS[config]} | MAE={mae:.3f} m, RMSE={rmse:.3f} m\n"
        "Second-row display ranges are calculated independently for this "
        "configuration; GT and prediction share a range only within this figure.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)

    return {
        "full_gt_vmin_m": full_gt_lower,
        "full_gt_vmax_m": full_gt_upper,
        "second_row_elevation_vmin_m": second_lower,
        "second_row_elevation_vmax_m": second_upper,
        "second_row_error_vmin_m": -error_limit,
        "second_row_error_vmax_m": error_limit,
    }

RENDER_VERSION = "H046_v10_processed_E001_E001c_aligned"


def expected_reach_paths(target: Path) -> Dict[str, str]:
    outputs = {
        f"{config}_6panel_png": str(target / f"{config}_6panel.png")
        for config in CONFIG_ORDER
    }
    outputs["four_configuration_overview_png"] = str(
        target / "four_configuration_prediction_overview.png"
    )
    outputs["display_ranges_json"] = str(target / "display_ranges.json")
    outputs["render_version_json"] = str(target / "render_version.json")
    return outputs


def six_panel_outputs_complete(target: Path) -> bool:
    paths = expected_reach_paths(target)
    version_path = Path(paths["render_version_json"])
    if not version_path.is_file():
        return False
    try:
        version_payload = json.loads(version_path.read_text())
    except Exception:
        return False
    if version_payload.get("render_version") != RENDER_VERSION:
        return False

    return all(
        Path(paths[f"{config}_6panel_png"]).is_file()
        for config in CONFIG_ORDER
    )


def safe_metric_text(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "NA"
    return f"{number:.3f}" if np.isfinite(number) else "NA"


def render_reach(
    metrics: Mapping[str, Any],
    data: Mapping[str, Any],
    target: Path,
    padding: int,
    dpi: int,
    include_overview: bool = False,
    resume: bool = True,
) -> Dict[str, str]:
    target.mkdir(parents=True, exist_ok=True)
    expected = expected_reach_paths(target)

    if resume and six_panel_outputs_complete(target):
        outputs = {
            f"{config}_6panel_png": expected[f"{config}_6panel_png"]
            for config in CONFIG_ORDER
        }
        overview_path = Path(expected["four_configuration_overview_png"])
        outputs["four_configuration_overview_png"] = (
            str(overview_path) if overview_path.is_file() else ""
        )
        outputs["display_ranges_json"] = expected["display_ranges_json"]
        outputs["render_version_json"] = expected["render_version_json"]
        return outputs

    # Include every finite processed GT pixel from both strict and relaxed
    # branches, plus the final prediction footprints.
    first_shape = data[CONFIG_ORDER[0]]["gt"].shape
    union = np.zeros(first_shape, dtype=bool)
    for config in CONFIG_ORDER:
        union |= (
            np.isfinite(data[config]["gt"])
            | data[config]["valid_gt_footprint"]
            | data[config]["core"]
            | data[config]["footprint"]
        )
    crop = crop_slice(union, padding)

    outputs: Dict[str, str] = {}
    display_ranges: Dict[str, Dict[str, float]] = {}

    for config in CONFIG_ORDER:
        path = target / f"{config}_6panel.png"
        display_ranges[config] = render_one(
            config,
            metrics,
            data,
            crop,
            path,
            dpi,
        )
        outputs[f"{config}_6panel_png"] = str(path)

    ranges_path = Path(expected["display_ranges_json"])
    ranges_path.write_text(
        json.dumps(
            {
                "render_version": RENDER_VERSION,
                "range_policy": (
                    "Each configuration independently calculates its second-row "
                    "GT/prediction elevation range and symmetric error range. "
                    "GT and masks are read from the same processed E001/E001c "
                    "tile branch used by that mask regime."
                ),
                "configurations": display_ranges,
            },
            indent=2,
        )
    )
    outputs["display_ranges_json"] = str(ranges_path)

    overview = target / "four_configuration_prediction_overview.png"
    if include_overview:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for ax, config in zip(axes.flat, CONFIG_ORDER):
            pred = data[config]["pred"][crop]
            final_mask = data[config].get(
                "final_mask",
                data[config]["core"],
            )[crop].astype(bool)
            pred_final = np.where(final_mask, pred, np.nan)
            lower = display_ranges[config]["second_row_elevation_vmin_m"]
            upper = display_ranges[config]["second_row_elevation_vmax_m"]
            image = ax.imshow(
                pred_final,
                vmin=lower,
                vmax=upper,
            )
            ax.set_title(
                f"{CONFIG_LABELS[config]}\n"
                f"MAE={safe_metric_text(metrics[f'{config}_mae_m'])} m | "
                f"own range={lower:.2f}–{upper:.2f} m"
            )
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        fig.suptitle(
            f"{metrics['river_label']} | {metrics['segment_id']} | "
            "prediction overview\n"
            "Each configuration uses its own final-mask elevation range."
        )
        fig.tight_layout()
        fig.savefig(overview, dpi=dpi)
        plt.close(fig)
        outputs["four_configuration_overview_png"] = str(overview)
    else:
        outputs["four_configuration_overview_png"] = (
            str(overview) if overview.is_file() else ""
        )

    version_path = Path(expected["render_version_json"])
    version_path.write_text(
        json.dumps(
            {
                "render_version": RENDER_VERSION,
                "second_row_range_policy": (
                    "configuration-specific; not shared across the four figures"
                ),
            },
            indent=2,
        )
    )
    outputs["render_version_json"] = str(version_path)
    return outputs

def create_relative_selection_link(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    relative = os.path.relpath(source, destination.parent)
    destination.symlink_to(relative, target_is_directory=True)
    return str(destination)


def build_all_reaches_gallery(
    records: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    selected_map: Dict[str, List[str]] = {}
    for row in selected_rows:
        selected_map.setdefault(str(row["segment_id"]), []).append(
            f"{row['selection_category']} rank {row['selection_rank']}"
        )

    def relative(path_text: str) -> str:
        if not path_text:
            return ""
        return os.path.relpath(Path(path_text), output).replace(os.sep, "/")

    rows_html: List[str] = []
    for row in sorted(
        records,
        key=lambda item: (
            str(item.get("preset", "")),
            str(item.get("line_id", "")),
            int(item.get("first_point_id", 0)),
        ),
    ):
        segment_id = str(row["segment_id"])
        selection = "; ".join(selected_map.get(segment_id, []))
        links = []
        for config in CONFIG_ORDER:
            path_text = str(row.get(f"{config}_6panel_png", "") or "")
            if path_text and Path(path_text).is_file():
                links.append(
                    f'<a href="{html.escape(relative(path_text))}">'
                    f'{html.escape(config)}</a>'
                )
        common_pixels = int(row.get("n_fourway_common_loss_pixels", 0) or 0)
        relaxed_meter_mae = safe_metric_text(
            row.get("relaxed_meter_mae_m", float("nan"))
        )
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('preset', '')))}</td>"
            f"<td>{html.escape(str(row.get('line_id', '')))}</td>"
            f"<td>{html.escape(segment_id)}</td>"
            f"<td>{common_pixels}</td>"
            f"<td>{relaxed_meter_mae}</td>"
            f"<td>{html.escape(selection)}</td>"
            f"<td>{' | '.join(links)}</td>"
            "</tr>"
        )

    content = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>H046 all continuous reaches</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
input {{ padding: 8px; width: min(520px, 90%); margin-bottom: 12px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ccc; padding: 6px; text-align: left; }}
th {{ position: sticky; top: 0; background: #eee; }}
tr:nth-child(even) {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>H046 all continuous reaches</h1>
<p>
Every successfully assembled continuous reach is listed below. Each reach has
four separate six-panel figures. Selection labels identify the automatically
picked best, median, worst, meter-advantage, and relaxed-mask-advantage cases.
</p>
<input id="filter" type="search" placeholder="Filter preset, LineID, segment ID, or selection category">
<table id="reachTable">
<thead>
<tr>
<th>Preset</th><th>LineID</th><th>Segment</th>
<th>Four-way common pixels</th><th>Relaxed-meter MAE (m)</th>
<th>Selected category</th><th>Figures</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
<script>
const input = document.getElementById('filter');
const rows = [...document.querySelectorAll('#reachTable tbody tr')];
input.addEventListener('input', () => {{
  const query = input.value.toLowerCase();
  rows.forEach(row => {{
    row.style.display = row.textContent.toLowerCase().includes(query) ? '' : 'none';
  }});
}});
</script>
</body>
</html>
"""
    (output / "H046_all_reaches_gallery.html").write_text(
        content,
        encoding="utf-8",
    )

def choose_indices(frame: pd.DataFrame, count: int, mode: str) -> List[int]:
    if frame.empty or count <= 0:
        return []
    if mode == "best":
        return frame.nsmallest(count, "relaxed_meter_mae_m").index.tolist()
    if mode == "median":
        median = float(frame["relaxed_meter_mae_m"].median())
        return (frame["relaxed_meter_mae_m"] - median).abs().nsmallest(count).index.tolist()
    if mode == "worst":
        return frame.nlargest(count, "relaxed_meter_mae_m").index.tolist()
    if mode == "meter_advantage":
        return frame.nlargest(count, "relaxed_meter_advantage_over_relaxed_normalized_m").index.tolist()
    if mode == "relaxed_advantage":
        return frame.nlargest(count, "relaxed_mask_advantage_for_meter_m").index.tolist()
    raise ValueError(mode)


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    strict_tile_root = resolve_processed_tile_root(args.strict_tile_base)
    relaxed_tile_root = resolve_processed_tile_root(args.relax_tile_base)
    tile_roots: Dict[str, Path] = {
        "strict_normalized": strict_tile_root,
        "strict_meter": strict_tile_root,
        "relaxed_normalized": relaxed_tile_root,
        "relaxed_meter": relaxed_tile_root,
    }

    print(f"[TILE-SOURCE] strict={strict_tile_root}")
    print(f"[TILE-SOURCE] relaxed={relaxed_tile_root}")

    all_metrics: List[Dict[str, Any]] = []
    lookup: Dict[
        str,
        Tuple[Segment, Mapping[str, Mapping[str, Dict[str, str]]]],
    ] = {}
    manifest_path_audit: List[Dict[str, str]] = []
    skipped_reaches: List[Dict[str, str]] = []
    visualization_failures: List[Dict[str, str]] = []
    rendered_per_preset: Dict[str, int] = {}

    for case in CASES:
        directories = branch_dirs(args, case)
        rows_by_config: Dict[str, List[Dict[str, str]]] = {}
        maps: Dict[str, Dict[str, Dict[str, str]]] = {}

        for config, directory in directories.items():
            manifest, summary = locate_manifest(directory)
            if config.startswith("relaxed_"):
                validate_relaxed_summary(summary, config)
            rows, path_audit = read_manifest_with_rebased_paths(
                manifest,
                directory,
                config,
            )
            rows_by_config[config] = rows
            maps[config] = {row["key"]: row for row in rows}
            manifest_path_audit.extend(path_audit)
            print(f"[{case['preset']}] {config}: {len(rows)} tiles")

        line_mapping, line_source = load_line_mapping(
            args.relax_tile_base,
            case["river"],
        )
        segments = build_segments(
            case["preset"],
            case["river"],
            case["label"],
            rows_by_config["relaxed_meter"],
            maps,
            line_mapping,
            args.segment_size,
            args.segment_stride,
            args.min_points,
        )
        print(
            f"[{case['preset']}] candidate reaches={len(segments)} "
            f"line_source={line_source}"
        )

        rendered_per_preset[case["preset"]] = 0
        for index, segment in enumerate(segments, start=1):
            if (
                args.max_render_reaches > 0
                and rendered_per_preset[case["preset"]]
                >= args.max_render_reaches
            ):
                print(
                    f"[DEBUG-LIMIT] {case['preset']} reached "
                    f"max_render_reaches={args.max_render_reaches}"
                )
                break

            try:
                data = segment_data(
                    segment,
                    maps,
                    tile_roots,
                    args.max_segment_dense_pixels,
                )
                metrics = segment_metrics(segment, data)
            except Exception as exc:
                message = str(exc)
                print(f"[SKIP] {case['preset']} reach {index}: {message}")
                skipped_reaches.append(
                    {
                        "preset": case["preset"],
                        "river": case["river"],
                        "reach_index": str(index),
                        "error_type": type(exc).__name__,
                        "error": message,
                    }
                )
                continue

            metrics["line_mapping_source"] = line_source
            metrics["visualization_status"] = "not_requested"
            metrics["all_reach_dir"] = ""

            lookup[metrics["segment_id"]] = (segment, maps)

            if args.render_all_reaches:
                target = (
                    output
                    / "all_reaches"
                    / case["preset"]
                    / f"Line_{sanitize(segment.line_id)}"
                    / metrics["segment_id"]
                )
                try:
                    paths = render_reach(
                        metrics,
                        data,
                        target,
                        args.crop_padding,
                        args.dpi,
                        include_overview=args.render_overview_for_all,
                        resume=args.resume_visuals,
                    )
                    metrics.update(paths)
                    metrics["all_reach_dir"] = str(target)
                    metrics["visualization_status"] = "complete"
                    (target / "metrics.json").write_text(
                        json.dumps(metrics, indent=2)
                    )
                    rendered_per_preset[case["preset"]] += 1
                except Exception as exc:
                    message = str(exc)
                    metrics["visualization_status"] = "failed"
                    visualization_failures.append(
                        {
                            "preset": case["preset"],
                            "river": case["river"],
                            "segment_id": metrics["segment_id"],
                            "error_type": type(exc).__name__,
                            "error": message,
                        }
                    )
                    print(
                        f"[VIS-FAIL] {case['preset']} "
                        f"{metrics['segment_id']}: {message}"
                    )

            all_metrics.append(metrics)

            if (
                index == 1
                or index == len(segments)
                or index % max(args.render_progress_every, 1) == 0
            ):
                print(
                    f"  [{case['preset']}] {index}/{len(segments)} "
                    f"common={metrics['n_fourway_common_loss_pixels']} "
                    f"visual={metrics['visualization_status']}"
                )

    write_csv(
        output / "H046_manifest_path_rebase_audit.csv",
        manifest_path_audit,
    )
    write_csv(output / "H046_skipped_reaches.csv", skipped_reaches)
    write_csv(
        output / "H046_visualization_failures.csv",
        visualization_failures,
    )

    frame = pd.DataFrame(all_metrics)
    if frame.empty:
        first_errors = "\n".join(
            f"  {row['preset']} reach {row['reach_index']}: "
            f"{row['error_type']}: {row['error']}"
            for row in skipped_reaches[:10]
        )
        raise RuntimeError(
            "No reach metrics produced. See H046_skipped_reaches.csv and "
            "H046_manifest_path_rebase_audit.csv.\n"
            + first_errors
        )

    eligible = frame[
        frame["n_fourway_common_loss_pixels"] >= args.min_common_pixels
    ].copy()
    if eligible.empty:
        raise RuntimeError(
            f"No reach has >= {args.min_common_pixels} four-way common "
            "loss pixels"
        )

    selected_rows: List[Dict[str, Any]] = []
    for case in CASES:
        river_frame = eligible[eligible["preset"] == case["preset"]]
        categories = {
            "best_relaxed_meter": choose_indices(
                river_frame,
                args.n_best,
                "best",
            ),
            "median_relaxed_meter": choose_indices(
                river_frame,
                args.n_median,
                "median",
            ),
            "worst_relaxed_meter": choose_indices(
                river_frame,
                args.n_worst,
                "worst",
            ),
            "largest_meter_objective_advantage": choose_indices(
                river_frame,
                args.n_meter_advantage,
                "meter_advantage",
            ),
            "largest_relaxed_mask_advantage": choose_indices(
                river_frame,
                args.n_relaxed_advantage,
                "relaxed_advantage",
            ),
        }

        for category, indices in categories.items():
            for rank, frame_index in enumerate(indices, start=1):
                metrics = frame.loc[frame_index].to_dict()
                segment_id = str(metrics["segment_id"])

                # All-reach rendering is the primary archive. If disabled or
                # one visualization failed, render this selected reach now.
                all_target = Path(str(metrics.get("all_reach_dir", "") or ""))
                if (
                    not all_target
                    or not six_panel_outputs_complete(all_target)
                ):
                    segment, maps = lookup[segment_id]
                    data = segment_data(
                        segment,
                        maps,
                        tile_roots,
                        args.max_segment_dense_pixels,
                    )
                    all_target = (
                        output
                        / "all_reaches"
                        / case["preset"]
                        / f"Line_{sanitize(segment.line_id)}"
                        / segment_id
                    )
                    paths = render_reach(
                        metrics,
                        data,
                        all_target,
                        args.crop_padding,
                        args.dpi,
                        include_overview=True,
                        resume=args.resume_visuals,
                    )
                    metrics.update(paths)
                    metrics["all_reach_dir"] = str(all_target)
                    metrics["visualization_status"] = "complete"
                    (all_target / "metrics.json").write_text(
                        json.dumps(metrics, indent=2)
                    )
                else:
                    paths = expected_reach_paths(all_target)
                    metrics.update(
                        {
                            f"{config}_6panel_png": paths[
                                f"{config}_6panel_png"
                            ]
                            for config in CONFIG_ORDER
                        }
                    )
                    overview_path = Path(
                        paths["four_configuration_overview_png"]
                    )
                    metrics["four_configuration_overview_png"] = (
                        str(overview_path)
                        if overview_path.is_file()
                        else ""
                    )

                link_path = (
                    output
                    / "selected_reaches"
                    / case["preset"]
                    / category
                    / f"rank{rank:02d}_{segment_id}"
                )
                selected_link = create_relative_selection_link(
                    all_target,
                    link_path,
                )

                record = {
                    **metrics,
                    "selection_category": category,
                    "selection_rank": rank,
                    "selection_link_dir": selected_link,
                }
                selected_rows.append(record)
                print(
                    f"[SELECT] {case['preset']} {category} rank={rank}: "
                    f"{segment_id}"
                )

    # Update frame with final figure paths, including selected-on-demand renders.
    selected_updates = {
        str(row["segment_id"]): {
            "all_reach_dir": row.get("all_reach_dir", ""),
            "visualization_status": row.get("visualization_status", "complete"),
            **{
                f"{config}_6panel_png": row.get(
                    f"{config}_6panel_png",
                    "",
                )
                for config in CONFIG_ORDER
            },
            "four_configuration_overview_png": row.get(
                "four_configuration_overview_png",
                "",
            ),
        }
        for row in selected_rows
    }
    final_records: List[Dict[str, Any]] = []
    for row in all_metrics:
        update = selected_updates.get(str(row["segment_id"]))
        final_records.append({**row, **update} if update else row)

    final_frame = pd.DataFrame(final_records)
    final_frame.to_csv(
        output / "H046_all_reach_metrics.csv",
        index=False,
    )
    for case in CASES:
        final_frame[final_frame["preset"] == case["preset"]].sort_values(
            ["line_id", "first_point_id"]
        ).to_csv(
            output / f"H046_all_reaches_{case['preset']}.csv",
            index=False,
        )

    write_csv(
        output / "H046_selected_reaches.csv",
        selected_rows,
    )
    build_all_reaches_gallery(
        final_records,
        selected_rows,
        output,
    )

    # Ordered reach MAE profiles for all four configurations.
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    for ax, case in zip(axes, CASES):
        river_frame = eligible[
            eligible["preset"] == case["preset"]
        ].sort_values(["line_id", "first_point_id"])
        x = np.arange(len(river_frame))
        for config in CONFIG_ORDER:
            ax.plot(
                x,
                river_frame[f"{config}_mae_m"],
                label=CONFIG_LABELS[config],
            )
        ax.set_title(case["label"])
        ax.set_xlabel("Ordered continuous reach index")
        ax.set_ylabel("MAE on four-way common loss pixels (m)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Continuous-reach recovery performance across four configurations"
    )
    fig.tight_layout()
    fig.savefig(
        output / "H046_reach_mae_profiles.png",
        dpi=args.dpi,
    )
    plt.close(fig)

    n_visualized = int(
        (
            final_frame.get(
                "visualization_status",
                pd.Series(index=final_frame.index, dtype=str),
            )
            == "complete"
        ).sum()
    )

    summary = {
        "strict_processed_tile_root": str(strict_tile_root),
        "relaxed_processed_tile_root": str(relaxed_tile_root),
        "tile_source_policy": (
            "Strict GT/Hidden/Loss/Core use processed E001 tiles; relaxed "
            "GT/Hidden/Loss/Core use processed E001c AnyVisiblePatch tiles. "
            "Predictions remain the formal overlap-averaged full-river outputs."
        ),
        "segment_size": args.segment_size,
        "segment_stride": args.segment_stride,
        "min_common_pixels_for_selection": args.min_common_pixels,
        "render_all_reaches": args.render_all_reaches,
        "resume_visuals": args.resume_visuals,
        "render_overview_for_all": args.render_overview_for_all,
        "metric_footprint": (
            "Core_Loss_Mask_Pixel AND valid GT AND valid predictions from "
            "all four configurations"
        ),
        "all_reach_visualization_policy": (
            "Every successfully assembled continuous reach receives four "
            "separate six-panel figures. The minimum common-pixel threshold "
            "is used only for automatic best/median/worst selection."
        ),
        "six_panels": [
            "Ground truth with sampling centers",
            "Patch-processed Hidden Mask (0/1; blank outside processed valid GT)",
            "Final prediction/loss mask (0/1; blank outside processed valid GT)",
            "GT inside final prediction/loss mask",
            "Prediction inside final prediction/loss mask",
            "Signed error inside final prediction/loss mask",
        ],
        "configurations": {
            config: CONFIG_LABELS[config]
            for config in CONFIG_ORDER
        },
        "manifest_path_rebase_count": len(manifest_path_audit),
        "skipped_reach_count": len(skipped_reaches),
        "visualization_failure_count": len(visualization_failures),
        "manifest_path_rebase_audit": (
            "H046_manifest_path_rebase_audit.csv"
        ),
        "skipped_reach_audit": "H046_skipped_reaches.csv",
        "visualization_failure_audit": (
            "H046_visualization_failures.csv"
        ),
        "all_reach_metrics": "H046_all_reach_metrics.csv",
        "all_reaches_gallery": "H046_all_reaches_gallery.html",
        "selected_reaches": "H046_selected_reaches.csv",
        "n_all_reaches_with_metrics": int(len(final_frame)),
        "n_all_reaches_visualized": n_visualized,
        "n_selection_eligible_reaches": int(len(eligible)),
        "n_selected_category_rows": int(len(selected_rows)),
    }
    (output / "H046_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    print("[DONE]", output)


if __name__ == "__main__":
    main()
