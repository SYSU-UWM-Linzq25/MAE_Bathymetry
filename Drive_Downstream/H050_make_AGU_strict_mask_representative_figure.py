#!/usr/bin/env python3
"""Build the single AGU strict-mask spatial result figure.

This script is self-contained at the project level: it does not depend on an
H046 results CSV from the other mask project. Representative continuous
reaches are selected directly from this project's formal prediction manifests
by calculating the local MAE for every eligible continuous reach and choosing
the reach closest to the river median.

Scientific configuration
------------------------
* Mask regime: strict.
* Objective/model: meter-domain objective.
* Prediction source: formal F060 overlap-averaged full-river predictions.
* Full-river metrics: F062 exact unique-geospatial-pixel summaries.
* Representative reach: closest local own-footprint MAE to the river median.
* GT and masks: the processed tile branch for this mask regime.
* Display footprint:

      Core_Loss_Mask_Pixel AND valid processed GT AND valid formal prediction

* GT and prediction share one elevation scale within each river row.
* Absolute error uses one shared 0--2 m scale across all three rivers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd


CASES: Tuple[Dict[str, str], ...] = (
    {
        "preset": "CA",
        "label": "CA Klamath",
        "river": "CA_KlamathRiver_TopoBathy_2018_D18",
        "experiment": "holdout_CA_D003MeterMAE_BaselineEval_D001NoDataSafe",
    },
    {
        "preset": "CO",
        "label": "CO Upper Colorado",
        "river": "CO_UpperColorado_Topobathy_1_2020",
        "experiment": "holdout_CO_D003MeterMAE_BaselineEval_D001NoDataSafe",
    },
    {
        "preset": "Santiam",
        "label": "OR Santiam",
        "river": "OR_SantiamRiverTB_Topobathy_1_D23",
        "experiment": "holdout_Santiam_D003MeterMAE_BaselineEval_D001NoDataSafe",
    },
)


@dataclass(frozen=True)
class ReachSelection:
    preset: str
    river: str
    river_label: str
    segment_id: str
    line_id: str
    first_point_id: int
    last_point_id: int
    n_sampling_points: int
    selection_mae_m: float
    river_median_mae_m: float
    common_pixels: int


@dataclass(frozen=True)
class ReachData:
    gt: np.ndarray
    prediction: np.ndarray
    final_mask: np.ndarray
    centers: Tuple[Tuple[float, float], ...]
    resolution_m: float
    local_mae_m: float
    local_rmse_m: float
    local_bias_m: float
    n_final_pixels: int


def parse_args() -> argparse.Namespace:
    root = Path("/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography")
    project_root = root / "Downstream_Task_Bathy"
    results_root = project_root / "Results"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Create a reproducible 3-river x 3-panel AGU strict-mask "
            "figure using the formal meter-objective full-river predictions."
        ),
    )
    parser.add_argument(
        "--h046_script",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "H050_AGU_geospatial_utils.py"
        ),
        help="Bundled H046 utility script used for GeoTIFF and manifest I/O.",
    )
    parser.add_argument(
        "--prediction_root",
        type=Path,
        default=results_root / "FullRiver_Predictions_F060_TileAvgVRT_D003MeterMAE_BaselineEval_D001NoDataSafe",
        help="Formal strict meter-objective prediction root.",
    )
    parser.add_argument(
        "--error_root",
        type=Path,
        default=results_root / "FullRiver_GT_Error_F062_UniquePixel_D003MeterMAE_BaselineEval_D001NoDataSafe",
        help="Formal strict exact unique-pixel error root.",
    )
    parser.add_argument(
        "--tile_base",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "Tiles_for_MAE_FullRiver_E001"
        ),
        help="Processed strict full-river tile branch.",
    )
    parser.add_argument(
        "--manual_selection_csv",
        type=Path,
        default=None,
        help=(
            "Optional selection CSV with one row per preset and columns "
            "preset, line_id, first_point_id, last_point_id. When absent, "
            "all eligible reaches are evaluated and the median-MAE reach is "
            "selected automatically."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--segment_size", type=int, default=10)
    parser.add_argument("--segment_stride", type=int, default=10)
    parser.add_argument("--min_segment_points", type=int, default=5)
    parser.add_argument(
        "--required_sampling_points",
        type=int,
        default=10,
        help="Prefer reaches containing exactly this many sampling points.",
    )
    parser.add_argument(
        "--min_final_pixels",
        type=int,
        default=1000,
        help="Minimum valid final prediction pixels for median selection.",
    )
    parser.add_argument(
        "--candidate_metrics_csv",
        type=Path,
        default=None,
        help=(
            "Optional precomputed candidate-reach metrics CSV. If absent, "
            "the script evaluates all candidate reaches and writes one under "
            "the output directory."
        ),
    )
    parser.add_argument("--crop_padding", type=int, default=24)
    parser.add_argument(
        "--elevation_percentiles",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument("--absolute_error_max_m", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--figure_width_in", type=float, default=13.0)
    parser.add_argument("--title", default="Representative strict-mask full-river recovery results")
    parser.add_argument("--subtitle", default="Entire river withheld from training; target-channel bathymetry fully hidden.")
    parser.add_argument("--output_stem", default="AGU_strict_mask_representative_reaches")
    return parser.parse_args()

def import_h046(path: Path) -> ModuleType:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"AGU helper not found: {path}\n"
            "Copy H050 next to H046_visualize_local_reaches_6panel.py or "
            "pass --h046_script explicitly."
        )
    spec = importlib.util.spec_from_file_location("h046_agu_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import AGU helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = (
        "resolve_river_dir",
        "locate_manifest",
        "read_manifest_with_rebased_paths",
        "resolve_processed_tile_root",
        "processed_tile_paths",
        "read_tif",
        "valid_gt",
        "valid_pred",
        "affine_compatible",
        "divide",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "The AGU helper is missing required functions: " + ", ".join(missing)
        )
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def discover_line_mapping(
    tile_base: Path,
    river: str,
) -> Tuple[Dict[int, str], str]:
    """Find an E001/E001c candidate QA file if available.

    When no QA file exists, consecutive tile IDs are used to preserve
    continuity and gaps start a new sequence.
    """
    candidates = sorted(
        path
        for path in tile_base.rglob(f"*candidate_QA_1m_{river}.csv")
        if river.lower() in str(path).lower()
    )
    if not candidates:
        return {}, "consecutive_tile_id_fallback"

    # Prefer a file whose prefix matches this mask branch.
    preferred_token = "e001c" if "strict" == "relaxed" else "e001"
    preferred = [
        path for path in candidates
        if preferred_token in path.name.lower()
    ]
    qa = preferred[0] if len(preferred) == 1 else candidates[0]

    frame = pd.read_csv(qa)
    columns = {column.lower(): column for column in frame.columns}
    point_column = columns.get("point_id")
    line_column = columns.get("line_id")
    kept_column = columns.get("kept")
    if point_column is None or line_column is None:
        return {}, f"consecutive_tile_id_fallback_missing_columns_{qa.name}"
    if kept_column is not None:
        frame = frame[pd.to_numeric(frame[kept_column], errors="coerce") > 0]

    mapping: Dict[int, str] = {}
    for point_id, line_id in zip(
        pd.to_numeric(frame[point_column], errors="coerce"),
        frame[line_column],
    ):
        if np.isfinite(point_id):
            mapping[int(round(float(point_id)))] = str(line_id)
    return mapping, str(qa)


def candidate_reach_rows(
    rows: Sequence[Dict[str, str]],
    line_mapping: Mapping[int, str],
    segment_size: int,
    segment_stride: int,
    min_segment_points: int,
) -> List[Tuple[str, Tuple[Dict[str, str], ...]]]:
    parsed: List[Tuple[int, Dict[str, str]]] = []
    for row in rows:
        try:
            point_id = int(round(float(row["tile_id"])))
        except Exception:
            continue
        parsed.append((point_id, row))
    parsed.sort(key=lambda item: item[0])

    groups: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
    if line_mapping:
        for point_id, row in parsed:
            groups.setdefault(
                str(line_mapping.get(point_id, "UNMAPPED")),
                [],
            ).append((point_id, row))
    else:
        sequence_index = 0
        previous: Optional[int] = None
        for point_id, row in parsed:
            if previous is None or point_id != previous + 1:
                sequence_index += 1
            groups.setdefault(f"SEQUENCE_{sequence_index:05d}", []).append(
                (point_id, row)
            )
            previous = point_id

    candidates: List[Tuple[str, Tuple[Dict[str, str], ...]]] = []
    for line_id, items in sorted(groups.items()):
        items.sort(key=lambda item: item[0])
        for start_index in range(0, len(items), int(segment_stride)):
            chunk = items[start_index : start_index + int(segment_size)]
            if len(chunk) < int(min_segment_points):
                continue
            # Even inside a QA line, protect against point-ID gaps.
            ids = [item[0] for item in chunk]
            if any(b != a + 1 for a, b in zip(ids[:-1], ids[1:])):
                continue
            candidates.append(
                (line_id, tuple(item[1] for item in chunk))
            )
    return candidates


def evaluate_candidate_reaches(
    h046: ModuleType,
    rows: Sequence[Dict[str, str]],
    tile_root: Path,
    tile_base: Path,
    case: Mapping[str, str],
    segment_size: int,
    segment_stride: int,
    min_segment_points: int,
) -> List[Dict[str, Any]]:
    line_mapping, line_source = discover_line_mapping(
        tile_base,
        case["river"],
    )
    candidates = candidate_reach_rows(
        rows,
        line_mapping,
        segment_size,
        segment_stride,
        min_segment_points,
    )
    metrics: List[Dict[str, Any]] = []
    for index, (line_id, chunk) in enumerate(candidates, start=1):
        try:
            data = assemble_reach(h046, chunk, tile_root)
        except Exception as exc:
            print(
                f"[CANDIDATE-SKIP] {case['preset']} {index}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        first = int(round(float(chunk[0]["tile_id"])))
        last = int(round(float(chunk[-1]["tile_id"])))
        metrics.append(
            {
                "preset": case["preset"],
                "river": case["river"],
                "river_label": case["label"],
                "line_id": str(line_id),
                "first_point_id": first,
                "last_point_id": last,
                "n_sampling_points": len(chunk),
                "segment_id": (
                    f"{case['preset']}_L{line_id}_"
                    f"P{first:06d}-{last:06d}"
                ),
                "local_mae_m": data.local_mae_m,
                "local_rmse_m": data.local_rmse_m,
                "local_bias_m": data.local_bias_m,
                "n_final_pixels": data.n_final_pixels,
                "line_mapping_source": line_source,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(candidates):
            print(
                f"[CANDIDATE] {case['preset']} {index}/{len(candidates)} "
                f"MAE={data.local_mae_m:.4f} m"
            )
    return metrics


def select_reaches_from_candidate_metrics(
    frame: pd.DataFrame,
    min_final_pixels: int,
    required_sampling_points: int,
) -> Dict[str, ReachSelection]:
    required = {
        "preset",
        "river",
        "river_label",
        "segment_id",
        "line_id",
        "first_point_id",
        "last_point_id",
        "n_sampling_points",
        "local_mae_m",
        "n_final_pixels",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Candidate metrics are missing columns: {missing}"
        )

    result: Dict[str, ReachSelection] = {}
    for case in CASES:
        subset = frame[frame["preset"].astype(str) == case["preset"]].copy()
        subset["local_mae_m"] = pd.to_numeric(
            subset["local_mae_m"], errors="coerce"
        )
        subset["n_final_pixels"] = pd.to_numeric(
            subset["n_final_pixels"], errors="coerce"
        )
        subset["n_sampling_points"] = pd.to_numeric(
            subset["n_sampling_points"], errors="coerce"
        )
        subset = subset[
            np.isfinite(subset["local_mae_m"])
            & (subset["n_final_pixels"] >= int(min_final_pixels))
        ]
        preferred = subset[
            subset["n_sampling_points"] == int(required_sampling_points)
        ]
        if not preferred.empty:
            subset = preferred
        if subset.empty:
            raise RuntimeError(
                f"No eligible representative reach for {case['preset']}."
            )

        median = float(subset["local_mae_m"].median())
        subset = subset.assign(
            distance_to_median=np.abs(subset["local_mae_m"] - median)
        ).sort_values(
            ["distance_to_median", "line_id", "first_point_id"],
            kind="mergesort",
        )
        row = subset.iloc[0]
        result[case["preset"]] = ReachSelection(
            preset=case["preset"],
            river=str(row["river"]),
            river_label=str(row["river_label"]),
            segment_id=str(row["segment_id"]),
            line_id=str(row["line_id"]),
            first_point_id=int(round(float(row["first_point_id"]))),
            last_point_id=int(round(float(row["last_point_id"]))),
            n_sampling_points=int(round(float(row["n_sampling_points"]))),
            selection_mae_m=float(row["local_mae_m"]),
            river_median_mae_m=median,
            common_pixels=int(round(float(row["n_final_pixels"]))),
        )
    return result

def select_reaches_manually(path: Path) -> Dict[str, ReachSelection]:
    frame = pd.read_csv(path)
    required = {"preset", "line_id", "first_point_id", "last_point_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Manual selection CSV is missing columns {missing}: {path}"
        )

    result: Dict[str, ReachSelection] = {}
    for case in CASES:
        rows = frame[frame["preset"].astype(str) == case["preset"]]
        if len(rows) != 1:
            raise RuntimeError(
                f"Manual selection must contain exactly one row for "
                f"preset={case['preset']}; found {len(rows)}."
            )
        row = rows.iloc[0]
        first = int(round(float(row["first_point_id"])))
        last = int(round(float(row["last_point_id"])))
        result[case["preset"]] = ReachSelection(
            preset=case["preset"],
            river=case["river"],
            river_label=case["label"],
            segment_id=str(
                row.get(
                    "segment_id",
                    f"{case['preset']}_L{row['line_id']}_P{first:06d}-{last:06d}",
                )
            ),
            line_id=str(row["line_id"]),
            first_point_id=first,
            last_point_id=last,
            n_sampling_points=int(
                round(float(row.get("n_sampling_points", last - first + 1)))
            ),
            selection_mae_m=float(row.get("selection_mae_m", np.nan)),
            river_median_mae_m=float(row.get("river_median_mae_m", np.nan)),
            common_pixels=int(round(float(row.get("common_pixels", 0)))),
        )
    return result


def load_prediction_manifest(
    h046: ModuleType,
    pred_root: Path,
    case: Mapping[str, str],
) -> Tuple[Path, List[Dict[str, str]]]:
    river_dir = h046.resolve_river_dir(
        pred_root,
        case["experiment"],
        case["preset"],
        case["river"],
    )
    manifest_path, _summary_path = h046.locate_manifest(river_dir)
    rows, audit = h046.read_manifest_with_rebased_paths(
        manifest_path,
        river_dir,
        "strict_meter_AGU",
    )
    if audit:
        print(
            f"[PATH-QA] {case['preset']}: rebased {len(audit)} manifest paths"
        )
    return river_dir, rows


def rows_for_selection(
    rows: Sequence[Dict[str, str]],
    selection: ReachSelection,
) -> List[Dict[str, str]]:
    candidates = []
    for row in rows:
        try:
            point_id = int(round(float(row["tile_id"])))
        except Exception:
            continue
        if selection.first_point_id <= point_id <= selection.last_point_id:
            candidates.append(row)
    candidates.sort(key=lambda row: int(round(float(row["tile_id"]))))

    if len(candidates) != selection.n_sampling_points:
        raise RuntimeError(
            f"Could not reconstruct selected reach {selection.segment_id}: "
            f"expected {selection.n_sampling_points} manifest rows between "
            f"point IDs {selection.first_point_id}-{selection.last_point_id}, "
            f"found {len(candidates)}. Use --manual_selection_csv if needed."
        )
    return candidates


def assemble_reach(
    h046: ModuleType,
    rows: Sequence[Dict[str, str]],
    tile_root: Path,
    max_dense_pixels: int = 30_000_000,
) -> ReachData:
    bounds: List[Tuple[float, float, float, float]] = []
    sources: List[Dict[str, str]] = []
    transforms: List[Any] = []

    for row in rows:
        source = h046.processed_tile_paths(tile_root, row)
        gt, transform = h046.read_tif(source["tile_path"])
        height, width = gt.shape
        bounds.append(
            (
                transform.c,
                transform.f,
                transform.c + width * abs(transform.a),
                transform.f - height * abs(transform.e),
            )
        )
        sources.append(source)
        transforms.append(transform)

    left = min(value[0] for value in bounds)
    top = max(value[1] for value in bounds)
    right = max(value[2] for value in bounds)
    bottom = min(value[3] for value in bounds)
    resolution = abs(float(transforms[0].a))
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    if width * height > max_dense_pixels:
        raise RuntimeError(f"Selected reach grid too large: {height}x{width}")

    shape = (height, width)
    gt_sum = np.zeros(shape, dtype=np.float64)
    gt_count = np.zeros(shape, dtype=np.uint16)
    pred_sum = np.zeros(shape, dtype=np.float64)
    pred_count = np.zeros(shape, dtype=np.uint16)
    final_mask_union = np.zeros(shape, dtype=bool)
    centers: List[Tuple[float, float]] = []

    for row, source in zip(rows, sources):
        gt, transform = h046.read_tif(source["tile_path"])
        core, core_transform = h046.read_tif(source["core_loss_path"])
        prediction, pred_transform = h046.read_tif(row["avg_pred_tile_path"])

        if not h046.affine_compatible(transform, core_transform):
            raise RuntimeError(f"Core-mask transform mismatch for key={row['key']}")
        if not h046.affine_compatible(transform, pred_transform):
            raise RuntimeError(f"Prediction transform mismatch for key={row['key']}")
        if not (gt.shape == core.shape == prediction.shape):
            raise RuntimeError(
                f"Shape mismatch for key={row['key']}: "
                f"GT={gt.shape}, core={core.shape}, prediction={prediction.shape}"
            )

        gt = gt.astype(np.float32, copy=False)
        prediction = prediction.astype(np.float32, copy=False)
        valid_gt = h046.valid_gt(gt)
        core_mask = (
            np.isfinite(core)
            & (core.astype(np.float32) > 0.5)
            & valid_gt
        )
        final = core_mask & h046.valid_pred(prediction)

        tile_height, tile_width = gt.shape
        row0 = int(round((top - transform.f) / resolution))
        col0 = int(round((transform.c - left) / resolution))
        target = np.s_[row0 : row0 + tile_height, col0 : col0 + tile_width]

        local_gt_sum = gt_sum[target]
        local_gt_count = gt_count[target]
        local_gt_sum[valid_gt] += gt[valid_gt]
        local_gt_count[valid_gt] += 1

        local_pred_sum = pred_sum[target]
        local_pred_count = pred_count[target]
        local_pred_sum[final] += prediction[final]
        local_pred_count[final] += 1
        final_mask_union[target] |= final

        center_x = transform.c + tile_width * abs(transform.a) / 2.0
        center_y = transform.f - tile_height * abs(transform.e) / 2.0
        centers.append(
            (
                (center_x - left) / resolution,
                (top - center_y) / resolution,
            )
        )

    gt_mosaic = h046.divide(gt_sum, gt_count)
    pred_mosaic = h046.divide(pred_sum, pred_count)
    final_mask = (
        final_mask_union
        & np.isfinite(gt_mosaic)
        & np.isfinite(pred_mosaic)
    )
    if not np.any(final_mask):
        raise RuntimeError("Selected reach has no final prediction pixels.")

    signed_error = (
        pred_mosaic[final_mask].astype(np.float64)
        - gt_mosaic[final_mask].astype(np.float64)
    )
    absolute = np.abs(signed_error)
    return ReachData(
        gt=gt_mosaic,
        prediction=pred_mosaic,
        final_mask=final_mask,
        centers=tuple(centers),
        resolution_m=resolution,
        local_mae_m=float(absolute.mean()),
        local_rmse_m=float(np.sqrt(np.square(signed_error).mean())),
        local_bias_m=float(signed_error.mean()),
        n_final_pixels=int(final_mask.sum()),
    )


def find_unique_summary(
    error_root: Path,
    case: Mapping[str, str],
) -> Path:
    candidates: List[Path] = []
    for path in error_root.rglob("*summary.json"):
        if case["river"].lower() not in str(path).lower():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if (
            "unique_geospatial_mae_m" in payload
            and "unique_geospatial_rmse_m" in payload
        ):
            candidates.append(path)

    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(
            f"No unique-geospatial summary found for {case['river']} "
            f"below {error_root}"
        )
    preferred = [
        path for path in unique
        if case["experiment"].lower() in str(path).lower()
    ]
    if len(preferred) == 1:
        return preferred[0]
    raise RuntimeError(
        f"Ambiguous unique-geospatial summaries for {case['river']}:\n"
        + "\n".join(str(path) for path in unique)
    )

def read_fullriver_metrics(
    error_root: Path,
    case: Mapping[str, str],
) -> Dict[str, Any]:
    summary_path = find_unique_summary(error_root, case)
    payload = json.loads(summary_path.read_text())
    mae = payload.get("unique_geospatial_mae_m")
    rmse = payload.get("unique_geospatial_rmse_m")
    if mae is None or rmse is None:
        raise RuntimeError(
            f"Unique-geospatial MAE/RMSE missing from {summary_path}"
        )
    return {
        "mae_m": float(mae),
        "rmse_m": float(rmse),
        "bias_m": float(payload.get("unique_geospatial_bias_m", np.nan)),
        "n_pixels": int(payload.get("unique_geospatial_n_pixels", 0)),
        "summary_path": str(summary_path),
        "comparison_mask": payload.get("comparison_mask", ""),
    }


def crop_from_mask(mask: np.ndarray, padding: int) -> Tuple[slice, slice]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.s_[:, :]
    row0 = max(int(rows.min()) - padding, 0)
    row1 = min(int(rows.max()) + padding + 1, mask.shape[0])
    col0 = max(int(cols.min()) - padding, 0)
    col1 = min(int(cols.max()) + padding + 1, mask.shape[1])
    return np.s_[row0:row1, col0:col1]


def robust_limits(
    arrays: Sequence[np.ndarray],
    low: float,
    high: float,
) -> Tuple[float, float]:
    values = [array[np.isfinite(array)] for array in arrays]
    values = [array for array in values if array.size]
    if not values:
        return 0.0, 1.0
    combined = np.concatenate(values).astype(np.float64, copy=False)
    vmin = float(np.percentile(combined, low))
    vmax = float(np.percentile(combined, high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(combined))
        vmax = float(np.nanmax(combined))
        if vmin == vmax:
            vmax = vmin + 1.0
    return vmin, vmax


def nice_scale_length(panel_width_m: float) -> float:
    target = max(panel_width_m * 0.25, 1.0)
    power = 10.0 ** math.floor(math.log10(target))
    normalized = target / power
    if normalized < 1.5:
        nice = 1.0
    elif normalized < 3.5:
        nice = 2.0
    elif normalized < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * power


def add_scale_bar(
    ax: plt.Axes,
    width_pixels: int,
    height_pixels: int,
    resolution_m: float,
) -> float:
    panel_width_m = width_pixels * resolution_m
    length_m = nice_scale_length(panel_width_m)
    length_px = length_m / resolution_m
    x0 = width_pixels * 0.06
    y0 = height_pixels * 0.92
    ax.plot(
        [x0, x0 + length_px],
        [y0, y0],
        color="black",
        linewidth=3.0,
        solid_capstyle="butt",
    )
    ax.plot(
        [x0, x0],
        [y0 - height_pixels * 0.012, y0 + height_pixels * 0.012],
        color="black",
        linewidth=1.5,
    )
    ax.plot(
        [x0 + length_px, x0 + length_px],
        [y0 - height_pixels * 0.012, y0 + height_pixels * 0.012],
        color="black",
        linewidth=1.5,
    )
    label = f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m:g} m"
    ax.text(
        x0 + length_px / 2.0,
        y0 - height_pixels * 0.025,
        label,
        ha="center",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2),
    )
    return length_m


def add_flow_arrow(
    ax: plt.Axes,
    centers: Sequence[Tuple[float, float]],
    crop: Tuple[slice, slice],
    width_pixels: int,
    height_pixels: int,
) -> str:
    if len(centers) < 2:
        return "unavailable"

    row_slice, col_slice = crop
    col_offset = col_slice.start or 0
    row_offset = row_slice.start or 0
    first = np.array(
        [centers[0][0] - col_offset, centers[0][1] - row_offset],
        dtype=float,
    )
    last = np.array(
        [centers[-1][0] - col_offset, centers[-1][1] - row_offset],
        dtype=float,
    )
    vector = last - first
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return "degenerate"
    unit = vector / length

    anchor = np.array([width_pixels * 0.82, height_pixels * 0.16])
    arrow_length = min(width_pixels, height_pixels) * 0.16
    start = anchor - unit * arrow_length / 2.0
    end = anchor + unit * arrow_length / 2.0
    ax.annotate(
        "",
        xy=(end[0], end[1]),
        xytext=(start[0], start[1]),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0),
    )
    ax.text(
        anchor[0],
        anchor[1] - height_pixels * 0.035,
        "Flow",
        ha="center",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
    )
    return f"first_to_last_sampling_center; dx={vector[0]:.3f}, dy={vector[1]:.3f}"


def make_colormaps() -> Tuple[Any, Any]:
    elevation = plt.get_cmap("viridis").copy()
    elevation.set_bad("white")
    error = plt.get_cmap("magma").copy()
    error.set_bad("white")
    return elevation, error


def render_figure(
    selections: Mapping[str, ReachSelection],
    reach_data: Mapping[str, ReachData],
    fullriver_metrics: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    output_png: Path,
    output_pdf: Path,
) -> List[Dict[str, Any]]:
    elevation_cmap, error_cmap = make_colormaps()
    error_max = float(args.absolute_error_max_m)
    low, high = (float(value) for value in args.elevation_percentiles)

    figure_height = args.figure_width_in * 1.14
    fig = plt.figure(figsize=(args.figure_width_in, figure_height))
    layout = gridspec.GridSpec(
        3,
        6,
        figure=fig,
        width_ratios=(1.35, 3.0, 3.0, 0.16, 3.0, 0.18),
        height_ratios=(1.0, 1.0, 1.0),
        left=0.035,
        right=0.965,
        bottom=0.065,
        top=0.895,
        wspace=0.10,
        hspace=0.12,
    )

    fig.suptitle(args.title, fontsize=18, fontweight="bold", y=0.972)
    fig.text(0.5, 0.942, args.subtitle, ha="center", va="center", fontsize=12)

    column_centers = {
        "gt": (layout[0, 1].get_position(fig).x0 + layout[0, 1].get_position(fig).x1) / 2,
        "pred": (layout[0, 2].get_position(fig).x0 + layout[0, 2].get_position(fig).x1) / 2,
        "error": (layout[0, 4].get_position(fig).x0 + layout[0, 4].get_position(fig).x1) / 2,
    }
    fig.text(column_centers["gt"], 0.910, "Ground truth", ha="center", fontsize=12, fontweight="bold")
    fig.text(column_centers["pred"], 0.910, "Prediction", ha="center", fontsize=12, fontweight="bold")
    fig.text(column_centers["error"], 0.910, "Absolute error", ha="center", fontsize=12, fontweight="bold")

    error_axes: List[plt.Axes] = []
    figure_rows: List[Dict[str, Any]] = []

    for row_index, case in enumerate(CASES):
        preset = case["preset"]
        selection = selections[preset]
        data = reach_data[preset]
        full = fullriver_metrics[preset]

        crop = crop_from_mask(data.final_mask, int(args.crop_padding))
        gt = np.where(data.final_mask, data.gt, np.nan)[crop]
        prediction = np.where(data.final_mask, data.prediction, np.nan)[crop]
        absolute_error = np.abs(prediction - gt)
        final_mask_crop = data.final_mask[crop]
        height_pixels, width_pixels = gt.shape

        vmin, vmax = robust_limits([gt, prediction], low, high)

        annotation_ax = fig.add_subplot(layout[row_index, 0])
        annotation_ax.axis("off")
        annotation_ax.text(
            0.0,
            0.88,
            case["label"],
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        annotation_ax.text(
            0.0,
            0.69,
            (
                f"Full-river MAE = {full['mae_m']:.2f} m\n"
                f"Full-river RMSE = {full['rmse_m']:.2f} m\n\n"
                f"Representative reach\n"
                f"local MAE = {data.local_mae_m:.2f} m\n"
                f"{selection.segment_id}"
            ),
            ha="left",
            va="top",
            fontsize=9,
            linespacing=1.35,
        )

        gt_ax = fig.add_subplot(layout[row_index, 1])
        pred_ax = fig.add_subplot(layout[row_index, 2])
        elevation_cax = fig.add_subplot(layout[row_index, 3])
        error_ax = fig.add_subplot(layout[row_index, 4])
        error_axes.append(error_ax)

        gt_image = gt_ax.imshow(
            gt,
            cmap=elevation_cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        pred_ax.imshow(
            prediction,
            cmap=elevation_cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        error_image = error_ax.imshow(
            absolute_error,
            cmap=error_cmap,
            vmin=0.0,
            vmax=error_max,
            interpolation="nearest",
        )

        for axis in (gt_ax, pred_ax, error_ax):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
            for spine in axis.spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor("0.35")

        scale_length = add_scale_bar(
            gt_ax,
            width_pixels,
            height_pixels,
            data.resolution_m,
        )
        flow_note = add_flow_arrow(
            gt_ax,
            data.centers,
            crop,
            width_pixels,
            height_pixels,
        )

        elevation_bar = fig.colorbar(gt_image, cax=elevation_cax)
        elevation_bar.set_label("Elevation / bathymetry (m)", fontsize=8)
        elevation_bar.ax.tick_params(labelsize=7)

        figure_rows.append(
            {
                "preset": preset,
                "river": case["river"],
                "river_label": case["label"],
                "segment_id": selection.segment_id,
                "line_id": selection.line_id,
                "first_point_id": selection.first_point_id,
                "last_point_id": selection.last_point_id,
                "n_sampling_points": selection.n_sampling_points,
                "selection_basis": "closest local strict-meter own-final MAE to river median",
                "selection_local_own_final_mae_m": selection.selection_mae_m,
                "river_median_selection_mae_m": selection.river_median_mae_m,
                "selection_final_pixels": selection.common_pixels,
                "display_local_own_final_mae_m": data.local_mae_m,
                "display_local_own_final_rmse_m": data.local_rmse_m,
                "display_local_own_final_bias_m": data.local_bias_m,
                "display_local_own_final_pixels": data.n_final_pixels,
                "fullriver_unique_mae_m": full["mae_m"],
                "fullriver_unique_rmse_m": full["rmse_m"],
                "fullriver_unique_bias_m": full["bias_m"],
                "fullriver_unique_pixels": full["n_pixels"],
                "fullriver_summary_path": full["summary_path"],
                "elevation_vmin_m": vmin,
                "elevation_vmax_m": vmax,
                "absolute_error_vmin_m": 0.0,
                "absolute_error_vmax_m": error_max,
                "scale_bar_m": scale_length,
                "flow_direction_source": flow_note,
                "resolution_m": data.resolution_m,
                "crop_height_pixels": height_pixels,
                "crop_width_pixels": width_pixels,
                "final_mask_pixels_in_crop": int(final_mask_crop.sum()),
            }
        )

    shared_error_cax = fig.add_subplot(layout[:, 5])
    shared_error_bar = fig.colorbar(error_image, cax=shared_error_cax)
    shared_error_bar.set_label(
        f"Absolute error (m; clipped at {error_max:g} m)",
        fontsize=9,
    )
    shared_error_bar.ax.tick_params(labelsize=8)

    fig.text(
        0.5,
        0.025,
        (
            "Representative continuous reaches selected automatically by "
            "median local strict-meter MAE; GT and prediction share a "
            "per-river scale; absolute error uses a common 0–"
            f"{error_max:g} m scale."
        ),
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return figure_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    h046 = import_h046(args.h046_script)
    tile_root = h046.resolve_processed_tile_root(args.tile_base)

    for path, label in (
        (args.prediction_root, "strict prediction root"),
        (args.error_root, "strict error root"),
        (tile_root, "resolved processed tile root"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label}: {path}")

    manifest_rows: Dict[str, List[Dict[str, str]]] = {}
    manifest_paths: Dict[str, str] = {}
    for case in CASES:
        river_dir, rows = load_prediction_manifest(
            h046,
            args.prediction_root,
            case,
        )
        manifest_rows[case["preset"]] = rows
        manifest_paths[case["preset"]] = str(river_dir)

    candidate_metrics_path = (
        args.candidate_metrics_csv
        if args.candidate_metrics_csv is not None
        else args.output_dir / "H050_all_candidate_reach_metrics.csv"
    )

    if args.manual_selection_csv is not None:
        selections = select_reaches_manually(args.manual_selection_csv)
        selection_mode = "manual_selection_csv"
        candidate_rows: List[Dict[str, Any]] = []
    else:
        if args.candidate_metrics_csv is not None:
            if not candidate_metrics_path.is_file():
                raise FileNotFoundError(candidate_metrics_path)
            candidate_frame = pd.read_csv(candidate_metrics_path)
            candidate_rows = candidate_frame.to_dict("records")
            selection_mode = "precomputed_candidate_metrics_csv"
        else:
            candidate_rows = []
            for case in CASES:
                candidate_rows.extend(
                    evaluate_candidate_reaches(
                        h046,
                        manifest_rows[case["preset"]],
                        tile_root,
                        args.tile_base,
                        case,
                        args.segment_size,
                        args.segment_stride,
                        args.min_segment_points,
                    )
                )
            write_csv(candidate_metrics_path, candidate_rows)
            candidate_frame = pd.DataFrame(candidate_rows)
            selection_mode = "automatic_median_local_own_final_mae"

        if not candidate_rows and args.candidate_metrics_csv is None:
            raise RuntimeError("No candidate reach metrics were produced.")
        if args.candidate_metrics_csv is not None:
            candidate_frame = pd.read_csv(candidate_metrics_path)
        else:
            candidate_frame = pd.DataFrame(candidate_rows)

        selections = select_reaches_from_candidate_metrics(
            candidate_frame,
            args.min_final_pixels,
            args.required_sampling_points,
        )

    reach_data: Dict[str, ReachData] = {}
    fullriver_metrics: Dict[str, Dict[str, Any]] = {}

    for case in CASES:
        preset = case["preset"]
        selected_rows = rows_for_selection(
            manifest_rows[preset],
            selections[preset],
        )
        reach_data[preset] = assemble_reach(
            h046,
            selected_rows,
            tile_root,
        )
        fullriver_metrics[preset] = read_fullriver_metrics(
            args.error_root,
            case,
        )
        print(
            f"[SELECT] {preset}: {selections[preset].segment_id}; "
            f"river median={selections[preset].river_median_mae_m:.4f} m; "
            f"selected={reach_data[preset].local_mae_m:.4f} m"
        )

    output_png = args.output_dir / f"{args.output_stem}.png"
    output_pdf = args.output_dir / f"{args.output_stem}.pdf"
    figure_rows = render_figure(
        selections,
        reach_data,
        fullriver_metrics,
        args,
        output_png,
        output_pdf,
    )

    selection_rows = []
    for case in CASES:
        selection = selections[case["preset"]]
        selection_rows.append(
            {
                "preset": selection.preset,
                "river": selection.river,
                "river_label": selection.river_label,
                "segment_id": selection.segment_id,
                "line_id": selection.line_id,
                "first_point_id": selection.first_point_id,
                "last_point_id": selection.last_point_id,
                "n_sampling_points": selection.n_sampling_points,
                "selection_mae_m": selection.selection_mae_m,
                "river_median_mae_m": selection.river_median_mae_m,
                "final_pixels": selection.common_pixels,
            }
        )
    write_csv(
        args.output_dir / "H050_selected_representative_reaches.csv",
        selection_rows,
    )
    write_csv(
        args.output_dir / "H050_figure_panel_metadata.csv",
        figure_rows,
    )

    summary = {
        "figure_png": str(output_png),
        "figure_pdf": str(output_pdf),
        "selection_mode": selection_mode,
        "mask_regime": "strict",
        "prediction_root": str(args.prediction_root),
        "error_root": str(args.error_root),
        "processed_tile_root": str(tile_root),
        "candidate_metrics_csv": str(candidate_metrics_path),
        "h046_helper": str(args.h046_script.resolve()),
        "h046_helper_sha256": sha256(args.h046_script.resolve()),
        "prediction_model": "strict mask + meter-domain objective",
        "fullriver_metric_source": (
            "F062 unique_geospatial_*; each final overlap-averaged "
            "pixel counted once"
        ),
        "display_footprint": (
            "Core_Loss_Mask_Pixel AND valid processed GT AND valid formal prediction"
        ),
        "elevation_range_policy": (
            f"GT and prediction share {args.elevation_percentiles[0]}–"
            f"{args.elevation_percentiles[1]} percentile limits within each river"
        ),
        "absolute_error_range_m": [0.0, float(args.absolute_error_max_m)],
        "manifest_river_directories": manifest_paths,
        "rows": figure_rows,
    }
    (args.output_dir / "H050_AGU_figure_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("[DONE]", output_png)
    print("[DONE]", output_pdf)


if __name__ == "__main__":
    main()
