#!/usr/bin/env python3
"""H045: full-river common-loss-pixel comparison for four configurations.

Configurations
--------------
1. Strict mask + normalized objective
2. Strict mask + meter objective
3. Relaxed mask + normalized objective
4. Relaxed mask + meter objective

The principal figures use an exact four-way common loss-pixel footprint:

    Core_Loss_Mask_Pixel
    AND valid ground truth
    AND valid prediction from all four configurations

Each overlapping geospatial pixel is counted once after overlap averaging.
Pairwise comparisons are also reported on exact pairwise common footprints.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    "strict_normalized": "Strict + Normalized objective",
    "strict_meter": "Strict + Meter objective",
    "relaxed_normalized": "Relaxed + Normalized objective",
    "relaxed_meter": "Relaxed + Meter objective",
}
PAIR_ORDER = (
    ("strict_normalized", "strict_meter", "Strict: normalized vs meter"),
    ("relaxed_normalized", "relaxed_meter", "Relaxed: normalized vs meter"),
    ("strict_normalized", "relaxed_normalized", "Normalized: strict vs relaxed"),
    ("strict_meter", "relaxed_meter", "Meter: strict vs relaxed"),
)


@dataclass(frozen=True)
class Affine:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


@dataclass(frozen=True)
class Grid:
    left: float
    top: float
    resx: float
    resy: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare full-river errors on common loss pixels.",
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
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--density_max_error_m", type=float, default=2.0)
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


@lru_cache(maxsize=192)
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


def valid_gt(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float64, copy=False)
    return np.isfinite(values) & (values > NODATA_THRESHOLD) & (values != NODATA)


def valid_pred(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float64, copy=False)
    return np.isfinite(values) & (values > NODATA_THRESHOLD) & (values != NODATA)


def canonical_grid(all_rows: Sequence[Dict[str, str]]) -> Grid:
    lefts: List[float] = []
    rights: List[float] = []
    tops: List[float] = []
    bottoms: List[float] = []
    resx = resy = None
    seen = set()
    for row in all_rows:
        path = row["tile_path"]
        if path in seen:
            continue
        seen.add(path)
        array, transform = read_tif(path)
        height, width = array.shape
        rx, ry = abs(transform.a), abs(transform.e)
        if resx is None:
            resx, resy = rx, ry
        if abs(rx - resx) > 1e-6 or abs(ry - resy) > 1e-6:
            raise RuntimeError("Resolution mismatch among prediction branches")
        lefts.append(transform.c)
        rights.append(transform.c + width * rx)
        tops.append(transform.f)
        bottoms.append(transform.f - height * ry)
    if not lefts:
        raise RuntimeError("No tiles for canonical grid")
    left, right = min(lefts), max(rights)
    top, bottom = max(tops), min(bottoms)
    return Grid(
        left=left,
        top=top,
        resx=float(resx),
        resy=float(resy),
        width=int(round((right - left) / float(resx))),
        height=int(round((top - bottom) / float(resy))),
    )


def deduplicate(keys: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    gt = gt[order].astype(np.float64, copy=False)
    pred = pred[order].astype(np.float64, copy=False)
    unique, starts = np.unique(keys, return_index=True)
    counts = np.diff(np.r_[starts, keys.size]).astype(np.float64)
    gt_unique = np.add.reduceat(gt, starts) / counts
    pred_unique = np.add.reduceat(pred, starts) / counts
    return unique, gt_unique.astype(np.float32), pred_unique.astype(np.float32)


def load_unique(rows: Sequence[Dict[str, str]], grid: Grid, label: str):
    key_parts: List[np.ndarray] = []
    gt_parts: List[np.ndarray] = []
    pred_parts: List[np.ndarray] = []
    for index, row in enumerate(rows, start=1):
        gt, transform = read_tif(row["tile_path"])
        pred, _ = read_tif(row["avg_pred_tile_path"])
        core_loss, _ = read_tif(row["core_loss_path"])
        gt = gt.astype(np.float32, copy=False)
        pred = pred.astype(np.float32, copy=False)
        mask = (
            np.isfinite(core_loss)
            & (core_loss.astype(np.float32) > 0.5)
            & valid_gt(gt)
            & valid_pred(pred)
        )
        flat = np.flatnonzero(mask.ravel()).astype(np.int64)
        if flat.size == 0:
            continue
        height, width = gt.shape
        row0 = int(round((grid.top - transform.f) / grid.resy))
        col0 = int(round((transform.c - grid.left) / grid.resx))
        rr = flat // width
        cc = flat % width
        keys = (np.int64(row0) + rr) * np.int64(grid.width) + (np.int64(col0) + cc)
        key_parts.append(keys)
        gt_parts.append(gt.ravel()[flat])
        pred_parts.append(pred.ravel()[flat])
        if index == 1 or index == len(rows) or index % 250 == 0:
            print(f"  [{label}] tiles {index}/{len(rows)}")
    if not key_parts:
        raise RuntimeError(f"No valid loss pixels for {label}")
    return deduplicate(np.concatenate(key_parts), np.concatenate(gt_parts), np.concatenate(pred_parts))


def align(keys: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(keys, target)
    if np.any(positions >= keys.size) or np.any(keys[positions] != target):
        raise RuntimeError("Target keys are not a subset of source keys")
    return values[positions]


def common_keys(branches: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], names: Sequence[str]) -> np.ndarray:
    common = branches[names[0]][0]
    for name in names[1:]:
        common = np.intersect1d(common, branches[name][0], assume_unique=True)
    return common


def stats(error: np.ndarray) -> Dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    error = error[np.isfinite(error)]
    if error.size == 0:
        return {key: float("nan") for key in (
            "mae_m", "rmse_m", "bias_m", "median_abs_error_m", "p90_abs_error_m", "p95_abs_error_m", "max_abs_error_m"
        )} | {"n_pixels": 0}
    absolute = np.abs(error)
    return {
        "n_pixels": int(error.size),
        "mae_m": float(absolute.mean()),
        "rmse_m": float(np.sqrt(np.square(error).mean())),
        "bias_m": float(error.mean()),
        "median_abs_error_m": float(np.median(absolute)),
        "p90_abs_error_m": float(np.percentile(absolute, 90)),
        "p95_abs_error_m": float(np.percentile(absolute, 95)),
        "max_abs_error_m": float(absolute.max()),
        "fraction_abs_error_le_0p10m": float((absolute <= 0.10).mean()),
        "fraction_abs_error_le_0p25m": float((absolute <= 0.25).mean()),
        "fraction_abs_error_le_0p50m": float((absolute <= 0.50).mean()),
        "fraction_abs_error_le_1p00m": float((absolute <= 1.00).mean()),
    }


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


def validate_relaxed_summary(summary_path: Optional[Path], label: str) -> None:
    if summary_path is None:
        raise RuntimeError(f"Missing summary for corrected relaxed branch: {label}")
    data = json.loads(summary_path.read_text())
    if data.get("prediction_patch_filter_applied") is not True:
        raise RuntimeError(
            f"Refusing legacy relaxed prediction without prediction_patch_filter_applied=true: {summary_path}"
        )


def grouped_bar(metrics: Sequence[Mapping[str, Any]], output: Path, dpi: int) -> None:
    rows = list(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    width = 0.19
    x = np.arange(len(CASES), dtype=float)
    for ax, metric_name, title in (
        (axes[0], "mae_m", "Four-way common loss pixels: MAE"),
        (axes[1], "rmse_m", "Four-way common loss pixels: RMSE"),
    ):
        for index, config in enumerate(CONFIG_ORDER):
            values = [
                next(row[metric_name] for row in rows if row["preset"] == case["preset"] and row["configuration"] == config)
                for case in CASES
            ]
            ax.bar(x + (index - 1.5) * width, values, width, label=CONFIG_LABELS[config])
        ax.set_xticks(x, [case["label"] for case in CASES])
        ax.set_ylabel("Error (m)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Common loss pixels = Core_Loss_Mask_Pixel ∩ valid GT ∩ valid predictions from all four configurations; each geospatial pixel counted once.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def distribution_figure(cache: Mapping[str, Mapping[str, np.ndarray]], output: Path, dpi: int, max_error: float) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    for row_index, case in enumerate(CASES):
        data = cache[case["preset"]]
        all_abs = np.concatenate([np.abs(data[config] - data["gt"]) for config in CONFIG_ORDER])
        limit = min(max_error, max(0.05, float(np.percentile(all_abs, 99.5))))
        bins = np.linspace(0, limit, 120)

        ax = axes[row_index, 0]
        for config in CONFIG_ORDER:
            absolute = np.abs(data[config] - data["gt"])
            ax.hist(
                absolute[absolute <= limit],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.6,
                label=CONFIG_LABELS[config],
            )
        ax.set_title(f"{case['label']} — absolute-error density")
        ax.set_xlabel("Absolute error (m)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[row_index, 1]
        for config in CONFIG_ORDER:
            absolute = np.sort(np.abs(data[config] - data["gt"]))
            cumulative = np.arange(1, absolute.size + 1) / absolute.size
            ax.plot(absolute, cumulative, label=CONFIG_LABELS[config])
        ax.set_xlim(0, limit)
        ax.set_ylim(0, 1)
        ax.set_title(f"{case['label']} — absolute-error CDF")
        ax.set_xlabel("Absolute error (m)")
        ax.set_ylabel("Cumulative fraction")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Full-river error distributions on the four-way common loss-pixel footprint")
    fig.text(
        0.5,
        0.01,
        "Common loss pixels = Core_Loss_Mask_Pixel ∩ valid GT ∩ valid predictions from all four configurations.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.98))
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def pairwise_delta_plot(pair_rows: Sequence[Mapping[str, Any]], output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (_, _, pair_label) in zip(axes.flat, PAIR_ORDER):
        subset = [row for row in pair_rows if row["pair"] == pair_label]
        values = [next(row["second_minus_first_mae_m"] for row in subset if row["preset"] == case["preset"]) for case in CASES]
        x = np.arange(len(CASES), dtype=float)
        bars = ax.bar(x, values)
        ax.axhline(0, linewidth=1)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:+.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
        ax.set_xticks(x, [case["label"] for case in CASES], rotation=12, ha="right")
        ax.set_ylabel("Second MAE − first MAE (m)")
        ax.set_title(pair_label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Pairwise full-river MAE differences on exact pairwise common loss pixels")
    fig.text(
        0.5,
        0.01,
        "Every panel uses its own exact pairwise common loss-pixel footprint; negative values favor the second configuration.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def threshold_figure(cache: Mapping[str, Mapping[str, np.ndarray]], output: Path, dpi: int) -> None:
    thresholds = (0.10, 0.25, 0.50, 1.00)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, case in zip(axes, CASES):
        data = cache[case["preset"]]
        x = np.arange(len(thresholds), dtype=float)
        width = 0.19
        for index, config in enumerate(CONFIG_ORDER):
            absolute = np.abs(data[config] - data["gt"])
            fractions = [100 * float((absolute <= threshold).mean()) for threshold in thresholds]
            ax.bar(x + (index - 1.5) * width, fractions, width, label=CONFIG_LABELS[config])
        ax.set_xticks(x, [f"≤{threshold:g} m" for threshold in thresholds])
        ax.set_title(case["label"])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Four-way common loss pixels within threshold (%)")
    fig.suptitle("Full-river absolute-error threshold coverage")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    four_way_metrics: List[Dict[str, Any]] = []
    pairwise_metrics: List[Dict[str, Any]] = []
    distribution_cache: Dict[str, Dict[str, np.ndarray]] = {}
    manifest_path_audit: List[Dict[str, str]] = []

    for case in CASES:
        print("=" * 76)
        print(case["label"])
        directories = branch_dirs(args, case)
        manifests: Dict[str, List[Dict[str, str]]] = {}
        all_rows: List[Dict[str, str]] = []
        for config, directory in directories.items():
            manifest, summary = locate_manifest(directory)
            if config.startswith("relaxed_"):
                validate_relaxed_summary(summary, config)
            rows, path_audit = read_manifest_with_rebased_paths(
                manifest,
                directory,
                config,
            )
            manifests[config] = rows
            all_rows.extend(rows)
            manifest_path_audit.extend(path_audit)
            print(f"[{config}] manifest={manifest} tiles={len(rows)}")

        grid = canonical_grid(all_rows)
        branches: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for config in CONFIG_ORDER:
            branches[config] = load_unique(manifests[config], grid, config)

        common = common_keys(branches, CONFIG_ORDER)
        if common.size == 0:
            raise RuntimeError(f"No four-way common loss pixels for {case['label']}")
        gt_stack = [align(branches[config][0], branches[config][1], common) for config in CONFIG_ORDER]
        gt = np.mean(np.stack(gt_stack, axis=0).astype(np.float64), axis=0).astype(np.float32)
        data: Dict[str, np.ndarray] = {"keys": common, "gt": gt}
        for config in CONFIG_ORDER:
            pred = align(branches[config][0], branches[config][2], common)
            data[config] = pred
            four_way_metrics.append(
                {
                    "preset": case["preset"],
                    "river": case["river"],
                    "river_label": case["label"],
                    "configuration": config,
                    "configuration_label": CONFIG_LABELS[config],
                    "comparison_footprint": "four-way common loss pixels",
                    "common_loss_pixels": int(common.size),
                    **stats(pred - gt),
                }
            )
        distribution_cache[case["preset"]] = data
        np.savez_compressed(cache_dir / f"H045_{case['preset']}_fourway_common_loss_pixels.npz", **data)

        for first, second, pair_label in PAIR_ORDER:
            pair_common = common_keys(branches, (first, second))
            first_gt = align(branches[first][0], branches[first][1], pair_common)
            second_gt = align(branches[second][0], branches[second][1], pair_common)
            pair_gt = ((first_gt.astype(np.float64) + second_gt.astype(np.float64)) / 2.0).astype(np.float32)
            first_pred = align(branches[first][0], branches[first][2], pair_common)
            second_pred = align(branches[second][0], branches[second][2], pair_common)
            first_stats = stats(first_pred - pair_gt)
            second_stats = stats(second_pred - pair_gt)
            pairwise_metrics.append(
                {
                    "preset": case["preset"],
                    "river": case["river"],
                    "river_label": case["label"],
                    "pair": pair_label,
                    "first_configuration": first,
                    "second_configuration": second,
                    "comparison_footprint": "exact pairwise common loss pixels",
                    "common_loss_pixels": int(pair_common.size),
                    "first_mae_m": first_stats["mae_m"],
                    "second_mae_m": second_stats["mae_m"],
                    "second_minus_first_mae_m": second_stats["mae_m"] - first_stats["mae_m"],
                    "first_rmse_m": first_stats["rmse_m"],
                    "second_rmse_m": second_stats["rmse_m"],
                    "second_minus_first_rmse_m": second_stats["rmse_m"] - first_stats["rmse_m"],
                }
            )

    write_csv(output / "H045_manifest_path_rebase_audit.csv", manifest_path_audit)
    write_csv(output / "H045_fourway_common_loss_metrics.csv", four_way_metrics)
    write_csv(output / "H045_pairwise_common_loss_metrics.csv", pairwise_metrics)
    grouped_bar(four_way_metrics, output / "H045_fourway_common_mae_rmse.png", args.dpi)
    distribution_figure(
        distribution_cache,
        output / "H045_fourway_common_error_density_cdf.png",
        args.dpi,
        args.density_max_error_m,
    )
    pairwise_delta_plot(pairwise_metrics, output / "H045_pairwise_common_mae_deltas.png", args.dpi)
    threshold_figure(distribution_cache, output / "H045_fourway_common_threshold_fraction.png", args.dpi)

    summary = {
        "main_footprint": (
            "Core_Loss_Mask_Pixel AND valid GT AND valid predictions from all four configurations; "
            "overlap-averaged geospatial pixels counted once"
        ),
        "configurations": {name: CONFIG_LABELS[name] for name in CONFIG_ORDER},
        "manifest_path_rebase_count": len(manifest_path_audit),
        "manifest_path_rebase_audit": "H045_manifest_path_rebase_audit.csv",
        "four_way_metrics": four_way_metrics,
        "pairwise_metrics": pairwise_metrics,
    }
    (output / "H045_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("[DONE]", output)


if __name__ == "__main__":
    main()
