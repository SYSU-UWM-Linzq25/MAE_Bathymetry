#!/usr/bin/env python3
"""
F065: Compare full-river error distributions for two downstream objectives.

Models compared
---------------
1. Normalized-loss model:
   FullRiver_GT_Error_F020_TileVRT_D001NoDataSafe
2. Meter-loss model:
   FullRiver_GT_Error_F062_UniquePixel_D003MeterMAE_BaselineEval_D001NoDataSafe

Fair-comparison rule
--------------------
F020's original summary is tile-footprint weighted and may double-count overlap
pixels, whereas F062 includes unique-geospatial metrics. This script therefore
reconstructs UNIQUE full-river geospatial error arrays for BOTH experiments from
the original prediction manifests. Each final overlap-averaged river pixel is
counted once.

Error definition
----------------
    signed error = Prediction - GT, in meters

Comparison mask
---------------
    Core_Loss_Mask_Pixel AND valid_GT AND valid_prediction

Main publication-style figure
-----------------------------
Three rows x two columns:
    left  = absolute-error density
    right = absolute-error empirical CDF
Rows correspond to CA, CO, and Santiam holdouts.

Additional outputs
------------------
- signed-error density figure
- per-river and combined CSV summaries
- cached unique-error arrays for fast reruns
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


DEFAULT_NORM_ROOT = Path(
    "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
    "FullRiver_GT_Error_F020_TileVRT_D001NoDataSafe"
)

DEFAULT_METER_ROOT = Path(
    "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
    "FullRiver_GT_Error_F062_UniquePixel_D003MeterMAE_BaselineEval_D001NoDataSafe"
)

DEFAULT_OUT_DIR = Path(
    "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
    "FullRiver_ErrorDistribution_F065_NormVsMeter_D001NoDataSafe"
)

DEFAULT_CASES = [
    {
        "short_name": "CA Klamath",
        "norm_experiment": "holdout_CA_D001NoDataSafe",
        "meter_experiment": "holdout_CA_D003MeterMAE_BaselineEval_D001NoDataSafe",
        "river": "CA_KlamathRiver_TopoBathy_2018_D18",
    },
    {
        "short_name": "CO Upper Colorado",
        "norm_experiment": "holdout_CO_D001NoDataSafe",
        "meter_experiment": "holdout_CO_D003MeterMAE_BaselineEval_D001NoDataSafe",
        "river": "CO_UpperColorado_Topobathy_1_2020",
    },
    {
        "short_name": "OR Santiam",
        "norm_experiment": "holdout_Santiam_D001NoDataSafe",
        "meter_experiment": "holdout_Santiam_D003MeterMAE_BaselineEval_D001NoDataSafe",
        "river": "OR_SantiamRiverTB_Topobathy_1_D23",
    },
]

NORMALIZED_COLOR = "tab:blue"
METER_COLOR = "tab:orange"


@dataclass(frozen=True)
class Case:
    short_name: str
    norm_experiment: str
    meter_experiment: str
    river: str


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
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


def _tag_value(tags, key, default=None):
    if hasattr(tags, "get"):
        value = tags.get(key)
        if value is not None:
            return getattr(value, "value", value)
    try:
        if key in tags:
            value = tags[key]
            return getattr(value, "value", value)
    except Exception:
        pass
    return default


def _parse_nodata(tags) -> Optional[float]:
    value = _tag_value(tags, 42113, None)
    if value is None:
        value = _tag_value(tags, "GDAL_NODATA", None)
    if value is None:
        return None

    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, (tuple, list)):
            value = value[0]
        return float(str(value).strip().strip("\x00"))
    except Exception:
        return None


def read_one(path: Path) -> Tuple[np.ndarray, Optional[float]]:
    path = Path(path)
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        array = page.asarray()
        nodata = _parse_nodata(page.tags)
    return array, nodata


def valid_gt_mask(
    array: np.ndarray,
    nodata: float,
    threshold: float,
    source_nodata: Optional[float],
) -> np.ndarray:
    values = array.astype(np.float64, copy=False)
    valid = (
        np.isfinite(values)
        & (values > float(threshold))
        & (values != float(nodata))
    )
    if (
        source_nodata is not None
        and math.isfinite(float(source_nodata))
        and abs(float(source_nodata)) > 1e-100
    ):
        valid &= values != float(source_nodata)
    return valid


def valid_pred_mask(array: np.ndarray, nodata: float) -> np.ndarray:
    values = array.astype(np.float64, copy=False)
    return np.isfinite(values) & (values != float(nodata))


def resolve_mosaic_width(summary: Dict[str, Any]) -> int:
    for key in ("virtual_mosaic_width", "mosaic_width", "width"):
        value = summary.get(key)
        if value is not None:
            return int(value)

    pred_vrt = summary.get("pred_vrt_path")
    if pred_vrt:
        path = Path(pred_vrt)
        if path.is_file():
            root = ET.parse(str(path)).getroot()
            value = root.attrib.get("rasterXSize")
            if value is not None:
                return int(value)

    raise KeyError(
        "Could not determine virtual mosaic width from summary or pred_vrt_path."
    )


def stats_from_error(error: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(error, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "n_pixels": 0,
            "rmse_m": np.nan,
            "mae_m": np.nan,
            "bias_m": np.nan,
            "median_signed_error_m": np.nan,
            "median_abs_error_m": np.nan,
            "p75_abs_error_m": np.nan,
            "p90_abs_error_m": np.nan,
            "p95_abs_error_m": np.nan,
            "p99_abs_error_m": np.nan,
            "max_abs_error_m": np.nan,
        }

    absolute = np.abs(values)
    return {
        "n_pixels": int(values.size),
        "rmse_m": float(np.sqrt(np.mean(values**2))),
        "mae_m": float(np.mean(absolute)),
        "bias_m": float(np.mean(values)),
        "median_signed_error_m": float(np.median(values)),
        "median_abs_error_m": float(np.median(absolute)),
        "p75_abs_error_m": float(np.percentile(absolute, 75)),
        "p90_abs_error_m": float(np.percentile(absolute, 90)),
        "p95_abs_error_m": float(np.percentile(absolute, 95)),
        "p99_abs_error_m": float(np.percentile(absolute, 99)),
        "max_abs_error_m": float(np.max(absolute)),
    }


def collect_unique_error(
    source_dir: Path,
    summary_name: str,
    manifest_name: str,
    nodata: float,
    nodata_threshold: float,
    progress_every: int,
) -> np.ndarray:
    source_dir = Path(source_dir)
    summary_path = source_dir / summary_name
    manifest_path = source_dir / manifest_name

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    summary = read_json(summary_path)
    manifest = read_csv(manifest_path)
    mosaic_width = resolve_mosaic_width(summary)

    key_parts: List[np.ndarray] = []
    error_parts: List[np.ndarray] = []

    print(f"[COLLECT] {source_dir}")
    print(f"[COLLECT] manifest rows = {len(manifest):,}")

    for index, row in enumerate(manifest, start=1):
        tile_path = Path(row["tile_path"])
        pred_path = Path(row["avg_pred_tile_path"])
        core_loss_path = Path(row["core_loss_path"])

        gt, gt_source_nodata = read_one(tile_path)
        pred, _ = read_one(pred_path)
        core_loss, _ = read_one(core_loss_path)

        gt = gt.astype(np.float32, copy=False)
        pred = pred.astype(np.float32, copy=False)
        core_loss = core_loss.astype(np.float32, copy=False)

        core_mask = (core_loss > 0.5) & np.isfinite(core_loss)
        valid_gt = valid_gt_mask(
            gt,
            nodata=nodata,
            threshold=nodata_threshold,
            source_nodata=gt_source_nodata,
        )
        valid_pred = valid_pred_mask(pred, nodata)
        comparison_mask = core_mask & valid_gt & valid_pred

        if comparison_mask.any():
            error = pred - gt
            _, width = gt.shape
            row0 = int(row["mosaic_row0"])
            col0 = int(row["mosaic_col0"])

            local_flat = np.flatnonzero(comparison_mask.ravel()).astype(np.int64)
            local_row = local_flat // width
            local_col = local_flat % width

            global_key = (
                (np.int64(row0) + local_row) * np.int64(mosaic_width)
                + (np.int64(col0) + local_col)
            )

            key_parts.append(global_key)
            error_parts.append(error.ravel()[local_flat].astype(np.float32))

        if (
            index == 1
            or index == len(manifest)
            or index % max(progress_every, 1) == 0
        ):
            print(f"  processed {index:,}/{len(manifest):,} tiles")

    if not key_parts:
        raise RuntimeError(f"No valid comparison pixels found under {source_dir}")

    all_keys = np.concatenate(key_parts).astype(np.int64, copy=False)
    all_errors = np.concatenate(error_parts).astype(np.float64, copy=False)

    order = np.argsort(all_keys, kind="mergesort")
    sorted_keys = all_keys[order]
    sorted_errors = all_errors[order]

    _, starts = np.unique(sorted_keys, return_index=True)
    error_sum = np.add.reduceat(sorted_errors, starts)
    duplicate_counts = np.diff(np.r_[starts, sorted_keys.size]).astype(np.int64)
    unique_error = (error_sum / duplicate_counts).astype(np.float32, copy=False)

    print(f"[COLLECT] tile-footprint pixels = {all_errors.size:,}")
    print(f"[COLLECT] unique geospatial pixels = {unique_error.size:,}")
    print(
        "[COLLECT] footprint / unique ratio = "
        f"{all_errors.size / unique_error.size:.4f}"
    )

    return unique_error


def collect_or_load(
    cache_path: Path,
    source_dir: Path,
    summary_name: str,
    manifest_name: str,
    nodata: float,
    nodata_threshold: float,
    progress_every: int,
    overwrite_cache: bool,
) -> np.ndarray:
    cache_path = Path(cache_path)
    if cache_path.exists() and not overwrite_cache:
        print(f"[CACHE] loading {cache_path}")
        with np.load(cache_path) as cached:
            return cached["unique_error"].astype(np.float32, copy=False)

    error = collect_unique_error(
        source_dir=source_dir,
        summary_name=summary_name,
        manifest_name=manifest_name,
        nodata=nodata,
        nodata_threshold=nodata_threshold,
        progress_every=progress_every,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, unique_error=error)
    return error


def signed_limits(a: np.ndarray, b: np.ndarray, percentile: float = 99.5):
    values = np.concatenate([a.astype(np.float64), b.astype(np.float64)])
    values = values[np.isfinite(values)]
    lower = float(np.percentile(values, 100 - percentile))
    upper = float(np.percentile(values, percentile))
    maximum = max(abs(lower), abs(upper))
    if not np.isfinite(maximum) or maximum <= 0:
        maximum = 1.0
    return -maximum, maximum


def absolute_limit(a: np.ndarray, b: np.ndarray, percentile: float = 99.5):
    values = np.abs(
        np.concatenate([a.astype(np.float64), b.astype(np.float64)])
    )
    values = values[np.isfinite(values)]
    maximum = float(np.percentile(values, percentile))
    if not np.isfinite(maximum) or maximum <= 0:
        maximum = 1.0
    return maximum


def histogram_density(values: np.ndarray, bins: np.ndarray):
    density, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def empirical_cdf(values: np.ndarray, max_points: int = 25_000):
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    if sorted_values.size <= max_points:
        indices = np.arange(sorted_values.size)
    else:
        indices = np.linspace(
            0,
            sorted_values.size - 1,
            max_points,
            dtype=np.int64,
        )
    x = sorted_values[indices]
    y = (indices + 1) / sorted_values.size
    return x, y


def metric_text(stats: Dict[str, Any]) -> str:
    return (
        f"MAE={stats['mae_m']:.3f} m, "
        f"RMSE={stats['rmse_m']:.3f} m, "
        f"P95={stats['p95_abs_error_m']:.3f} m"
    )


def plot_publication_3x2(
    cases: Sequence[Case],
    data: Dict[str, Dict[str, np.ndarray]],
    stats: Dict[str, Dict[str, Dict[str, Any]]],
    output_png: Path,
    output_pdf: Path,
    display_percentile: float,
) -> None:
    fig, axes = plt.subplots(
        len(cases),
        2,
        figsize=(14, 4.0 * len(cases)),
        constrained_layout=True,
    )
    if len(cases) == 1:
        axes = np.asarray([axes])

    for row_index, case in enumerate(cases):
        normalized_abs = np.abs(data[case.river]["normalized"])
        meter_abs = np.abs(data[case.river]["meter"])
        upper = absolute_limit(
            normalized_abs,
            meter_abs,
            percentile=display_percentile,
        )

        # Left: absolute-error density.
        density_axis = axes[row_index, 0]
        bins = np.linspace(0, upper, 241)
        x_norm, y_norm = histogram_density(normalized_abs, bins)
        x_meter, y_meter = histogram_density(meter_abs, bins)

        density_axis.plot(
            x_norm,
            y_norm,
            color=NORMALIZED_COLOR,
            linewidth=2.0,
            label="Normalized-loss model",
        )
        density_axis.plot(
            x_meter,
            y_meter,
            color=METER_COLOR,
            linewidth=2.0,
            label="Meter-loss model",
        )
        density_axis.set_xlim(0, upper)
        density_axis.set_xlabel("Absolute error |Prediction - GT| (m)")
        density_axis.set_ylabel("Probability density")
        density_axis.set_title(f"{case.short_name}: absolute-error distribution")
        density_axis.grid(True, alpha=0.25)
        density_axis.legend(loc="upper right")

        norm_stats = stats[case.river]["normalized"]
        meter_stats = stats[case.river]["meter"]
        density_axis.text(
            0.98,
            0.72,
            "Normalized: " + metric_text(norm_stats) + "\n"
            "Meter:       " + metric_text(meter_stats),
            transform=density_axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

        # Right: empirical absolute-error CDF.
        cdf_axis = axes[row_index, 1]
        x_norm_cdf, y_norm_cdf = empirical_cdf(normalized_abs)
        x_meter_cdf, y_meter_cdf = empirical_cdf(meter_abs)

        cdf_axis.plot(
            x_norm_cdf,
            y_norm_cdf,
            color=NORMALIZED_COLOR,
            linewidth=2.0,
            label="Normalized-loss model",
        )
        cdf_axis.plot(
            x_meter_cdf,
            y_meter_cdf,
            color=METER_COLOR,
            linewidth=2.0,
            label="Meter-loss model",
        )
        cdf_axis.set_xlim(0, upper)
        cdf_axis.set_ylim(0, 1)
        cdf_axis.set_xlabel("Absolute error |Prediction - GT| (m)")
        cdf_axis.set_ylabel("Cumulative fraction of pixels")
        cdf_axis.set_title(f"{case.short_name}: absolute-error CDF")
        cdf_axis.grid(True, alpha=0.25)
        cdf_axis.legend(loc="lower right")

    fig.suptitle(
        "Full-river error comparison: normalized-loss vs meter-loss models\n"
        "Unique geospatial Core-Loss pixels; signed error = Prediction - GT",
        fontsize=16,
    )
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_signed_distribution(
    cases: Sequence[Case],
    data: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
    display_percentile: float,
) -> None:
    fig, axes = plt.subplots(
        len(cases),
        1,
        figsize=(10, 3.6 * len(cases)),
        constrained_layout=True,
    )
    if len(cases) == 1:
        axes = [axes]

    for axis, case in zip(axes, cases):
        normalized = data[case.river]["normalized"]
        meter = data[case.river]["meter"]
        lower, upper = signed_limits(
            normalized,
            meter,
            percentile=display_percentile,
        )
        bins = np.linspace(lower, upper, 241)
        x_norm, y_norm = histogram_density(normalized, bins)
        x_meter, y_meter = histogram_density(meter, bins)

        axis.plot(
            x_norm,
            y_norm,
            color=NORMALIZED_COLOR,
            linewidth=2.0,
            label="Normalized-loss model",
        )
        axis.plot(
            x_meter,
            y_meter,
            color=METER_COLOR,
            linewidth=2.0,
            label="Meter-loss model",
        )
        axis.axvline(0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xlim(lower, upper)
        axis.set_title(case.short_name)
        axis.set_xlabel("Signed error (Prediction - GT) (m)")
        axis.set_ylabel("Probability density")
        axis.grid(True, alpha=0.25)
        axis.legend()

    fig.suptitle(
        "Full-river signed-error distributions\n"
        "Unique geospatial Core-Loss pixels",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--norm_root", type=Path, default=DEFAULT_NORM_ROOT)
    parser.add_argument("--meter_root", type=Path, default=DEFAULT_METER_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--case_json", type=Path, default=None)
    parser.add_argument("--nodata", type=float, default=-999999.0)
    parser.add_argument("--nodata_threshold", type=float, default=-9999.0)
    parser.add_argument("--progress_every", type=int, default=200)
    parser.add_argument("--display_percentile", type=float, default=99.5)
    parser.add_argument("--overwrite_cache", action="store_true")
    args = parser.parse_args()

    if not 50 < args.display_percentile < 100:
        raise ValueError("--display_percentile must be between 50 and 100.")

    if args.case_json is None:
        cases = [Case(**item) for item in DEFAULT_CASES]
    else:
        cases = [
            Case(**item)
            for item in json.loads(args.case_json.read_text())
        ]

    output_dir = args.out_dir
    figure_dir = output_dir / "figures"
    cache_dir = output_dir / "cache_unique_error_arrays"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_data: Dict[str, Dict[str, np.ndarray]] = {}
    all_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for case in cases:
        print("=" * 90)
        print(f"[CASE] {case.river}")

        norm_error_dir = args.norm_root / case.norm_experiment / case.river
        meter_error_dir = args.meter_root / case.meter_experiment / case.river

        if not norm_error_dir.is_dir():
            raise FileNotFoundError(norm_error_dir)
        if not meter_error_dir.is_dir():
            raise FileNotFoundError(meter_error_dir)

        norm_error_summary = read_json(norm_error_dir / "F020_summary.json")
        meter_error_summary = read_json(meter_error_dir / "F062_summary.json")

        norm_prediction_dir = Path(norm_error_summary["source_f010_dir"])
        meter_prediction_dir = Path(meter_error_summary["source_f060_dir"])

        normalized_error = collect_or_load(
            cache_path=cache_dir / f"{case.river}__normalized_unique_error.npz",
            source_dir=norm_prediction_dir,
            summary_name="F010_summary.json",
            manifest_name="F010_tileavg_prediction_manifest.csv",
            nodata=args.nodata,
            nodata_threshold=args.nodata_threshold,
            progress_every=args.progress_every,
            overwrite_cache=args.overwrite_cache,
        )

        meter_error = collect_or_load(
            cache_path=cache_dir / f"{case.river}__meter_unique_error.npz",
            source_dir=meter_prediction_dir,
            summary_name="F060_summary.json",
            manifest_name="F060_tileavg_prediction_manifest.csv",
            nodata=args.nodata,
            nodata_threshold=args.nodata_threshold,
            progress_every=args.progress_every,
            overwrite_cache=args.overwrite_cache,
        )

        all_data[case.river] = {
            "normalized": normalized_error,
            "meter": meter_error,
        }

        normalized_stats = stats_from_error(normalized_error)
        meter_stats = stats_from_error(meter_error)
        all_stats[case.river] = {
            "normalized": normalized_stats,
            "meter": meter_stats,
        }

        normalized_row = {
            "river": case.river,
            "method": "normalized_loss",
            **normalized_stats,
        }
        meter_row = {
            "river": case.river,
            "method": "meter_loss",
            **meter_stats,
        }
        delta_row = {
            "river": case.river,
            "method": "meter_minus_normalized",
            "delta_rmse_m": meter_stats["rmse_m"] - normalized_stats["rmse_m"],
            "delta_mae_m": meter_stats["mae_m"] - normalized_stats["mae_m"],
            "delta_bias_m": meter_stats["bias_m"] - normalized_stats["bias_m"],
            "delta_median_abs_error_m": (
                meter_stats["median_abs_error_m"]
                - normalized_stats["median_abs_error_m"]
            ),
            "delta_p90_abs_error_m": (
                meter_stats["p90_abs_error_m"]
                - normalized_stats["p90_abs_error_m"]
            ),
            "delta_p95_abs_error_m": (
                meter_stats["p95_abs_error_m"]
                - normalized_stats["p95_abs_error_m"]
            ),
            "delta_p99_abs_error_m": (
                meter_stats["p99_abs_error_m"]
                - normalized_stats["p99_abs_error_m"]
            ),
            "mae_improvement_percent": (
                100.0
                * (normalized_stats["mae_m"] - meter_stats["mae_m"])
                / normalized_stats["mae_m"]
            ),
            "rmse_improvement_percent": (
                100.0
                * (normalized_stats["rmse_m"] - meter_stats["rmse_m"])
                / normalized_stats["rmse_m"]
            ),
        }

        summary_rows.extend([normalized_row, meter_row, delta_row])
        write_csv(
            output_dir / f"F065_summary_{case.river}.csv",
            [normalized_row, meter_row, delta_row],
        )

        print(
            "[NORMALIZED] "
            f"n={normalized_stats['n_pixels']:,} "
            f"RMSE={normalized_stats['rmse_m']:.4f} m "
            f"MAE={normalized_stats['mae_m']:.4f} m "
            f"Bias={normalized_stats['bias_m']:.4f} m"
        )
        print(
            "[METER]      "
            f"n={meter_stats['n_pixels']:,} "
            f"RMSE={meter_stats['rmse_m']:.4f} m "
            f"MAE={meter_stats['mae_m']:.4f} m "
            f"Bias={meter_stats['bias_m']:.4f} m"
        )
        print(
            "[DELTA meter-normalized] "
            f"RMSE={delta_row['delta_rmse_m']:.4f} m "
            f"MAE={delta_row['delta_mae_m']:.4f} m"
        )

    write_csv(
        output_dir / "F065_error_distribution_summary_all.csv",
        summary_rows,
    )

    publication_png = (
        figure_dir
        / "F065_abs_error_distribution_and_cdf_3x2_norm_vs_meter.png"
    )
    publication_pdf = (
        figure_dir
        / "F065_abs_error_distribution_and_cdf_3x2_norm_vs_meter.pdf"
    )
    signed_png = figure_dir / "F065_signed_error_distribution_norm_vs_meter.png"

    plot_publication_3x2(
        cases=cases,
        data=all_data,
        stats=all_stats,
        output_png=publication_png,
        output_pdf=publication_pdf,
        display_percentile=args.display_percentile,
    )
    plot_signed_distribution(
        cases=cases,
        data=all_data,
        output_path=signed_png,
        display_percentile=args.display_percentile,
    )

    manifest = {
        "normalized_loss_root": str(args.norm_root),
        "meter_loss_root": str(args.meter_root),
        "output_dir": str(output_dir),
        "comparison_mask": (
            "Core_Loss_Mask_Pixel AND valid_GT AND valid_prediction"
        ),
        "error_definition": "Prediction - GT in meters",
        "comparison_scope": (
            "Unique geospatial full-river pixels; each final overlap-averaged "
            "pixel is counted once for both experiments."
        ),
        "display_percentile": args.display_percentile,
        "cases": [case.__dict__ for case in cases],
        "outputs": {
            "summary_csv": str(
                output_dir / "F065_error_distribution_summary_all.csv"
            ),
            "publication_3x2_png": str(publication_png),
            "publication_3x2_pdf": str(publication_pdf),
            "signed_error_png": str(signed_png),
            "cache_dir": str(cache_dir),
        },
    }
    (output_dir / "F065_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print("=" * 90)
    print("[DONE] F065 normalized-loss versus meter-loss distribution analysis")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
