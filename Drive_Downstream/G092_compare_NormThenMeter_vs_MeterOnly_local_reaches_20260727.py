#!/usr/bin/env python3
"""G092: comprehensive native-resolution comparison of two full-river MAE models.

Models
------
A. Normalized-loss Stage 1 -> meter-MAE Stage 2
B. Meter-MAE only

The script consumes the validated sparse overlap-averaged F060 prediction
products and F062 error summaries for both model families.  It then:

1. recomputes exact COMMON-footprint unique-geospatial metrics for both models;
2. pairs the same E001/F060 local tiles and ranks local reaches by their mean
   difficulty across both models;
3. selects spatially separated best, middle, and worst reaches per river;
4. writes native 1 m GeoTIFF subsets and detailed comparison figures;
5. writes a self-contained local HTML report and an optional ZIP package.

Fair comparison mask
--------------------
    Core_Loss_Mask_Pixel AND valid_GT AND valid_prediction_A
    AND valid_prediction_B

Display policy
--------------
The local figures read the original native E001/F060 GeoTIFF tiles directly.
They do NOT reproject to EPSG:3857 and do NOT bilinearly resample errors.
The primary signed-error panels use the actual local maximum absolute error as
one shared symmetric scale, so no error value is hidden by percentile clipping.
A separate robust-detail figure is also produced and explicitly reports its
saturated-pixel count.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


NODATA_DEFAULT = -999999.0


@dataclass(frozen=True)
class SimpleAffine:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


@dataclass(frozen=True)
class Case:
    preset: str
    short_name: str
    river: str
    norm_meter_experiment: str
    meter_only_experiment: str


DEFAULT_CASES = (
    Case(
        preset="CA",
        short_name="CA Klamath",
        river="CA_KlamathRiver_TopoBathy_2018_D18",
        norm_meter_experiment="holdout_CA_D005Stage2MeterMAE_FromNorm_D001NoDataSafe",
        meter_only_experiment="holdout_CA_D003MeterMAE_BaselineEval_D001NoDataSafe",
    ),
    Case(
        preset="CO",
        short_name="CO Upper Colorado",
        river="CO_UpperColorado_Topobathy_1_2020",
        norm_meter_experiment="holdout_CO_D005Stage2MeterMAE_FromNorm_D001NoDataSafe",
        meter_only_experiment="holdout_CO_D003MeterMAE_BaselineEval_D001NoDataSafe",
    ),
    Case(
        preset="Santiam",
        short_name="OR Santiam",
        river="OR_SantiamRiverTB_Topobathy_1_D23",
        norm_meter_experiment="holdout_Santiam_D005Stage2MeterMAE_FromNorm_D001NoDataSafe",
        meter_only_experiment="holdout_Santiam_D003MeterMAE_BaselineEval_D001NoDataSafe",
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare normalized->meter and meter-only full-river predictions at native local resolution.",
    )
    p.add_argument(
        "--norm_meter_pred_root",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "FullRiver_Predictions_G001_NormThenMeter_D001NoDataSafe"
        ),
    )
    p.add_argument(
        "--norm_meter_error_root",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "FullRiver_GT_Error_G002_NormThenMeter_D001NoDataSafe"
        ),
    )
    p.add_argument(
        "--meter_only_pred_root",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "FullRiver_Predictions_F060_TileAvgVRT_D003MeterMAE_BaselineEval_D001NoDataSafe"
        ),
    )
    p.add_argument(
        "--meter_only_error_root",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "FullRiver_GT_Error_F062_UniquePixel_D003MeterMAE_BaselineEval_D001NoDataSafe"
        ),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "FullRiver_Analysis_G004_NormThenMeter_vs_MeterOnly_D001NoDataSafe"
        ),
    )
    p.add_argument("--min_common_pixels", type=int, default=512)
    p.add_argument(
        "--min_selected_center_distance",
        type=float,
        default=400.0,
        help="Minimum source-CRS map-unit distance among best/middle/worst selected tile centres.",
    )
    p.add_argument("--robust_error_percentile", type=float, default=98.0)
    p.add_argument("--nodata", type=float, default=NODATA_DEFAULT)
    p.add_argument("--nodata_threshold", type=float, default=-9999.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_zip", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _tag_value(tags, code_or_name, default=None):
    tag = tags.get(code_or_name)
    if tag is None:
        return default
    return tag.value


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


def _norm_tag_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_norm_tag_value(v) for v in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_norm_tag_value(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _geo_tags(tags) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for code in (34735, 34736, 34737):
        value = _tag_value(tags, code, None)
        if value is not None:
            result[str(code)] = _norm_tag_value(value)
    return result


def _transform(tags) -> SimpleAffine:
    scale = _tag_value(tags, 33550, None) or _tag_value(tags, "ModelPixelScaleTag", None)
    tie = _tag_value(tags, 33922, None) or _tag_value(tags, "ModelTiepointTag", None)
    matrix = _tag_value(tags, 34264, None) or _tag_value(tags, "ModelTransformationTag", None)
    if scale is not None and tie is not None:
        scale = tuple(float(x) for x in scale)
        tie = tuple(float(x) for x in tie)
        sx, sy = abs(scale[0]), abs(scale[1])
        c = tie[3] - tie[0] * sx
        f = tie[4] + tie[1] * sy
        return SimpleAffine(sx, 0.0, c, 0.0, -sy, f)
    if matrix is not None:
        m = tuple(float(x) for x in matrix)
        return SimpleAffine(m[0], m[1], m[3], m[4], m[5], m[7])
    raise RuntimeError("Missing GeoTIFF georeference tags")


def read_one(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        tags = page.tags
        crs_wkt_raw = _tag_value(tags, 34737, "")
        if isinstance(crs_wkt_raw, bytes):
            crs_wkt = crs_wkt_raw.decode("utf-8", errors="ignore").strip("\x00")
        else:
            crs_wkt = str(crs_wkt_raw).strip("\x00") if crs_wkt_raw else ""
        meta = {
            "transform": _transform(tags),
            "nodata": _parse_nodata(tags),
            "crs_tags": _geo_tags(tags),
            "crs_wkt": crs_wkt,
            "height": int(arr.shape[0]),
            "width": int(arr.shape[1]),
        }
    return arr, meta


def _extratags(meta: Dict[str, Any], nodata: float):
    t: SimpleAffine = meta["transform"]
    tags = [
        (33550, "d", 3, (abs(t.a), abs(t.e), 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, t.c, t.f, 0.0), False),
    ]
    crs_tags = meta.get("crs_tags", {})
    if "34735" in crs_tags:
        v = tuple(int(x) for x in np.asarray(crs_tags["34735"]).ravel())
        tags.append((34735, "H", len(v), v, False))
    if "34736" in crs_tags:
        v = tuple(float(x) for x in np.asarray(crs_tags["34736"]).ravel())
        tags.append((34736, "d", len(v), v, False))
    if "34737" in crs_tags:
        v = crs_tags["34737"]
        if isinstance(v, (list, tuple)):
            v = "".join(str(x) for x in v)
        else:
            v = str(v)
        if not v.endswith("\x00"):
            v += "\x00"
        tags.append((34737, "s", len(v), v, False))
    nd = str(nodata)
    if not nd.endswith("\x00"):
        nd += "\x00"
    tags.append((42113, "s", len(nd), nd, False))
    return tags


def write_tif(path: Path, arr: np.ndarray, meta: Dict[str, Any], nodata: float, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.asarray(arr).astype(dtype, copy=False)
    tifffile.imwrite(
        str(path),
        out,
        dtype=out.dtype,
        photometric="minisblack",
        metadata=None,
        extratags=_extratags(meta, nodata),
    )
    t: SimpleAffine = meta["transform"]
    x_center = t.c + t.a / 2.0
    y_center = t.f + t.e / 2.0
    path.with_suffix(".tfw").write_text(
        f"{t.a:.12f}\n{t.d:.12f}\n{t.b:.12f}\n{t.e:.12f}\n{x_center:.12f}\n{y_center:.12f}\n"
    )
    if meta.get("crs_wkt"):
        path.with_suffix(".prj").write_text(meta["crs_wkt"])


def valid_dem(arr: np.ndarray, nodata: float, threshold: float, source_nodata: Optional[float]) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(a) & (a > threshold) & (a != nodata)
    if source_nodata is not None and math.isfinite(source_nodata) and abs(source_nodata) > 1e-100:
        valid &= a != source_nodata
    return valid


def valid_pred(arr: np.ndarray, nodata: float, source_nodata: Optional[float]) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(a) & (a != nodata)
    if source_nodata is not None and math.isfinite(source_nodata) and abs(source_nodata) > 1e-100:
        valid &= a != source_nodata
    return valid


def error_stats(error: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(error, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "n_pixels": 0,
            "mae_m": np.nan,
            "rmse_m": np.nan,
            "bias_m": np.nan,
            "median_abs_error_m": np.nan,
            "p90_abs_error_m": np.nan,
            "p95_abs_error_m": np.nan,
            "p99_abs_error_m": np.nan,
            "max_abs_error_m": np.nan,
            "min_signed_error_m": np.nan,
            "max_signed_error_m": np.nan,
        }
    absolute = np.abs(values)
    return {
        "n_pixels": int(values.size),
        "mae_m": float(absolute.mean()),
        "rmse_m": float(np.sqrt(np.mean(values ** 2))),
        "bias_m": float(values.mean()),
        "median_abs_error_m": float(np.median(absolute)),
        "p90_abs_error_m": float(np.percentile(absolute, 90)),
        "p95_abs_error_m": float(np.percentile(absolute, 95)),
        "p99_abs_error_m": float(np.percentile(absolute, 99)),
        "max_abs_error_m": float(absolute.max()),
        "min_signed_error_m": float(values.min()),
        "max_signed_error_m": float(values.max()),
    }


def center_xy(meta: Dict[str, Any], shape: Tuple[int, int]) -> Tuple[float, float]:
    h, w = shape
    t: SimpleAffine = meta["transform"]
    col = w / 2.0
    row = h / 2.0
    return t.c + t.a * col + t.b * row, t.f + t.d * col + t.e * row


def metric_prefix(prefix: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def find_paths(pred_root: Path, error_root: Path, experiment: str, river: str) -> Dict[str, Path]:
    pred_dir = pred_root / experiment / river
    error_dir = error_root / experiment / river
    paths = {
        "pred_dir": pred_dir,
        "error_dir": error_dir,
        "manifest": pred_dir / "F060_tileavg_prediction_manifest.csv",
        "pred_summary": pred_dir / "F060_summary.json",
        "error_summary": error_dir / "F062_summary.json",
    }
    for label, path in paths.items():
        if label.endswith("dir"):
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)
    return paths


def spatially_distinct_order(
    candidates: Sequence[Dict[str, Any]],
    targets: Sequence[Tuple[str, float]],
    min_distance: float,
) -> List[Tuple[str, Dict[str, Any]]]:
    selected: List[Tuple[str, Dict[str, Any]]] = []
    used_keys = set()

    for label, target in targets:
        ranked = sorted(
            candidates,
            key=lambda row: abs(float(row["difficulty_mean_mae_m"]) - target),
        )
        choice = None
        for row in ranked:
            if row["key"] in used_keys:
                continue
            if all(
                math.hypot(
                    float(row["center_x"]) - float(old["center_x"]),
                    float(row["center_y"]) - float(old["center_y"]),
                ) >= min_distance
                for _, old in selected
            ):
                choice = row
                break
        if choice is None:
            choice = next((r for r in ranked if r["key"] not in used_keys), ranked[0])
        selected.append((label, choice))
        used_keys.add(choice["key"])
    return selected


def select_best_middle_worst(candidates: Sequence[Dict[str, Any]], min_distance: float):
    scores = np.asarray([float(r["difficulty_mean_mae_m"]) for r in candidates], dtype=np.float64)
    return spatially_distinct_order(
        candidates,
        (
            ("best", float(np.min(scores))),
            ("worst", float(np.max(scores))),
            ("middle", float(np.median(scores))),
        ),
        min_distance,
    )


def aggregate_unique(keys: np.ndarray, err_a: np.ndarray, err_b: np.ndarray):
    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    a_sorted = err_a[order].astype(np.float64, copy=False)
    b_sorted = err_b[order].astype(np.float64, copy=False)
    unique_keys, starts = np.unique(keys_sorted, return_index=True)
    counts = np.diff(np.r_[starts, keys_sorted.size]).astype(np.int64)
    a_unique = np.add.reduceat(a_sorted, starts) / counts
    b_unique = np.add.reduceat(b_sorted, starts) / counts
    return unique_keys, a_unique, b_unique, counts


def safe_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def global_summary_row(case: Case, stats_a: Dict[str, Any], stats_b: Dict[str, Any], checkpoint_a: str, checkpoint_b: str):
    mae_a = float(stats_a["mae_m"])
    mae_b = float(stats_b["mae_m"])
    rmse_a = float(stats_a["rmse_m"])
    rmse_b = float(stats_b["rmse_m"])
    return {
        "preset": case.preset,
        "river": case.river,
        "short_name": case.short_name,
        "comparison_footprint": "common unique geospatial pixels for both models",
        "common_n_pixels": int(stats_a["n_pixels"]),
        "norm_then_meter_mae_m": mae_a,
        "meter_only_mae_m": mae_b,
        "norm_then_meter_rmse_m": rmse_a,
        "meter_only_rmse_m": rmse_b,
        "norm_then_meter_bias_m": stats_a["bias_m"],
        "meter_only_bias_m": stats_b["bias_m"],
        "norm_then_meter_p95_abs_error_m": stats_a["p95_abs_error_m"],
        "meter_only_p95_abs_error_m": stats_b["p95_abs_error_m"],
        "norm_then_meter_max_abs_error_m": stats_a["max_abs_error_m"],
        "meter_only_max_abs_error_m": stats_b["max_abs_error_m"],
        "norm_then_meter_mae_improvement_vs_meter_only_pct": 100.0 * (mae_b - mae_a) / mae_b if mae_b else None,
        "norm_then_meter_rmse_improvement_vs_meter_only_pct": 100.0 * (rmse_b - rmse_a) / rmse_b if rmse_b else None,
        "winner_by_mae": "normalized_then_meter" if mae_a < mae_b else ("meter_only" if mae_b < mae_a else "tie"),
        "norm_then_meter_checkpoint": checkpoint_a,
        "meter_only_checkpoint": checkpoint_b,
    }


def load_region_arrays(row: Dict[str, Any], args: argparse.Namespace):
    gt, gt_meta = read_one(Path(row["tile_path"]))
    pred_a, pred_a_meta = read_one(Path(row["norm_then_meter_pred_tile_path"]))
    pred_b, pred_b_meta = read_one(Path(row["meter_only_pred_tile_path"]))
    hidden, hidden_meta = read_one(Path(row["hidden_path"]))
    core_loss, core_meta = read_one(Path(row["core_loss_path"]))

    gt = gt.astype(np.float32, copy=False)
    pred_a = pred_a.astype(np.float32, copy=False)
    pred_b = pred_b.astype(np.float32, copy=False)
    hidden_bool = np.isfinite(hidden) & (hidden.astype(np.float32) > 0.5) & (hidden.astype(np.float32) < 255)
    core_bool = np.isfinite(core_loss) & (core_loss.astype(np.float32) > 0.5) & (core_loss.astype(np.float32) < 255)
    common = (
        core_bool
        & valid_dem(gt, args.nodata, args.nodata_threshold, gt_meta.get("nodata"))
        & valid_pred(pred_a, args.nodata, pred_a_meta.get("nodata"))
        & valid_pred(pred_b, args.nodata, pred_b_meta.get("nodata"))
    )
    err_a = np.where(common, pred_a - gt, np.nan)
    err_b = np.where(common, pred_b - gt, np.nan)
    return {
        "gt": gt,
        "pred_a": pred_a,
        "pred_b": pred_b,
        "hidden": hidden_bool,
        "core": core_bool,
        "common": common,
        "err_a": err_a,
        "err_b": err_b,
        "meta": gt_meta,
    }


def finite_on(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)[mask]
    return values[np.isfinite(values)]


def add_mask_contour(ax, mask: np.ndarray) -> None:
    if np.any(mask) and np.any(~mask):
        ax.contour(mask.astype(np.uint8), levels=[0.5], linewidths=0.7)


def plot_region_comparison(region_dir: Path, arrays: Dict[str, Any], title: str, robust_pct: float) -> Dict[str, Any]:
    common = arrays["common"]
    gt = arrays["gt"]
    pred_a = arrays["pred_a"]
    pred_b = arrays["pred_b"]
    err_a = arrays["err_a"]
    err_b = arrays["err_b"]
    hidden = arrays["hidden"]

    gt_show = np.where(common, gt, np.nan)
    pred_a_show = np.where(common, pred_a, np.nan)
    pred_b_show = np.where(common, pred_b, np.nan)
    input_show = gt.astype(np.float64).copy()
    input_valid = np.isfinite(input_show) & (input_show > -9999)
    input_show[~input_valid | hidden] = np.nan

    elev_values = np.concatenate([
        finite_on(gt, common),
        finite_on(pred_a, common),
        finite_on(pred_b, common),
    ])
    elev_min = float(np.min(elev_values))
    elev_max = float(np.max(elev_values))
    if elev_max <= elev_min:
        elev_max = elev_min + 1.0

    error_values = np.concatenate([finite_on(err_a, common), finite_on(err_b, common)])
    full_error_max = float(np.max(np.abs(error_values))) if error_values.size else 1.0
    full_error_max = max(full_error_max, 1e-6)
    robust_error_max = float(np.percentile(np.abs(error_values), robust_pct)) if error_values.size else full_error_max
    robust_error_max = max(robust_error_max, 1e-6)
    saturated_a = int(np.count_nonzero(np.abs(err_a[common]) > robust_error_max))
    saturated_b = int(np.count_nonzero(np.abs(err_b[common]) > robust_error_max))

    abs_difference = np.where(common, np.abs(err_a) - np.abs(err_b), np.nan)
    diff_values = finite_on(abs_difference, common)
    diff_max = max(float(np.max(np.abs(diff_values))) if diff_values.size else 1.0, 1e-6)

    # Primary figure: actual full local error range. No percentile clipping.
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    panels = [
        (input_show, "Visible model input", "terrain", elev_min, elev_max),
        (gt_show, "GT — common evaluation pixels", "terrain", elev_min, elev_max),
        (pred_a_show, "Normalized → meter", "terrain", elev_min, elev_max),
        (pred_b_show, "Meter-only", "terrain", elev_min, elev_max),
        (err_a, "Signed error: normalized → meter", "RdBu_r", -full_error_max, full_error_max),
        (err_b, "Signed error: meter-only", "RdBu_r", -full_error_max, full_error_max),
        (abs_difference, "|Error| difference: N→M minus meter-only", "RdBu_r", -diff_max, diff_max),
        (hidden.astype(float), "Strict Hidden Mask (1 = hidden)", "gray_r", 0.0, 1.0),
    ]
    for ax, (arr, label, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        add_mask_contour(ax, common)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        title + f"\nNative source grid; full signed-error range = ±{full_error_max:.3f} m (no clipping)",
        fontsize=13,
    )
    primary = region_dir / "G004_native_fullrange_comparison.png"
    fig.savefig(primary, dpi=180)
    plt.close(fig)

    # Robust-detail figure: useful for seeing small patterns, explicitly marked.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    detail_panels = [
        (err_a, "N→M signed error"),
        (err_b, "Meter-only signed error"),
        (abs_difference, "|Error| difference"),
    ]
    for idx, (ax, (arr, label)) in enumerate(zip(axes, detail_panels)):
        if idx < 2:
            lim = robust_error_max
        else:
            lim = float(np.percentile(np.abs(diff_values), robust_pct)) if diff_values.size else diff_max
            lim = max(lim, 1e-6)
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        add_mask_contour(ax, common)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"{title}\nRobust-detail scale = {robust_pct:g}th percentile; saturated pixels: "
        f"N→M={saturated_a}, meter-only={saturated_b}",
        fontsize=12,
    )
    robust = region_dir / "G004_native_robust_detail_comparison.png"
    fig.savefig(robust, dpi=180)
    plt.close(fig)

    # Local error distributions and empirical CDF.
    a_abs = np.abs(err_a[common].astype(np.float64))
    b_abs = np.abs(err_b[common].astype(np.float64))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    hist_max = max(float(np.percentile(np.r_[a_abs, b_abs], 99.5)), 1e-6)
    bins = np.linspace(0, hist_max, 100)
    axes[0].hist(a_abs, bins=bins, density=True, histtype="step", linewidth=2, label="Normalized → meter")
    axes[0].hist(b_abs, bins=bins, density=True, histtype="step", linewidth=2, label="Meter-only")
    axes[0].set_xlabel("Absolute error (m)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Local absolute-error density (x-axis to P99.5)")
    axes[0].legend()
    for values, label in ((a_abs, "Normalized → meter"), (b_abs, "Meter-only")):
        values = np.sort(values)
        y = np.arange(1, values.size + 1) / values.size
        axes[1].plot(values, y, linewidth=2, label=label)
    axes[1].set_xlabel("Absolute error (m)")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_xlim(0, hist_max)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Local absolute-error CDF (x-axis to P99.5)")
    axes[1].legend()
    dist = region_dir / "G004_local_error_distribution.png"
    fig.savefig(dist, dpi=180)
    plt.close(fig)

    return {
        "primary_png": str(primary),
        "robust_png": str(robust),
        "distribution_png": str(dist),
        "full_error_display_max_m": full_error_max,
        "robust_error_display_max_m": robust_error_max,
        "robust_saturated_pixels_norm_then_meter": saturated_a,
        "robust_saturated_pixels_meter_only": saturated_b,
    }


def write_region_rasters(region_dir: Path, arrays: Dict[str, Any], args: argparse.Namespace) -> Dict[str, str]:
    common = arrays["common"]
    meta = arrays["meta"]
    outputs: Dict[str, Tuple[np.ndarray, str, float]] = {}

    def masked(arr):
        out = np.full(arr.shape, args.nodata, dtype=np.float32)
        out[common] = np.asarray(arr, dtype=np.float32)[common]
        return out

    outputs["gt"] = (masked(arrays["gt"]), "float32", args.nodata)
    outputs["prediction_norm_then_meter"] = (masked(arrays["pred_a"]), "float32", args.nodata)
    outputs["prediction_meter_only"] = (masked(arrays["pred_b"]), "float32", args.nodata)
    outputs["error_norm_then_meter"] = (masked(arrays["err_a"]), "float32", args.nodata)
    outputs["error_meter_only"] = (masked(arrays["err_b"]), "float32", args.nodata)
    outputs["abs_error_difference_norm_then_meter_minus_meter_only"] = (
        masked(np.abs(arrays["err_a"]) - np.abs(arrays["err_b"])),
        "float32",
        args.nodata,
    )
    outputs["common_evaluation_mask"] = (common.astype(np.uint8), "uint8", 255)
    outputs["hidden_mask"] = (arrays["hidden"].astype(np.uint8), "uint8", 255)

    result: Dict[str, str] = {}
    for name, (arr, dtype, nodata) in outputs.items():
        path = region_dir / f"G004_{name}.tif"
        write_tif(path, arr, meta, nodata, dtype)
        result[name] = str(path)
    return result


def plot_candidate_summary(out_path: Path, rows: Sequence[Dict[str, Any]], selected: Sequence[Tuple[str, Dict[str, Any]]], title: str) -> None:
    a = np.asarray([float(r["norm_then_meter_mae_m"]) for r in rows])
    b = np.asarray([float(r["meter_only_mae_m"]) for r in rows])
    delta = a - b
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].scatter(b, a, s=14, alpha=0.55)
    lim = max(float(np.max(a)), float(np.max(b)), 1e-6)
    axes[0].plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    for label, row in selected:
        axes[0].scatter(float(row["meter_only_mae_m"]), float(row["norm_then_meter_mae_m"]), s=90, label=label)
    axes[0].set_xlabel("Meter-only local MAE (m)")
    axes[0].set_ylabel("Normalized → meter local MAE (m)")
    axes[0].set_title("Paired local reaches")
    axes[0].legend()
    axes[1].hist(delta, bins=50)
    axes[1].axvline(0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("MAE difference: normalized→meter − meter-only (m)")
    axes[1].set_ylabel("Candidate reach count")
    axes[1].set_title("Negative means normalized→meter is better")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def collect_case(case: Case, args: argparse.Namespace, out_dir: Path):
    a_paths = find_paths(args.norm_meter_pred_root, args.norm_meter_error_root, case.norm_meter_experiment, case.river)
    b_paths = find_paths(args.meter_only_pred_root, args.meter_only_error_root, case.meter_only_experiment, case.river)
    a_manifest = {row["key"]: row for row in read_csv(a_paths["manifest"])}
    b_manifest = {row["key"]: row for row in read_csv(b_paths["manifest"])}
    common_keys = sorted(set(a_manifest) & set(b_manifest))
    if not common_keys:
        raise RuntimeError(f"No paired F060 manifest keys for {case.river}")

    a_pred_summary = read_json(a_paths["pred_summary"])
    b_pred_summary = read_json(b_paths["pred_summary"])
    width_a = int(a_pred_summary["virtual_mosaic_width"])
    width_b = int(b_pred_summary["virtual_mosaic_width"])
    if width_a != width_b:
        raise RuntimeError(f"Virtual mosaic width differs for {case.river}: {width_a} vs {width_b}")

    candidate_rows: List[Dict[str, Any]] = []
    key_parts: List[np.ndarray] = []
    err_a_parts: List[np.ndarray] = []
    err_b_parts: List[np.ndarray] = []

    print(f"[CASE] {case.river}: paired manifest keys={len(common_keys):,}")
    for idx, key in enumerate(common_keys, start=1):
        ra = a_manifest[key]
        rb = b_manifest[key]
        tile_path = Path(ra["tile_path"])
        if Path(rb["tile_path"]).resolve() != tile_path.resolve():
            raise RuntimeError(f"GT tile mismatch for paired key {key}")

        gt, gt_meta = read_one(tile_path)
        pred_a, pred_a_meta = read_one(Path(ra["avg_pred_tile_path"]))
        pred_b, pred_b_meta = read_one(Path(rb["avg_pred_tile_path"]))
        core_loss, _ = read_one(Path(ra["core_loss_path"]))

        gt = gt.astype(np.float32, copy=False)
        pred_a = pred_a.astype(np.float32, copy=False)
        pred_b = pred_b.astype(np.float32, copy=False)
        common = (
            np.isfinite(core_loss)
            & (core_loss.astype(np.float32) > 0.5)
            & valid_dem(gt, args.nodata, args.nodata_threshold, gt_meta.get("nodata"))
            & valid_pred(pred_a, args.nodata, pred_a_meta.get("nodata"))
            & valid_pred(pred_b, args.nodata, pred_b_meta.get("nodata"))
        )
        n = int(common.sum())
        if n == 0:
            continue
        err_a = (pred_a - gt).astype(np.float32)
        err_b = (pred_b - gt).astype(np.float32)
        a_stats = error_stats(err_a[common])
        b_stats = error_stats(err_b[common])

        h, w = gt.shape
        row0_a = int(ra["mosaic_row0"])
        col0_a = int(ra["mosaic_col0"])
        row0_b = int(rb["mosaic_row0"])
        col0_b = int(rb["mosaic_col0"])
        if row0_a != row0_b or col0_a != col0_b:
            raise RuntimeError(f"Mosaic offset mismatch for {key}")
        local_flat = np.flatnonzero(common.ravel()).astype(np.int64)
        rr = local_flat // w
        cc = local_flat % w
        global_key = (np.int64(row0_a) + rr) * np.int64(width_a) + (np.int64(col0_a) + cc)
        key_parts.append(global_key)
        err_a_parts.append(err_a.ravel()[local_flat])
        err_b_parts.append(err_b.ravel()[local_flat])

        cx, cy = center_xy(gt_meta, gt.shape)
        mae_a = float(a_stats["mae_m"])
        mae_b = float(b_stats["mae_m"])
        candidate_rows.append({
            "preset": case.preset,
            "short_name": case.short_name,
            "river": case.river,
            "key": key,
            "tile_id": ra.get("tile_id", ""),
            "center_x": cx,
            "center_y": cy,
            "common_n_pixels": n,
            "difficulty_mean_mae_m": 0.5 * (mae_a + mae_b),
            "norm_then_meter_minus_meter_only_mae_m": mae_a - mae_b,
            "winner_by_local_mae": "normalized_then_meter" if mae_a < mae_b else ("meter_only" if mae_b < mae_a else "tie"),
            "tile_path": str(tile_path),
            "hidden_path": ra["hidden_path"],
            "core_loss_path": ra["core_loss_path"],
            "norm_then_meter_pred_tile_path": ra["avg_pred_tile_path"],
            "meter_only_pred_tile_path": rb["avg_pred_tile_path"],
            **metric_prefix("norm_then_meter", a_stats),
            **metric_prefix("meter_only", b_stats),
        })
        if idx == 1 or idx == len(common_keys) or idx % 200 == 0:
            print(f"  processed {idx:,}/{len(common_keys):,} paired tiles")

    eligible = [r for r in candidate_rows if int(r["common_n_pixels"]) >= args.min_common_pixels]
    if len(eligible) < 3:
        raise RuntimeError(
            f"Only {len(eligible)} paired candidate reaches have >= {args.min_common_pixels} common pixels for {case.river}"
        )

    all_keys = np.concatenate(key_parts).astype(np.int64, copy=False)
    all_a = np.concatenate(err_a_parts).astype(np.float32, copy=False)
    all_b = np.concatenate(err_b_parts).astype(np.float32, copy=False)
    unique_keys, unique_a, unique_b, duplicate_counts = aggregate_unique(all_keys, all_a, all_b)
    global_a = error_stats(unique_a)
    global_b = error_stats(unique_b)
    global_row = global_summary_row(
        case,
        global_a,
        global_b,
        str(a_pred_summary.get("checkpoint", "")),
        str(b_pred_summary.get("checkpoint", "")),
    )
    global_row["max_overlap_multiplicity_common"] = int(duplicate_counts.max()) if duplicate_counts.size else 0

    selected = select_best_middle_worst(eligible, args.min_selected_center_distance)
    river_out = out_dir / case.preset
    river_out.mkdir(parents=True, exist_ok=True)
    plot_candidate_summary(
        river_out / "G004_candidate_reach_summary.png",
        eligible,
        selected,
        f"{case.short_name}: local reach comparison",
    )

    selected_rows: List[Dict[str, Any]] = []
    for label, row in selected:
        region_dir = river_out / f"{label}_{row['key']}"
        region_dir.mkdir(parents=True, exist_ok=True)
        arrays = load_region_arrays(row, args)
        title = (
            f"{case.short_name} | {label.upper()} local reach | {row['key']} | "
            f"common pixels={int(row['common_n_pixels']):,}"
        )
        figures = plot_region_comparison(region_dir, arrays, title, args.robust_error_percentile)
        rasters = write_region_rasters(region_dir, arrays, args)
        selected_row = {
            **row,
            "selection_class": label,
            **{k: str(Path(v).relative_to(out_dir)) for k, v in figures.items() if k.endswith("_png")},
            **{k: v for k, v in figures.items() if not k.endswith("_png")},
            "region_dir": str(region_dir.relative_to(out_dir)),
        }
        for key, value in rasters.items():
            selected_row[f"raster_{key}"] = str(Path(value).relative_to(out_dir))
        (region_dir / "G004_region_metrics.json").write_text(json.dumps(selected_row, indent=2))
        selected_rows.append(selected_row)

    return global_row, candidate_rows, selected_rows


def fmt(value: Any, digits: int = 3) -> str:
    x = safe_float(value)
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def relative_link(path: str) -> str:
    return html.escape(path.replace("\\", "/"))


def build_html(out_dir: Path, global_rows: Sequence[Dict[str, Any]], selected_rows: Sequence[Dict[str, Any]]) -> None:
    by_preset: Dict[str, List[Dict[str, Any]]] = {}
    for row in selected_rows:
        by_preset.setdefault(str(row["preset"]), []).append(row)

    global_table = []
    for row in global_rows:
        global_table.append(
            "<tr>"
            f"<td>{html.escape(str(row['short_name']))}</td>"
            f"<td>{int(row['common_n_pixels']):,}</td>"
            f"<td>{fmt(row['norm_then_meter_mae_m'])}</td>"
            f"<td>{fmt(row['meter_only_mae_m'])}</td>"
            f"<td>{fmt(row['norm_then_meter_rmse_m'])}</td>"
            f"<td>{fmt(row['meter_only_rmse_m'])}</td>"
            f"<td>{fmt(row['norm_then_meter_mae_improvement_vs_meter_only_pct'], 2)}%</td>"
            f"<td>{html.escape(str(row['winner_by_mae']))}</td>"
            "</tr>"
        )

    sections = []
    for global_row in global_rows:
        preset = str(global_row["preset"])
        rows = sorted(by_preset.get(preset, []), key=lambda r: ("best", "middle", "worst").index(r["selection_class"]))
        cards = []
        for row in rows:
            cards.append(f"""
            <article class="card">
              <h3>{html.escape(row['selection_class'].upper())}: {html.escape(row['key'])}</h3>
              <div class="metrics">
                <span>Common pixels: <b>{int(row['common_n_pixels']):,}</b></span>
                <span>N→M MAE: <b>{fmt(row['norm_then_meter_mae_m'])} m</b></span>
                <span>Meter-only MAE: <b>{fmt(row['meter_only_mae_m'])} m</b></span>
                <span>N→M RMSE: <b>{fmt(row['norm_then_meter_rmse_m'])} m</b></span>
                <span>Meter-only RMSE: <b>{fmt(row['meter_only_rmse_m'])} m</b></span>
                <span>Local winner: <b>{html.escape(row['winner_by_local_mae'])}</b></span>
                <span>Actual local max |error| scale: <b>±{fmt(row['full_error_display_max_m'])} m</b></span>
              </div>
              <p class="note">The first figure uses the full local error range with no percentile clipping. The second figure deliberately uses a robust scale for pattern visibility and reports how many pixels saturate.</p>
              <a href="{relative_link(row['primary_png'])}"><img src="{relative_link(row['primary_png'])}" loading="lazy"></a>
              <div class="two">
                <a href="{relative_link(row['robust_png'])}"><img src="{relative_link(row['robust_png'])}" loading="lazy"></a>
                <a href="{relative_link(row['distribution_png'])}"><img src="{relative_link(row['distribution_png'])}" loading="lazy"></a>
              </div>
              <details><summary>Native GeoTIFF outputs</summary><ul>
                <li><a href="{relative_link(row['raster_gt'])}">GT</a></li>
                <li><a href="{relative_link(row['raster_prediction_norm_then_meter'])}">Normalized → meter prediction</a></li>
                <li><a href="{relative_link(row['raster_prediction_meter_only'])}">Meter-only prediction</a></li>
                <li><a href="{relative_link(row['raster_error_norm_then_meter'])}">Normalized → meter signed error</a></li>
                <li><a href="{relative_link(row['raster_error_meter_only'])}">Meter-only signed error</a></li>
                <li><a href="{relative_link(row['raster_abs_error_difference_norm_then_meter_minus_meter_only'])}">Absolute-error difference</a></li>
              </ul></details>
            </article>
            """)
        summary_png = f"{preset}/G004_candidate_reach_summary.png"
        sections.append(f"""
        <section>
          <h2>{html.escape(str(global_row['short_name']))}</h2>
          <p>Common full-river MAE: normalized→meter <b>{fmt(global_row['norm_then_meter_mae_m'])} m</b>; meter-only <b>{fmt(global_row['meter_only_mae_m'])} m</b>. Winner: <b>{html.escape(str(global_row['winner_by_mae']))}</b>.</p>
          <a href="{relative_link(summary_png)}"><img class="summary" src="{relative_link(summary_png)}" loading="lazy"></a>
          {''.join(cards)}
        </section>
        """)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G004 Native Local-Reach MAE Comparison</title>
<style>
body{{font-family:Arial,sans-serif;margin:0;background:#f5f6f8;color:#18202a}} header,main{{max-width:1500px;margin:auto;padding:22px}} header{{background:white;border-bottom:1px solid #ddd;max-width:none}} h1{{margin:0 0 8px}} section{{margin:24px 0}} table{{border-collapse:collapse;width:100%;background:white}} th,td{{border:1px solid #d7dce2;padding:8px;text-align:right}} th:first-child,td:first-child{{text-align:left}} .card{{background:white;border:1px solid #d7dce2;border-radius:8px;padding:16px;margin:18px 0;box-shadow:0 2px 8px #0001}} img{{width:100%;height:auto;border:1px solid #ddd}} .summary{{max-width:1100px}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}} .metrics{{display:flex;flex-wrap:wrap;gap:10px 18px;background:#eef2f7;padding:10px;border-radius:6px}} .note{{color:#4b5563}} code{{background:#eef2f7;padding:2px 5px}} @media(max-width:900px){{.two{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>G004: normalized→meter vs meter-only</h1>
<p>Exact common-footprint full-river metrics plus native-resolution best/middle/worst local reaches. No EPSG:3857 reprojection or bilinear resampling is used in the local figures.</p>
<p>The previous full-river web dashboard used a robust percentile colour scale. Values beyond that range were colour-saturated for display, not truncated in the source error raster. Here the primary local error panels use the actual local maximum absolute error.</p></header>
<main>
<h2>Exact common unique-geospatial full-river comparison</h2>
<table><thead><tr><th>River</th><th>Pixels</th><th>N→M MAE</th><th>Meter-only MAE</th><th>N→M RMSE</th><th>Meter-only RMSE</th><th>N→M MAE improvement</th><th>Winner</th></tr></thead><tbody>{''.join(global_table)}</tbody></table>
<p>Downloads: <a href="G004_global_common_metrics.csv">global metrics CSV</a> · <a href="G004_all_candidate_reach_metrics.csv">all local candidates CSV</a> · <a href="G004_selected_reaches.csv">selected reaches CSV</a> · <a href="G004_analysis_manifest.json">manifest JSON</a></p>
{''.join(sections)}
</main></body></html>"""
    (out_dir / "G004_local_reach_dashboard.html").write_text(doc, encoding="utf-8")


def zip_output(out_dir: Path) -> Path:
    zip_path = out_dir.parent / f"{out_dir.name}_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=f"{out_dir.name}/{path.relative_to(out_dir)}")
    return zip_path


def main() -> None:
    args = parse_args()
    for path in (
        args.norm_meter_pred_root,
        args.norm_meter_error_root,
        args.meter_only_pred_root,
        args.meter_only_error_root,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise RuntimeError(f"Output is non-empty: {out_dir}. Use --overwrite or choose another folder.")
    out_dir.mkdir(parents=True, exist_ok=True)

    global_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []

    for case in DEFAULT_CASES:
        global_row, candidates, selected = collect_case(case, args, out_dir)
        global_rows.append(global_row)
        candidate_rows.extend(candidates)
        selected_rows.extend(selected)

    write_csv(out_dir / "G004_global_common_metrics.csv", global_rows)
    write_csv(out_dir / "G004_all_candidate_reach_metrics.csv", candidate_rows)
    write_csv(out_dir / "G004_selected_reaches.csv", selected_rows)
    build_html(out_dir, global_rows, selected_rows)

    manifest = {
        "generated_at_unix": time.time(),
        "generator": Path(__file__).name,
        "models": {
            "normalized_then_meter": {
                "prediction_root": str(args.norm_meter_pred_root.resolve()),
                "error_root": str(args.norm_meter_error_root.resolve()),
            },
            "meter_only": {
                "prediction_root": str(args.meter_only_pred_root.resolve()),
                "error_root": str(args.meter_only_error_root.resolve()),
            },
        },
        "comparison_mask": "Core_Loss_Mask_Pixel AND valid_GT AND valid_prediction_A AND valid_prediction_B",
        "global_metric_scope": "each common overlap-averaged geospatial pixel counted once",
        "local_candidate_unit": "paired native E001/F060 tile; 336x336 context with core-loss evaluation pixels",
        "local_selection": "spatially separated best, median-difficulty, and worst reaches ranked by mean MAE across both models",
        "display_policy": {
            "native_grid": True,
            "reprojected": False,
            "resampled": False,
            "primary_error_scale": "shared symmetric actual local max absolute error; no clipping",
            "secondary_error_scale": f"shared {args.robust_error_percentile:g}th percentile detail view with saturated counts reported",
        },
        "parameters": {
            "min_common_pixels": args.min_common_pixels,
            "min_selected_center_distance": args.min_selected_center_distance,
            "robust_error_percentile": args.robust_error_percentile,
            "nodata": args.nodata,
            "nodata_threshold": args.nodata_threshold,
        },
        "global_rows": global_rows,
        "selected_reaches": selected_rows,
    }
    (out_dir / "G004_analysis_manifest.json").write_text(json.dumps(manifest, indent=2))

    zip_path = None
    if not args.no_zip:
        zip_path = zip_output(out_dir)

    print("============================================================")
    print("DONE G004")
    print(f"OUT_DIR={out_dir}")
    print(f"HTML={out_dir / 'G004_local_reach_dashboard.html'}")
    if zip_path:
        print(f"ZIP={zip_path}")
    print("============================================================")


if __name__ == "__main__":
    main()
