#!/usr/bin/env python3
"""G094: exact common-footprint comparison of three downstream MAE models.

Models
------
1. NormOnly       : normalized-loss only
2. MeterOnly      : meter-MAE only
3. NormThenMeter  : normalized-loss stage followed by meter-MAE fine-tuning

This script intentionally does NOT build a full-river web map or HTML report.
It produces:

A. Full-river statistical comparison on the exact same common unique
   geospatial Core-Loss pixels for all three models.
B. Native-resolution local-region visualizations and GeoTIFF subsets.

Fair common mask
----------------
    Core_Loss_Mask_Pixel
    AND valid_GT
    AND valid_prediction_NormOnly
    AND valid_prediction_MeterOnly
    AND valid_prediction_NormThenMeter

Each overlap-averaged geospatial pixel is counted once in global metrics.

Local-region selection
----------------------
For every river, candidate units are paired native 336 x 336 E001 prediction tiles from F010/F060/F025
tile contexts.  Three spatially separated regions are selected:
    easy, typical, and hard,
ranked by the mean local MAE across all three models.

The primary local signed-error panels use the actual local maximum absolute
error shared by all three models.  No percentile clipping is used there.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tifffile


NODATA_DEFAULT = -999999.0
MODEL_ORDER = ("norm_only", "meter_only", "norm_then_meter")
MODEL_LABELS = {
    "norm_only": "Normalized only",
    "meter_only": "Meter only",
    "norm_then_meter": "Normalized → meter",
}
MODEL_COLORS = {
    "norm_only": "tab:blue",
    "meter_only": "tab:orange",
    "norm_then_meter": "tab:green",
}


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
    norm_only_experiment: str
    meter_only_experiment: str
    norm_then_meter_experiment: str


DEFAULT_CASES = (
    Case(
        preset="CA",
        short_name="CA Klamath",
        river="CA_KlamathRiver_TopoBathy_2018_D18",
        norm_only_experiment="holdout_CA_D001NoDataSafe",
        meter_only_experiment="holdout_CA_D003MeterMAE_BaselineEval_D001NoDataSafe",
        norm_then_meter_experiment="holdout_CA_D005Stage2MeterMAE_FromNorm_D001NoDataSafe",
    ),
    Case(
        preset="CO",
        short_name="CO Upper Colorado",
        river="CO_UpperColorado_Topobathy_1_2020",
        norm_only_experiment="holdout_CO_D001NoDataSafe",
        meter_only_experiment="holdout_CO_D003MeterMAE_BaselineEval_D001NoDataSafe",
        norm_then_meter_experiment="holdout_CO_D005Stage2MeterMAE_FromNorm_D001NoDataSafe",
    ),
    Case(
        preset="Santiam",
        short_name="OR Santiam",
        river="OR_SantiamRiverTB_Topobathy_1_D23",
        norm_only_experiment="holdout_Santiam_D001NoDataSafe",
        meter_only_experiment="holdout_Santiam_D003MeterMAE_BaselineEval_D001NoDataSafe",
        norm_then_meter_experiment="holdout_Santiam_D005Stage2MeterMAE_FromNorm_D001NoDataSafe",
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare NormOnly, MeterOnly, and NormThenMeter globally and in native local regions.",
    )
    p.add_argument(
        "--norm_only_pred_root",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "FullRiver_Predictions_F010_TileAvgVRT_D001NoDataSafe"
        ),
    )
    p.add_argument(
        "--norm_only_error_root",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "FullRiver_GT_Error_F020_TileVRT_D001NoDataSafe"
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
        "--norm_then_meter_pred_root",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "FullRiver_Predictions_G001_NormThenMeter_D001NoDataSafe"
        ),
    )
    p.add_argument(
        "--norm_then_meter_error_root",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "FullRiver_GT_Error_G002_NormThenMeter_D001NoDataSafe"
        ),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "FullRiver_Analysis_G094_ThreeModels_CommonFootprint_Local_D001NoDataSafe"
        ),
    )
    p.add_argument("--min_common_pixels", type=int, default=512)
    p.add_argument(
        "--min_selected_center_distance",
        type=float,
        default=400.0,
        help="Minimum source-CRS map-unit separation among selected local regions.",
    )
    p.add_argument(
        "--global_display_percentile",
        type=float,
        default=99.5,
        help="Only controls x-axis range in distribution plots; metrics use all pixels.",
    )
    p.add_argument("--nodata", type=float, default=NODATA_DEFAULT)
    p.add_argument("--nodata_threshold", type=float, default=-9999.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_zip", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _tag_value(tags, code_or_name, default=None):
    tag = tags.get(code_or_name)
    if tag is None:
        return default
    return getattr(tag, "value", tag)


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
        value = tuple(int(x) for x in np.asarray(crs_tags["34735"]).ravel())
        tags.append((34735, "H", len(value), value, False))
    if "34736" in crs_tags:
        value = tuple(float(x) for x in np.asarray(crs_tags["34736"]).ravel())
        tags.append((34736, "d", len(value), value, False))
    if "34737" in crs_tags:
        value = crs_tags["34737"]
        if isinstance(value, (list, tuple)):
            value = "".join(str(x) for x in value)
        else:
            value = str(value)
        if not value.endswith("\x00"):
            value += "\x00"
        tags.append((34737, "s", len(value), value, False))
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
    values = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(values) & (values > threshold) & (values != nodata)
    if source_nodata is not None and math.isfinite(source_nodata) and abs(source_nodata) > 1e-100:
        valid &= values != source_nodata
    return valid


def valid_pred(arr: np.ndarray, nodata: float, source_nodata: Optional[float]) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(values) & (values != nodata)
    if source_nodata is not None and math.isfinite(source_nodata) and abs(source_nodata) > 1e-100:
        valid &= values != source_nodata
    return valid


def error_stats(error: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(error, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n_pixels": 0,
            "mae_m": np.nan,
            "rmse_m": np.nan,
            "bias_m": np.nan,
            "median_signed_error_m": np.nan,
            "median_abs_error_m": np.nan,
            "p75_abs_error_m": np.nan,
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
        "median_signed_error_m": float(np.median(values)),
        "median_abs_error_m": float(np.median(absolute)),
        "p75_abs_error_m": float(np.percentile(absolute, 75)),
        "p90_abs_error_m": float(np.percentile(absolute, 90)),
        "p95_abs_error_m": float(np.percentile(absolute, 95)),
        "p99_abs_error_m": float(np.percentile(absolute, 99)),
        "max_abs_error_m": float(absolute.max()),
        "min_signed_error_m": float(values.min()),
        "max_signed_error_m": float(values.max()),
    }


def center_xy(meta: Dict[str, Any], shape: Tuple[int, int]) -> Tuple[float, float]:
    height, width = shape
    t: SimpleAffine = meta["transform"]
    col = width / 2.0
    row = height / 2.0
    return t.c + t.a * col + t.b * row, t.f + t.d * col + t.e * row


def manifest_key(row: Dict[str, str]) -> str:
    for field in ("key", "tile_key"):
        value = row.get(field, "").strip()
        if value:
            return value
    tile_id = row.get("tile_id", "").strip()
    if tile_id:
        return f"ID{tile_id}"
    match = re.search(r"_ID(\d+)", Path(row["tile_path"]).stem)
    if match:
        return f"ID{match.group(1)}"
    return Path(row["tile_path"]).stem


def index_manifest(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    indexed: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = manifest_key(row)
        if key in indexed:
            raise RuntimeError(f"Duplicate manifest key: {key}")
        indexed[key] = row
    return indexed


def resolve_mosaic_width(summary: Dict[str, Any]) -> int:
    for key in ("virtual_mosaic_width", "mosaic_width", "width"):
        value = summary.get(key)
        if value is not None:
            return int(value)
    raise KeyError("Could not determine virtual mosaic width from prediction summary.")


def model_paths(
    pred_root: Path,
    error_root: Path,
    experiment: str,
    river: str,
    pred_summary_name: str,
    manifest_name: str,
    error_summary_name: str,
) -> Dict[str, Path]:
    pred_dir = pred_root / experiment / river
    error_dir = error_root / experiment / river
    paths = {
        "pred_dir": pred_dir,
        "error_dir": error_dir,
        "pred_summary": pred_dir / pred_summary_name,
        "manifest": pred_dir / manifest_name,
        "error_summary": error_dir / error_summary_name,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def get_case_paths(case: Case, args: argparse.Namespace) -> Dict[str, Dict[str, Path]]:
    return {
        "norm_only": model_paths(
            args.norm_only_pred_root,
            args.norm_only_error_root,
            case.norm_only_experiment,
            case.river,
            "F010_summary.json",
            "F010_tileavg_prediction_manifest.csv",
            "F020_summary.json",
        ),
        "meter_only": model_paths(
            args.meter_only_pred_root,
            args.meter_only_error_root,
            case.meter_only_experiment,
            case.river,
            "F060_summary.json",
            "F060_tileavg_prediction_manifest.csv",
            "F062_summary.json",
        ),
        "norm_then_meter": model_paths(
            args.norm_then_meter_pred_root,
            args.norm_then_meter_error_root,
            case.norm_then_meter_experiment,
            case.river,
            "F025_summary.json",
            "F025_tileavg_prediction_manifest.csv",
            "F027_summary.json",
        ),
    }


def aggregate_unique_multi(keys: np.ndarray, errors: Dict[str, np.ndarray]):
    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    unique_keys, starts = np.unique(keys_sorted, return_index=True)
    counts = np.diff(np.r_[starts, keys_sorted.size]).astype(np.int64)
    unique_errors: Dict[str, np.ndarray] = {}
    for model, values in errors.items():
        sorted_values = values[order].astype(np.float64, copy=False)
        unique_errors[model] = np.add.reduceat(sorted_values, starts) / counts
    return unique_keys, unique_errors, counts


def metric_prefix(prefix: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def choose_spatial(
    ranked: Sequence[Dict[str, Any]],
    selected: Sequence[Tuple[str, Dict[str, Any]]],
    used_keys: set,
    min_distance: float,
) -> Dict[str, Any]:
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
            return row
    for row in ranked:
        if row["key"] not in used_keys:
            return row
    return ranked[0]


def select_local_regions(
    candidates: Sequence[Dict[str, Any]],
    min_distance: float,
) -> List[Tuple[str, Dict[str, Any]]]:
    selected: List[Tuple[str, Dict[str, Any]]] = []
    used_keys: set = set()

    scores = np.asarray([float(row["difficulty_mean_mae_m"]) for row in candidates])
    median_score = float(np.median(scores))
    specifications: List[Tuple[str, Sequence[Dict[str, Any]]]] = [
        ("easy", sorted(candidates, key=lambda row: float(row["difficulty_mean_mae_m"]))),
        ("typical", sorted(candidates, key=lambda row: abs(float(row["difficulty_mean_mae_m"]) - median_score))),
        ("hard", sorted(candidates, key=lambda row: float(row["difficulty_mean_mae_m"]), reverse=True)),
    ]

    for label, ranked in specifications:
        choice = choose_spatial(ranked, selected, used_keys, min_distance)
        selected.append((label, choice))
        used_keys.add(choice["key"])
    return selected


def finite_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)[mask]
    return values[np.isfinite(values)]


def add_mask_contour(ax, mask: np.ndarray) -> None:
    if np.any(mask) and np.any(~mask):
        ax.contour(mask.astype(np.uint8), levels=[0.5], linewidths=0.7, colors="black")


def load_local_arrays(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    gt, gt_meta = read_one(Path(row["tile_path"]))
    hidden, _ = read_one(Path(row["hidden_path"]))
    core_loss, _ = read_one(Path(row["core_loss_path"]))

    predictions: Dict[str, np.ndarray] = {}
    prediction_meta: Dict[str, Dict[str, Any]] = {}
    for model in MODEL_ORDER:
        pred, meta = read_one(Path(row[f"{model}_pred_tile_path"]))
        predictions[model] = pred.astype(np.float32, copy=False)
        prediction_meta[model] = meta

    gt = gt.astype(np.float32, copy=False)
    hidden_bool = np.isfinite(hidden) & (hidden.astype(np.float32) > 0.5) & (hidden.astype(np.float32) < 255)
    core_bool = np.isfinite(core_loss) & (core_loss.astype(np.float32) > 0.5) & (core_loss.astype(np.float32) < 255)

    common = core_bool & valid_dem(gt, args.nodata, args.nodata_threshold, gt_meta.get("nodata"))
    for model in MODEL_ORDER:
        common &= valid_pred(predictions[model], args.nodata, prediction_meta[model].get("nodata"))

    errors = {
        model: np.where(common, predictions[model] - gt, np.nan)
        for model in MODEL_ORDER
    }
    abs_stack = np.stack([np.abs(errors[model]) for model in MODEL_ORDER], axis=0)
    safe_stack = np.where(np.isfinite(abs_stack), abs_stack, np.inf)
    winner_index = np.argmin(safe_stack, axis=0).astype(np.uint8) + 1
    winner_index[~common] = 0
    sorted_abs = np.sort(safe_stack, axis=0)
    winner_margin = np.full(common.shape, np.nan, dtype=np.float32)
    winner_margin[common] = (
        sorted_abs[1][common] - sorted_abs[0][common]
    ).astype(np.float32, copy=False)

    return {
        "gt": gt,
        "hidden": hidden_bool,
        "core": core_bool,
        "common": common,
        "predictions": predictions,
        "errors": errors,
        "winner_index": winner_index,
        "winner_margin": winner_margin,
        "meta": gt_meta,
    }


def plot_local_primary(region_dir: Path, arrays: Dict[str, Any], title: str) -> Dict[str, Any]:
    common = arrays["common"]
    gt = arrays["gt"]
    predictions = arrays["predictions"]
    errors = arrays["errors"]
    hidden = arrays["hidden"]

    input_show = gt.astype(np.float64).copy()
    input_valid = np.isfinite(input_show) & (input_show > -9999)
    input_show[~input_valid | hidden] = np.nan
    gt_show = np.where(common, gt, np.nan)

    elevation_values = [finite_values(gt, common)]
    elevation_values += [finite_values(predictions[m], common) for m in MODEL_ORDER]
    elevation_values_all = np.concatenate(elevation_values)
    elev_min = float(np.min(elevation_values_all))
    elev_max = float(np.max(elevation_values_all))
    if elev_max <= elev_min:
        elev_max = elev_min + 1.0

    error_values = np.concatenate([finite_values(errors[m], common) for m in MODEL_ORDER])
    full_error_max = max(float(np.max(np.abs(error_values))) if error_values.size else 1.0, 1e-6)

    fig, axes = plt.subplots(3, 3, figsize=(16, 15), constrained_layout=True)
    panels = [
        (input_show, "Visible model input", "terrain", elev_min, elev_max),
        (gt_show, "GT — common evaluation pixels", "terrain", elev_min, elev_max),
        (hidden.astype(float), "Strict Hidden Mask (1 = hidden)", "gray_r", 0.0, 1.0),
        (np.where(common, predictions["norm_only"], np.nan), "Prediction: normalized only", "terrain", elev_min, elev_max),
        (np.where(common, predictions["meter_only"], np.nan), "Prediction: meter only", "terrain", elev_min, elev_max),
        (np.where(common, predictions["norm_then_meter"], np.nan), "Prediction: normalized → meter", "terrain", elev_min, elev_max),
        (errors["norm_only"], "Signed error: normalized only", "RdBu_r", -full_error_max, full_error_max),
        (errors["meter_only"], "Signed error: meter only", "RdBu_r", -full_error_max, full_error_max),
        (errors["norm_then_meter"], "Signed error: normalized → meter", "RdBu_r", -full_error_max, full_error_max),
    ]
    for ax, (arr, label, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        image = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        add_mask_contour(ax, common)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        title + f"\nNative source grid; shared full signed-error range = ±{full_error_max:.3f} m; no clipping",
        fontsize=14,
    )
    output = region_dir / "G094_native_three_model_fullrange_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {
        "primary_png": str(output),
        "full_error_display_max_m": full_error_max,
    }


def plot_local_distribution(region_dir: Path, arrays: Dict[str, Any], title: str) -> str:
    common = arrays["common"]
    absolute = {
        model: np.abs(arrays["errors"][model][common].astype(np.float64))
        for model in MODEL_ORDER
    }
    combined = np.concatenate(list(absolute.values()))
    x_max = max(float(np.percentile(combined, 99.5)), 1e-6)
    bins = np.linspace(0, x_max, 100)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for model in MODEL_ORDER:
        axes[0].hist(
            absolute[model],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        values = np.sort(absolute[model])
        y = np.arange(1, values.size + 1) / values.size
        axes[1].plot(values, y, linewidth=2, color=MODEL_COLORS[model], label=MODEL_LABELS[model])

    axes[0].set_xlabel("Absolute error (m)")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(0, x_max)
    axes[0].set_title("Local absolute-error density; x-axis to P99.5")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel("Absolute error (m)")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_xlim(0, x_max)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Local absolute-error CDF; x-axis to P99.5")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(title)
    output = region_dir / "G094_local_three_model_error_distribution.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def plot_local_winner(region_dir: Path, arrays: Dict[str, Any], title: str) -> str:
    winner = arrays["winner_index"]
    margin = arrays["winner_margin"]
    common = arrays["common"]

    winner_show = np.ma.masked_where(~common, winner)
    cmap = ListedColormap([
        MODEL_COLORS["norm_only"],
        MODEL_COLORS["meter_only"],
        MODEL_COLORS["norm_then_meter"],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    image0 = axes[0].imshow(winner_show, cmap=cmap, vmin=1, vmax=3, interpolation="nearest")
    add_mask_contour(axes[0], common)
    axes[0].set_title("Lowest absolute-error model per pixel\n1=NormOnly, 2=MeterOnly, 3=NormThenMeter")
    axes[0].axis("off")
    colorbar = fig.colorbar(image0, ax=axes[0], fraction=0.046, pad=0.03, ticks=[1, 2, 3])
    colorbar.ax.set_yticklabels(["NormOnly", "MeterOnly", "Norm→Meter"])

    margin_limit = max(float(np.percentile(margin[common], 99)) if np.any(common) else 1.0, 1e-6)
    image1 = axes[1].imshow(margin, cmap="viridis", vmin=0, vmax=margin_limit, interpolation="nearest")
    add_mask_contour(axes[1], common)
    axes[1].set_title("Winner margin over second-best absolute error (m)\nDisplay capped at P99")
    axes[1].axis("off")
    fig.colorbar(image1, ax=axes[1], fraction=0.046, pad=0.03)
    fig.suptitle(title)
    output = region_dir / "G094_local_pixelwise_winner_map.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def write_local_rasters(region_dir: Path, arrays: Dict[str, Any], args: argparse.Namespace) -> Dict[str, str]:
    common = arrays["common"]
    meta = arrays["meta"]

    def masked_float(arr: np.ndarray) -> np.ndarray:
        output = np.full(arr.shape, args.nodata, dtype=np.float32)
        output[common] = np.asarray(arr, dtype=np.float32)[common]
        return output

    products: Dict[str, Tuple[np.ndarray, str, float]] = {
        "gt": (masked_float(arrays["gt"]), "float32", args.nodata),
        "common_evaluation_mask": (common.astype(np.uint8), "uint8", 255),
        "hidden_mask": (arrays["hidden"].astype(np.uint8), "uint8", 255),
        "pixelwise_winner_model": (
            np.where(common, arrays["winner_index"], 255).astype(np.uint8),
            "uint8",
            255,
        ),
        "winner_margin_m": (masked_float(arrays["winner_margin"]), "float32", args.nodata),
    }
    for model in MODEL_ORDER:
        products[f"prediction_{model}"] = (
            masked_float(arrays["predictions"][model]),
            "float32",
            args.nodata,
        )
        products[f"error_{model}"] = (
            masked_float(arrays["errors"][model]),
            "float32",
            args.nodata,
        )

    paths: Dict[str, str] = {}
    for name, (array, dtype, nodata) in products.items():
        path = region_dir / f"G094_{name}.tif"
        write_tif(path, array, meta, nodata, dtype)
        paths[name] = str(path)
    return paths


def absolute_limit(data: Dict[str, np.ndarray], percentile: float) -> float:
    values = np.concatenate([np.abs(data[m].astype(np.float64)) for m in MODEL_ORDER])
    values = values[np.isfinite(values)]
    limit = float(np.percentile(values, percentile))
    return max(limit, 1e-6)


def signed_limit(data: Dict[str, np.ndarray], percentile: float) -> float:
    values = np.concatenate([data[m].astype(np.float64) for m in MODEL_ORDER])
    values = values[np.isfinite(values)]
    low = float(np.percentile(values, 100 - percentile))
    high = float(np.percentile(values, percentile))
    return max(abs(low), abs(high), 1e-6)


def empirical_cdf(values: np.ndarray, max_points: int = 25_000):
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    if sorted_values.size <= max_points:
        index = np.arange(sorted_values.size)
    else:
        index = np.linspace(0, sorted_values.size - 1, max_points, dtype=np.int64)
    return sorted_values[index], (index + 1) / sorted_values.size


def plot_global_distribution(
    cases: Sequence[Case],
    global_errors: Dict[str, Dict[str, np.ndarray]],
    global_stats: Dict[str, Dict[str, Dict[str, Any]]],
    out_dir: Path,
    percentile: float,
) -> Tuple[str, str]:
    fig, axes = plt.subplots(len(cases), 2, figsize=(14, 4.2 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes = np.asarray([axes])

    for row_idx, case in enumerate(cases):
        data = global_errors[case.river]
        upper = absolute_limit(data, percentile)
        bins = np.linspace(0, upper, 241)

        for model in MODEL_ORDER:
            absolute = np.abs(data[model])
            density, edges = np.histogram(absolute, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            axes[row_idx, 0].plot(
                centers,
                density,
                linewidth=2,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
            x, y = empirical_cdf(absolute)
            axes[row_idx, 1].plot(
                x,
                y,
                linewidth=2,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )

        axes[row_idx, 0].set_xlim(0, upper)
        axes[row_idx, 0].set_xlabel("Absolute error |Prediction - GT| (m)")
        axes[row_idx, 0].set_ylabel("Probability density")
        axes[row_idx, 0].set_title(f"{case.short_name}: absolute-error density")
        axes[row_idx, 0].grid(True, alpha=0.25)
        axes[row_idx, 0].legend()

        stats_text = "\n".join(
            f"{MODEL_LABELS[m]}: MAE={global_stats[case.river][m]['mae_m']:.3f}, "
            f"RMSE={global_stats[case.river][m]['rmse_m']:.3f}, "
            f"P95={global_stats[case.river][m]['p95_abs_error_m']:.3f} m"
            for m in MODEL_ORDER
        )
        axes[row_idx, 0].text(
            0.98,
            0.68,
            stats_text,
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

        axes[row_idx, 1].set_xlim(0, upper)
        axes[row_idx, 1].set_ylim(0, 1)
        axes[row_idx, 1].set_xlabel("Absolute error |Prediction - GT| (m)")
        axes[row_idx, 1].set_ylabel("Cumulative fraction")
        axes[row_idx, 1].set_title(f"{case.short_name}: absolute-error CDF")
        axes[row_idx, 1].grid(True, alpha=0.25)
        axes[row_idx, 1].legend()

    fig.suptitle(
        "Three-model full-river error comparison\n"
        "Exact common unique-geospatial Core-Loss pixels for all models",
        fontsize=16,
    )
    png = out_dir / "G094_abs_error_density_and_cdf_3x2_three_models.png"
    pdf = out_dir / "G094_abs_error_density_and_cdf_3x2_three_models.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return str(png), str(pdf)


def plot_global_signed(
    cases: Sequence[Case],
    global_errors: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    percentile: float,
) -> str:
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 3.8 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        data = global_errors[case.river]
        limit = signed_limit(data, percentile)
        bins = np.linspace(-limit, limit, 241)
        for model in MODEL_ORDER:
            density, edges = np.histogram(data[model], bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(
                centers,
                density,
                linewidth=2,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(-limit, limit)
        ax.set_title(case.short_name)
        ax.set_xlabel("Signed error: Prediction - GT (m)")
        ax.set_ylabel("Probability density")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(
        "Three-model signed-error distributions\nExact common unique-geospatial Core-Loss pixels",
        fontsize=15,
    )
    output = out_dir / "G094_signed_error_density_three_models.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def plot_global_metric_bars(
    cases: Sequence[Case],
    stats: Dict[str, Dict[str, Dict[str, Any]]],
    out_dir: Path,
) -> str:
    metrics = ("mae_m", "rmse_m", "p95_abs_error_m")
    metric_labels = ("MAE", "RMSE", "P95 absolute error")
    x = np.arange(len(cases))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, metric, metric_label in zip(axes, metrics, metric_labels):
        for offset_idx, model in enumerate(MODEL_ORDER):
            values = [stats[case.river][model][metric] for case in cases]
            ax.bar(
                x + (offset_idx - 1) * width,
                values,
                width,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([case.short_name for case in cases], rotation=15, ha="right")
        ax.set_ylabel("Error (m)")
        ax.set_title(metric_label)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Three-model full-river metrics on the exact common footprint")
    output = out_dir / "G094_global_metric_bars_three_models.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def plot_candidate_summary(
    path: Path,
    candidates: Sequence[Dict[str, Any]],
    selected: Sequence[Tuple[str, Dict[str, Any]]],
    title: str,
) -> None:
    ordered = sorted(candidates, key=lambda row: float(row["difficulty_mean_mae_m"]))
    rank = np.arange(1, len(ordered) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for model in MODEL_ORDER:
        axes[0].plot(
            rank,
            [float(row[f"{model}_mae_m"]) for row in ordered],
            linewidth=1.5,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    axes[0].set_xlabel("Candidate local region rank by mean three-model MAE")
    axes[0].set_ylabel("Local MAE (m)")
    axes[0].set_title("Local-region MAE profiles")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    winner_counts = {
        model: sum(row["winner_by_local_mae"] == model for row in candidates)
        for model in MODEL_ORDER
    }
    axes[1].bar(
        [MODEL_LABELS[m] for m in MODEL_ORDER],
        [winner_counts[m] for m in MODEL_ORDER],
        color=[MODEL_COLORS[m] for m in MODEL_ORDER],
    )
    axes[1].set_ylabel("Candidate region count")
    axes[1].set_title("Lowest-MAE model by candidate region")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(title + f"\nSelected regions: {', '.join(label for label, _ in selected)}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def collect_case(case: Case, args: argparse.Namespace, out_dir: Path):
    paths = get_case_paths(case, args)
    manifests = {
        model: index_manifest(read_csv(paths[model]["manifest"]))
        for model in MODEL_ORDER
    }
    common_keys = sorted(set.intersection(*(set(manifests[m]) for m in MODEL_ORDER)))
    if not common_keys:
        raise RuntimeError(f"No common manifest keys across all three models for {case.river}")

    summaries = {
        model: read_json(paths[model]["pred_summary"])
        for model in MODEL_ORDER
    }
    widths = {model: resolve_mosaic_width(summaries[model]) for model in MODEL_ORDER}
    if len(set(widths.values())) != 1:
        raise RuntimeError(f"Virtual mosaic widths differ for {case.river}: {widths}")
    mosaic_width = next(iter(widths.values()))

    candidate_rows: List[Dict[str, Any]] = []
    key_parts: List[np.ndarray] = []
    error_parts: Dict[str, List[np.ndarray]] = {model: [] for model in MODEL_ORDER}

    print(f"[CASE] {case.river}: common manifest keys={len(common_keys):,}")
    for index, key in enumerate(common_keys, start=1):
        rows = {model: manifests[model][key] for model in MODEL_ORDER}

        tile_path = Path(rows["norm_then_meter"]["tile_path"])
        gt, gt_meta = read_one(tile_path)
        core_loss, _ = read_one(Path(rows["norm_then_meter"]["core_loss_path"]))

        predictions: Dict[str, np.ndarray] = {}
        pred_meta: Dict[str, Dict[str, Any]] = {}
        for model in MODEL_ORDER:
            pred, meta = read_one(Path(rows[model]["avg_pred_tile_path"]))
            predictions[model] = pred.astype(np.float32, copy=False)
            pred_meta[model] = meta

        gt = gt.astype(np.float32, copy=False)
        core_mask = (
            np.isfinite(core_loss)
            & (core_loss.astype(np.float32) > 0.5)
            & (core_loss.astype(np.float32) < 255)
        )
        common = core_mask & valid_dem(gt, args.nodata, args.nodata_threshold, gt_meta.get("nodata"))
        for model in MODEL_ORDER:
            common &= valid_pred(predictions[model], args.nodata, pred_meta[model].get("nodata"))

        n_common = int(common.sum())
        if n_common == 0:
            continue

        errors = {
            model: (predictions[model] - gt).astype(np.float32)
            for model in MODEL_ORDER
        }
        stats = {
            model: error_stats(errors[model][common])
            for model in MODEL_ORDER
        }

        offsets = {
            model: (int(rows[model]["mosaic_row0"]), int(rows[model]["mosaic_col0"]))
            for model in MODEL_ORDER
        }
        if len(set(offsets.values())) != 1:
            raise RuntimeError(f"Mosaic offset mismatch for {case.river} key={key}: {offsets}")
        row0, col0 = next(iter(offsets.values()))

        height, width = gt.shape
        local_flat = np.flatnonzero(common.ravel()).astype(np.int64)
        rr = local_flat // width
        cc = local_flat % width
        global_key = (
            (np.int64(row0) + rr) * np.int64(mosaic_width)
            + (np.int64(col0) + cc)
        )
        key_parts.append(global_key)
        for model in MODEL_ORDER:
            error_parts[model].append(errors[model].ravel()[local_flat])

        cx, cy = center_xy(gt_meta, gt.shape)
        maes = {model: float(stats[model]["mae_m"]) for model in MODEL_ORDER}
        winner = min(MODEL_ORDER, key=lambda model: maes[model])
        candidate = {
            "preset": case.preset,
            "short_name": case.short_name,
            "river": case.river,
            "key": key,
            "tile_id": rows["norm_then_meter"].get("tile_id", ""),
            "center_x": cx,
            "center_y": cy,
            "common_n_pixels": n_common,
            "difficulty_mean_mae_m": float(np.mean(list(maes.values()))),
            "local_mae_spread_m": max(maes.values()) - min(maes.values()),
            "winner_by_local_mae": winner,
            "tile_path": str(tile_path),
            "hidden_path": rows["norm_then_meter"]["hidden_path"],
            "core_loss_path": rows["norm_then_meter"]["core_loss_path"],
        }
        for model in MODEL_ORDER:
            candidate[f"{model}_pred_tile_path"] = rows[model]["avg_pred_tile_path"]
            candidate.update(metric_prefix(model, stats[model]))
            other_mae = [maes[m] for m in MODEL_ORDER if m != model]
            candidate[f"{model}_advantage_m"] = float(np.mean(other_mae) - maes[model])
        candidate_rows.append(candidate)

        if index == 1 or index == len(common_keys) or index % 200 == 0:
            print(f"  processed {index:,}/{len(common_keys):,} paired tiles")

    eligible = [
        row for row in candidate_rows
        if int(row["common_n_pixels"]) >= args.min_common_pixels
    ]
    if len(eligible) < 3:
        raise RuntimeError(
            f"Only {len(eligible)} local candidates have >= {args.min_common_pixels} common pixels for {case.river}"
        )
    if not key_parts:
        raise RuntimeError(f"No common evaluation pixels found for {case.river}")

    all_keys = np.concatenate(key_parts).astype(np.int64, copy=False)
    all_errors = {
        model: np.concatenate(error_parts[model]).astype(np.float32, copy=False)
        for model in MODEL_ORDER
    }
    unique_keys, unique_errors, overlap_counts = aggregate_unique_multi(all_keys, all_errors)
    global_stats = {model: error_stats(unique_errors[model]) for model in MODEL_ORDER}

    long_rows: List[Dict[str, Any]] = []
    for model in MODEL_ORDER:
        long_rows.append({
            "preset": case.preset,
            "short_name": case.short_name,
            "river": case.river,
            "model": model,
            "model_label": MODEL_LABELS[model],
            "comparison_footprint": "exact common unique geospatial pixels for all three models",
            **global_stats[model],
            "checkpoint": str(summaries[model].get("checkpoint", "")),
        })

    wide_row: Dict[str, Any] = {
        "preset": case.preset,
        "short_name": case.short_name,
        "river": case.river,
        "common_n_pixels": int(global_stats["norm_only"]["n_pixels"]),
        "max_overlap_multiplicity_common": int(overlap_counts.max()) if overlap_counts.size else 0,
    }
    for model in MODEL_ORDER:
        for metric in ("mae_m", "rmse_m", "bias_m", "median_abs_error_m", "p95_abs_error_m", "p99_abs_error_m", "max_abs_error_m"):
            wide_row[f"{model}_{metric}"] = global_stats[model][metric]
    wide_row["winner_by_mae"] = min(MODEL_ORDER, key=lambda model: global_stats[model]["mae_m"])
    wide_row["winner_by_rmse"] = min(MODEL_ORDER, key=lambda model: global_stats[model]["rmse_m"])

    selected = select_local_regions(eligible, args.min_selected_center_distance)
    river_dir = out_dir / "local_regions" / case.preset
    river_dir.mkdir(parents=True, exist_ok=True)
    plot_candidate_summary(
        river_dir / "G094_candidate_local_mae_summary.png",
        eligible,
        selected,
        f"{case.short_name}: all paired native local candidates",
    )

    selected_rows: List[Dict[str, Any]] = []
    for label, row in selected:
        region_dir = river_dir / f"{label}_{row['key']}"
        region_dir.mkdir(parents=True, exist_ok=True)
        arrays = load_local_arrays(row, args)
        title = (
            f"{case.short_name} | {label.replace('_', ' ').upper()} | {row['key']} | "
            f"common pixels={int(row['common_n_pixels']):,}"
        )
        primary_info = plot_local_primary(region_dir, arrays, title)
        distribution_png = plot_local_distribution(region_dir, arrays, title)
        winner_png = plot_local_winner(region_dir, arrays, title)
        raster_paths = write_local_rasters(region_dir, arrays, args)

        selected_row: Dict[str, Any] = {
            **row,
            "selection_class": label,
            "region_dir": str(region_dir.relative_to(out_dir)),
            **{
                key: str(Path(value).relative_to(out_dir))
                for key, value in primary_info.items()
                if key.endswith("_png")
            },
            **{
                key: value
                for key, value in primary_info.items()
                if not key.endswith("_png")
            },
            "distribution_png": str(Path(distribution_png).relative_to(out_dir)),
            "winner_png": str(Path(winner_png).relative_to(out_dir)),
        }
        for product, path in raster_paths.items():
            selected_row[f"raster_{product}"] = str(Path(path).relative_to(out_dir))
        (region_dir / "G094_region_metrics.json").write_text(json.dumps(selected_row, indent=2))
        selected_rows.append(selected_row)

    return {
        "long_rows": long_rows,
        "wide_row": wide_row,
        "candidate_rows": candidate_rows,
        "selected_rows": selected_rows,
        "unique_errors": unique_errors,
        "global_stats": global_stats,
        "n_unique_keys": int(unique_keys.size),
    }


def zip_output(out_dir: Path) -> Path:
    zip_path = out_dir.parent / f"{out_dir.name}_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"{out_dir.name}/{path.relative_to(out_dir)}")
    return zip_path


def main() -> None:
    args = parse_args()
    if not 50 < args.global_display_percentile < 100:
        raise ValueError("--global_display_percentile must be between 50 and 100.")
    required_roots = (
        args.norm_only_pred_root,
        args.norm_only_error_root,
        args.meter_only_pred_root,
        args.meter_only_error_root,
        args.norm_then_meter_pred_root,
        args.norm_then_meter_error_root,
    )
    for path in required_roots:
        if not path.is_dir():
            raise FileNotFoundError(path)

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise RuntimeError(f"Output is non-empty: {out_dir}. Use --overwrite or choose another directory.")
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = out_dir / "global_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    global_long_rows: List[Dict[str, Any]] = []
    global_wide_rows: List[Dict[str, Any]] = []
    all_candidate_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    global_errors: Dict[str, Dict[str, np.ndarray]] = {}
    global_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for case in DEFAULT_CASES:
        result = collect_case(case, args, out_dir)
        global_long_rows.extend(result["long_rows"])
        global_wide_rows.append(result["wide_row"])
        all_candidate_rows.extend(result["candidate_rows"])
        selected_rows.extend(result["selected_rows"])
        global_errors[case.river] = result["unique_errors"]
        global_stats[case.river] = result["global_stats"]

    write_csv(out_dir / "G094_global_common_metrics_long.csv", global_long_rows)
    write_csv(out_dir / "G094_global_common_metrics_wide.csv", global_wide_rows)
    write_csv(out_dir / "G094_all_local_candidate_metrics.csv", all_candidate_rows)
    write_csv(out_dir / "G094_selected_local_regions.csv", selected_rows)

    global_abs_png, global_abs_pdf = plot_global_distribution(
        DEFAULT_CASES,
        global_errors,
        global_stats,
        figure_dir,
        args.global_display_percentile,
    )
    global_signed_png = plot_global_signed(
        DEFAULT_CASES,
        global_errors,
        figure_dir,
        args.global_display_percentile,
    )
    global_bar_png = plot_global_metric_bars(
        DEFAULT_CASES,
        global_stats,
        figure_dir,
    )

    manifest = {
        "generated_at_unix": time.time(),
        "generator": Path(__file__).name,
        "models": {
            "norm_only": {
                "prediction_root": str(args.norm_only_pred_root.resolve()),
                "error_root": str(args.norm_only_error_root.resolve()),
            },
            "meter_only": {
                "prediction_root": str(args.meter_only_pred_root.resolve()),
                "error_root": str(args.meter_only_error_root.resolve()),
            },
            "norm_then_meter": {
                "prediction_root": str(args.norm_then_meter_pred_root.resolve()),
                "error_root": str(args.norm_then_meter_error_root.resolve()),
            },
        },
        "comparison_mask": (
            "Core_Loss_Mask_Pixel AND valid_GT AND valid_prediction_NormOnly "
            "AND valid_prediction_MeterOnly AND valid_prediction_NormThenMeter"
        ),
        "global_metric_scope": "each exact-common overlap-averaged geospatial pixel counted once",
        "local_candidate_unit": "paired native 336x336 E001 prediction-tile context from F010/F060/F025",
        "local_selection": [
            "easy by minimum mean three-model MAE",
            "typical by median mean three-model MAE",
            "hard by maximum mean three-model MAE",
        ],
        "html_generated": False,
        "full_river_web_map_generated": False,
        "display_policy": {
            "global_distribution_axis_percentile": args.global_display_percentile,
            "global_metrics_use_all_common_pixels": True,
            "local_native_grid": True,
            "local_reprojected": False,
            "local_resampled": False,
            "primary_local_error_scale": "actual shared local max absolute error; no clipping",
        },
        "parameters": {
            "min_common_pixels": args.min_common_pixels,
            "min_selected_center_distance": args.min_selected_center_distance,
            "nodata": args.nodata,
            "nodata_threshold": args.nodata_threshold,
        },
        "outputs": {
            "global_metrics_long_csv": str(out_dir / "G094_global_common_metrics_long.csv"),
            "global_metrics_wide_csv": str(out_dir / "G094_global_common_metrics_wide.csv"),
            "all_local_candidates_csv": str(out_dir / "G094_all_local_candidate_metrics.csv"),
            "selected_local_regions_csv": str(out_dir / "G094_selected_local_regions.csv"),
            "global_abs_distribution_png": global_abs_png,
            "global_abs_distribution_pdf": global_abs_pdf,
            "global_signed_distribution_png": global_signed_png,
            "global_metric_bars_png": global_bar_png,
            "local_regions_root": str(out_dir / "local_regions"),
        },
        "global_rows": global_wide_rows,
        "selected_regions": selected_rows,
    }
    (out_dir / "G094_analysis_manifest.json").write_text(json.dumps(manifest, indent=2))

    zip_path = None
    if not args.no_zip:
        zip_path = zip_output(out_dir)

    print("=" * 90)
    print("DONE G094 three-model common-footprint global and local analysis")
    print(f"OUT_DIR={out_dir}")
    print(f"GLOBAL_METRICS={out_dir / 'G094_global_common_metrics_wide.csv'}")
    print(f"GLOBAL_FIGURE={global_abs_png}")
    print(f"LOCAL_REGIONS={out_dir / 'local_regions'}")
    print("HTML_GENERATED=NO")
    if zip_path is not None:
        print(f"ZIP={zip_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
