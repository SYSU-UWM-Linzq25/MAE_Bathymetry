#!/usr/bin/env python3
"""Build a high-detail Leaflet dashboard package for three MAE holdout experiments.

Why this version exists
-----------------------
F034 embedded one downsampled PNG per layer in a single HTML.  That design is
convenient, but no browser can recover details that were removed during the
quicklook downsampling.  It also used a checkerboard to indicate transparent
NoData pixels.

F045 creates local XYZ raster tiles and displays them in Leaflet over an
online satellite basemap. Unlike the earlier F038 version, it does not require
gdal2tiles.py or
the osgeo_utils Python package.  Its internal strip tiler uses only GDAL core
executables plus Pillow.  Only tiles visible in the current map view are loaded,
so the browser can inspect a much finer map without decoding one enormous image.
The package is zipped after generation, so the user only needs to download one
ZIP, extract it, and open the HTML.

Included experiment/layer structure
-----------------------------------
Experiments (allowlisted by default):
  - holdout_CA_D001NoDataSafe
  - holdout_CO_D001NoDataSafe
  - holdout_Santiam_D001NoDataSafe

For each experiment/river:
  - Prediction and GT, with one shared elevation scale
  - Signed error and absolute error
  - Raw E001 bathymetry input
  - E001 Hidden_Mask
  - E001 Loss_Mask_Pixel

Important
---------
* The generated local XYZ overlays work offline after extraction.
* The satellite/light-gray basemaps require an Internet connection in the browser.
* The official F020 metrics are read from the existing summary JSON.  They are not
  recomputed from display tiles. Original F010/F020 rasters are never modified.
* DETAIL_RES_M controls the DISPLAY overlay resolution in EPSG:3857. This is a
  visualization-only resampling setting. The default 4 m is much sharper than
  the old ~2400-pixel whole-river quicklook while keeping
  the package size practical.  Use 2 m for a larger, slower package.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required for the core-GDAL XYZ tiler") from exc

try:
    import tifffile
except Exception as exc:  # pragma: no cover
    raise RuntimeError("tifffile is required") from exc

try:
    import matplotlib
    from matplotlib import colormaps
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required") from exc


DEFAULT_PRED_ROOT = Path(
    "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
    "FullRiver_Predictions_F010_TileAvgVRT_D001NoDataSafe"
)
DEFAULT_ERROR_ROOT = Path(
    "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
    "FullRiver_GT_Error_F020_TileVRT_D001NoDataSafe"
)
DEFAULT_TILE_ROOT = Path(
    "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
    "Tiles_for_MAE_FullRiver_E001/Tiles_1m"
)
DEFAULT_OUT_DIR = Path(
    "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
    "FullRiver_WebMap_F045_HoldoutOnly_GTiffStripParentPyramid_D001NoDataSafe"
)
DEFAULT_HTML_NAME = "F045_HoldoutOnly_HighDetail_Satellite_Dashboard_GTiffStripParentPyramid.html"
DEFAULT_ZIP_NAME = "F045_HoldoutOnly_HighDetail_Satellite_Dashboard_GTiffStripParentPyramid_Package.zip"
DEFAULT_EXPERIMENTS = (
    "holdout_CA_D001NoDataSafe",
    "holdout_CO_D001NoDataSafe",
    "holdout_Santiam_D001NoDataSafe",
)
NODATA_DEFAULT = -999999.0
MASK_NODATA_DEFAULT = 255.0
WEB_MERCATOR_INITIAL_RES = 156543.03392804097
WEB_MERCATOR_HALF_WORLD = 20037508.342789244


@dataclass
class RasterInfo:
    path: Path
    width: int
    height: int
    nodata: Optional[float]
    geotransform: Optional[Tuple[float, float, float, float, float, float]]


@dataclass
class RiverSources:
    experiment: str
    river: str
    pred_vrt: Path
    gt_vrt: Path
    error_vrt: Optional[Path]
    input_bathy_tiles: List[Path]
    hidden_mask_tiles: List[Path]
    loss_mask_tiles: List[Path]
    pred_summary: Dict[str, Any]
    error_summary: Dict[str, Any]
    pred_summary_path: Optional[Path]
    error_summary_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create a high-detail satellite Leaflet package for holdout full-river results.",
    )
    p.add_argument("--pred_root", type=Path, default=DEFAULT_PRED_ROOT)
    p.add_argument("--error_root", type=Path, default=DEFAULT_ERROR_ROOT)
    p.add_argument("--tile_root", type=Path, default=DEFAULT_TILE_ROOT)
    p.add_argument("--tile_res", default="1m")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--out_html", default=DEFAULT_HTML_NAME)
    p.add_argument("--zip_name", default=DEFAULT_ZIP_NAME)
    p.add_argument("--experiments", nargs="*", default=list(DEFAULT_EXPERIMENTS))
    p.add_argument("--rivers", nargs="*", default=[])

    p.add_argument(
        "--detail_res_m",
        type=float,
        default=4.0,
        help="Native overlay pixel size in EPSG:3857 metres. Use 2 for more detail/larger output.",
    )
    p.add_argument("--min_zoom", type=int, default=-1, help="-1 means auto from max zoom.")
    p.add_argument("--max_zoom", type=int, default=-1, help="-1 means auto from detail_res_m.")
    p.add_argument("--tile_processes", type=int, default=8)
    p.add_argument("--stats_max_px", type=int, default=2200)
    p.add_argument("--elev_low_pct", type=float, default=2.0)
    p.add_argument("--elev_high_pct", type=float, default=98.0)
    p.add_argument("--error_abs_pct", type=float, default=98.0)
    p.add_argument("--nodata", type=float, default=NODATA_DEFAULT)
    p.add_argument("--nodata_threshold", type=float, default=-9999.0)
    p.add_argument("--mask_nodata", type=float, default=MASK_NODATA_DEFAULT)
    p.add_argument("--elev_cmap", default="terrain")
    p.add_argument("--error_cmap", default="RdBu_r")
    p.add_argument("--abs_error_cmap", default="magma")
    p.add_argument("--overlay_opacity", type=float, default=0.82)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep_intermediate", action="store_true")
    p.add_argument("--no_zip", action="store_true")
    return p.parse_args()


def run(cmd: Sequence[str], *, capture: bool = False, env: Optional[Dict[str, str]] = None) -> str:
    printable = " ".join(str(x) for x in cmd)
    print(f"[CMD] {printable}", flush=True)
    proc = subprocess.run(
        [str(x) for x in cmd],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=env,
    )
    return proc.stdout if capture else ""


def find_executable(candidates: Sequence[str]) -> str:
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(f"Required executable not found. Tried: {', '.join(candidates)}")


def safe_json_load(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {"value": value}
    except Exception as exc:
        print(f"[WARN] Cannot parse JSON {path}: {exc}", file=sys.stderr)
        return {}


def find_summary_json(directory: Path, preferred_names: Sequence[str]) -> Optional[Path]:
    for name in preferred_names:
        p = directory / name
        if p.is_file():
            return p
    ranked: List[Tuple[int, Path]] = []
    for p in sorted(directory.glob("*.json")):
        low = p.name.lower()
        score = (10 if "summary" in low else 0) - (8 if "args" in low else 0) - (4 if "all_rivers" in low else 0)
        ranked.append((score, p))
    return max(ranked, default=(0, None), key=lambda x: (x[0], str(x[1])))[1]


def iter_string_values(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_string_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_string_values(value)
    elif isinstance(obj, str):
        yield obj


def paths_from_summary(summary: Dict[str, Any], directory: Path) -> List[Path]:
    out: List[Path] = []
    for value in iter_string_values(summary):
        if not value.lower().endswith((".vrt", ".tif", ".tiff")):
            continue
        p = Path(value)
        if not p.is_absolute():
            p = directory / p
        out.append(p)
    return out


def score_prediction(path: Path) -> int:
    low = path.name.lower()
    score = 4 if path.suffix.lower() == ".vrt" else 0
    score += 4 if "fullriver" in low else 0
    score += 12 if ("pred" in low or "prediction" in low) else 0
    score += 2 if "avg" in low else 0
    score -= 30 if ("count" in low or "overlap" in low) else 0
    score -= 20 if ("error" in low or "err" in low) else 0
    score -= 10 if ("gt" in low or "ground" in low or "bathy" in low) else 0
    return score


def score_gt(path: Path) -> int:
    low = path.name.lower()
    score = 4 if path.suffix.lower() == ".vrt" else 0
    score += 3 if "fullriver" in low else 0
    if re.search(r"(^|[_-])gt([_.-]|$)", low) or "groundtruth" in low or "ground_truth" in low:
        score += 18
    elif "gt" in low:
        score += 8
    if "bathy" in low and "pred" not in low and "error" not in low and "err" not in low:
        score += 12
    score -= 30 if ("error" in low or "err" in low) else 0
    score -= 12 if "pred" in low else 0
    score -= 15 if ("count" in low or "mask" in low) else 0
    return score


def score_error(path: Path) -> int:
    low = path.name.lower()
    score = 4 if path.suffix.lower() == ".vrt" else 0
    score += 3 if "fullriver" in low else 0
    score += 20 if "error" in low else (14 if re.search(r"(^|[_-])err([_.-]|$)", low) else 0)
    score += 14 if any(t in low for t in ("difference", "diff", "residual", "pred_minus_gt", "gt_minus_pred")) else 0
    score += 4 if "signed" in low else 0
    score -= 4 if ("abs" in low or "absolute" in low) else 0
    score -= 20 if ("count" in low or "mask" in low) else 0
    score -= 10 if ("gt" in low and "error" not in low) else 0
    return score


def select_raster(directory: Path, summary: Dict[str, Any], scorer, label: str, required: bool = True) -> Optional[Path]:
    candidates: Dict[str, Path] = {}
    for p in paths_from_summary(summary, directory):
        if p.exists():
            candidates[str(p.resolve())] = p
    for pattern in ("*.vrt", "*/*.vrt"):
        for p in directory.glob(pattern):
            if p.is_file():
                candidates[str(p.resolve())] = p
    ranked = sorted(((scorer(p), p) for p in candidates.values()), key=lambda x: (x[0], str(x[1])), reverse=True)
    if not ranked or ranked[0][0] <= 0:
        if required:
            raise FileNotFoundError(f"Could not auto-detect {label} raster under {directory}")
        return None
    print(f"[DETECT] {label}: score={ranked[0][0]} path={ranked[0][1]}")
    return ranked[0][1]


def _tile_id(path: Path) -> int:
    m = re.search(r"_ID(\d+)", path.name, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse tile ID from {path.name}")
    return int(m.group(1))


def discover_input_tiles(tile_root: Path, river: str, res: str) -> Tuple[List[Path], List[Path], List[Path]]:
    full_dir = tile_root / "FullRiver_tile"
    hidden_dir = tile_root / "Hidden_Mask"
    loss_dir = tile_root / "Loss_Mask_Pixel"
    for directory in (full_dir, hidden_dir, loss_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing E001 input directory: {directory}")
    bathy = sorted(full_dir.glob(f"E001_FullRiver_tile_{res}_{river}_ID*.tif"), key=_tile_id)
    hidden = sorted(hidden_dir.glob(f"E001_tile_{res}_{river}_ID*_HiddenMask.tif"), key=_tile_id)
    loss = sorted(loss_dir.glob(f"E001_tile_{res}_{river}_ID*_LossMaskPixel.tif"), key=_tile_id)
    if not bathy or not hidden or not loss:
        raise FileNotFoundError(f"Incomplete E001 input tile sets for river={river}")
    maps = [{_tile_id(p): p for p in seq} for seq in (bathy, hidden, loss)]
    ids = [set(m) for m in maps]
    if not (ids[0] == ids[1] == ids[2]):
        raise RuntimeError(f"E001 Bathy/Hidden/Loss tile IDs do not match for river={river}")
    ordered = sorted(ids[0])
    return ([maps[0][i] for i in ordered], [maps[1][i] for i in ordered], [maps[2][i] for i in ordered])


def discover_sources(args: argparse.Namespace) -> Tuple[List[RiverSources], List[str]]:
    pred_root = args.pred_root.resolve()
    error_root = args.error_root.resolve()
    tile_root = args.tile_root.resolve()
    for p in (pred_root, error_root, tile_root):
        if not p.is_dir():
            raise FileNotFoundError(p)
    exp_allow = set(args.experiments)
    river_allow = set(args.rivers)
    warnings: List[str] = []
    found: List[RiverSources] = []
    pred_exps = {p.name: p for p in pred_root.iterdir() if p.is_dir()}
    err_exps = {p.name: p for p in error_root.iterdir() if p.is_dir()}
    experiments = sorted(set(pred_exps) & set(err_exps))
    if exp_allow:
        experiments = [e for e in experiments if e in exp_allow]
    for experiment in experiments:
        pred_rivers = {p.name: p for p in pred_exps[experiment].iterdir() if p.is_dir()}
        err_rivers = {p.name: p for p in err_exps[experiment].iterdir() if p.is_dir()}
        rivers = sorted(set(pred_rivers) & set(err_rivers))
        if river_allow:
            rivers = [r for r in rivers if r in river_allow]
        for river in rivers:
            try:
                pred_dir = pred_rivers[river]
                err_dir = err_rivers[river]
                pred_summary_path = find_summary_json(pred_dir, ["F010_summary.json"])
                error_summary_path = find_summary_json(err_dir, ["F020_summary.json", "F020_gt_error_summary.json", "summary.json"])
                pred_summary = safe_json_load(pred_summary_path)
                error_summary = safe_json_load(error_summary_path)
                pred_vrt = select_raster(pred_dir, pred_summary, score_prediction, "Prediction", True)
                gt_vrt = select_raster(err_dir, error_summary, score_gt, "GT", True)
                error_vrt = select_raster(err_dir, error_summary, score_error, "Error", False)
                bathy, hidden, loss = discover_input_tiles(tile_root, river, args.tile_res)
                assert pred_vrt and gt_vrt
                found.append(RiverSources(
                    experiment=experiment,
                    river=river,
                    pred_vrt=pred_vrt.resolve(),
                    gt_vrt=gt_vrt.resolve(),
                    error_vrt=error_vrt.resolve() if error_vrt else None,
                    input_bathy_tiles=[p.resolve() for p in bathy],
                    hidden_mask_tiles=[p.resolve() for p in hidden],
                    loss_mask_tiles=[p.resolve() for p in loss],
                    pred_summary=pred_summary,
                    error_summary=error_summary,
                    pred_summary_path=pred_summary_path,
                    error_summary_path=error_summary_path,
                ))
            except Exception as exc:
                msg = f"SKIP {experiment}/{river}: {exc}"
                warnings.append(msg)
                print(f"[WARN] {msg}", file=sys.stderr)
    if not found:
        raise RuntimeError("No complete holdout experiment/river pairs were found.")
    return found, warnings


def gdal_info(path: Path) -> RasterInfo:
    info = json.loads(run(["gdalinfo", "-json", str(path)], capture=True))
    width, height = map(int, info["size"])
    bands = info.get("bands") or []
    nodata = bands[0].get("noDataValue") if bands else None
    gt = info.get("geoTransform")
    return RasterInfo(
        path=path,
        width=width,
        height=height,
        nodata=float(nodata) if nodata is not None else None,
        geotransform=tuple(float(x) for x in gt) if gt and len(gt) == 6 else None,
    )


def raster_bounds(info: RasterInfo) -> Tuple[float, float, float, float]:
    if info.geotransform is None:
        raise RuntimeError(f"Missing geotransform: {info.path}")
    gt = info.geotransform
    if abs(gt[2]) > 1e-10 or abs(gt[4]) > 1e-10:
        raise RuntimeError(f"Rotated raster is not supported: {info.path}")
    left = gt[0]
    top = gt[3]
    right = left + info.width * gt[1]
    bottom = top + info.height * gt[5]
    return min(left, right), max(top, bottom), max(left, right), min(top, bottom)


def mercator_to_lonlat(x: float, y: float) -> Tuple[float, float]:
    lon = x / WEB_MERCATOR_HALF_WORLD * 180.0
    lat = math.degrees(math.atan(math.sinh(y / 6378137.0)))
    return lon, lat


def build_tile_vrt(files: Sequence[Path], output: Path, file_list: Path, src_nodata: float, vrt_nodata: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    file_list.write_text("\n".join(str(p.resolve()) for p in files) + "\n", encoding="utf-8")
    run([
        "gdalbuildvrt", "-q", "-overwrite", "-resolution", "highest",
        "-srcnodata", str(src_nodata), "-vrtnodata", str(vrt_nodata),
        "-input_file_list", str(file_list), str(output),
    ])


def make_target_grid(source: Path, output_vrt: Path, detail_res_m: float, nodata: float) -> RasterInfo:
    output_vrt.parent.mkdir(parents=True, exist_ok=True)
    run([
        "gdalwarp", "-q", "-overwrite", "-of", "VRT",
        "-t_srs", "EPSG:3857", "-tr", str(detail_res_m), str(detail_res_m), "-tap",
        "-r", "near", "-srcnodata", str(nodata), "-dstnodata", str(nodata),
        str(source), str(output_vrt),
    ])
    return gdal_info(output_vrt)


def warp_aligned(
    source: Path,
    output: Path,
    target: RasterInfo,
    detail_res_m: float,
    src_nodata: float,
    dst_nodata: float,
    resampling: str,
    out_type: str = "Float32",
) -> None:
    left, top, right, bottom = raster_bounds(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdalwarp", "-q", "-overwrite", "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
        "-of", "GTiff", "-t_srs", "EPSG:3857", "-te_srs", "EPSG:3857",
        "-te", f"{left:.6f}", f"{bottom:.6f}", f"{right:.6f}", f"{top:.6f}",
        "-tr", str(detail_res_m), str(detail_res_m), "-tap",
        "-r", resampling,
        "-srcnodata", str(src_nodata), "-dstnodata", str(dst_nodata),
        "-ot", out_type,
        "-co", "TILED=YES", "-co", "BLOCKXSIZE=512", "-co", "BLOCKYSIZE=512",
        "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2" if out_type.lower().startswith("float") else "PREDICTOR=1",
        "-co", "SPARSE_OK=TRUE", "-co", "BIGTIFF=YES",
        str(source), str(output),
    ]
    run(cmd)


def sample_raster(source: Path, output: Path, max_px: int, nodata: float, resampling: str = "average") -> np.ndarray:
    info = gdal_info(source)
    scale = min(1.0, float(max_px) / float(max(info.width, info.height)))
    out_w = max(1, int(round(info.width * scale)))
    out_h = max(1, int(round(info.height * scale)))
    run([
        "gdal_translate", "-q", "-of", "GTiff", "-ot", "Float32", "-r", resampling,
        "-outsize", str(out_w), str(out_h), "-a_nodata", str(nodata),
        "-co", "COMPRESS=NONE", str(source), str(output),
    ])
    with tifffile.TiffFile(str(output)) as tif:
        arr = tif.pages[0].asarray().astype(np.float32, copy=False)
    return arr


def valid_values(arr: np.ndarray, nodata: float, threshold: float, max_samples: int = 2_000_000) -> np.ndarray:
    mask = np.isfinite(arr) & (arr != float(nodata)) & (arr > float(threshold))
    vals = arr[mask].astype(np.float64, copy=False)
    if vals.size > max_samples:
        step = max(1, vals.size // max_samples)
        vals = vals[::step][:max_samples]
    return vals


def mask_values(arr: np.ndarray, nodata: float, max_samples: int = 2_000_000) -> np.ndarray:
    mask = np.isfinite(arr) & (arr != float(nodata))
    vals = arr[mask].astype(np.float64, copy=False)
    if vals.size > max_samples:
        step = max(1, vals.size // max_samples)
        vals = vals[::step][:max_samples]
    return vals


def robust_limits(values: Sequence[np.ndarray], low_pct: float, high_pct: float) -> Tuple[float, float]:
    good = [v[np.isfinite(v)] for v in values if v.size]
    if not good:
        return 0.0, 1.0
    merged = np.concatenate(good)
    lo, hi = np.percentile(merged, [low_pct, high_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(merged)), float(np.nanmax(merged))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def error_limit(values: np.ndarray, pct: float) -> float:
    vals = np.abs(values[np.isfinite(values)])
    if not vals.size:
        return 1.0
    limit = float(np.percentile(vals, pct))
    return limit if np.isfinite(limit) and limit > 0 else max(float(np.nanmax(vals)), 1.0)


def write_color_file(path: Path, vmin: float, vmax: float, cmap_name: str, n: int = 17) -> List[str]:
    cmap = colormaps.get_cmap(cmap_name)
    colors: List[str] = []
    lines: List[str] = []
    for i in range(n):
        t = i / max(1, n - 1)
        value = vmin + t * (vmax - vmin)
        rgba = cmap(t)
        r, g, b = (int(round(255 * x)) for x in rgba[:3])
        lines.append(f"{value:.12g} {r} {g} {b} 255")
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    lines.append("nv 0 0 0 0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return colors


def write_abs_error_color_file(path: Path, vmax: float, cmap_name: str, n: int = 17) -> List[str]:
    """Color signed error by absolute magnitude without creating an abs raster.

    The color table is symmetric in value, but both negative and positive errors
    receive the same color at equal absolute magnitude. This is visualization
    only; the source signed-error raster remains unchanged.
    """
    cmap = colormaps.get_cmap(cmap_name)
    lines: List[str] = []
    colors: List[str] = []

    # Negative side, ascending from -vmax to 0, with magnitude decreasing.
    for i in range(n - 1, -1, -1):
        t = i / max(1, n - 1)
        value = -vmax * t
        rgba = cmap(t)
        r, g, b = (int(round(255 * x)) for x in rgba[:3])
        lines.append(f"{value:.12g} {r} {g} {b} 255")

    # Positive side. Skip zero because it was already written above.
    for i in range(1, n):
        t = i / max(1, n - 1)
        value = vmax * t
        rgba = cmap(t)
        r, g, b = (int(round(255 * x)) for x in rgba[:3])
        lines.append(f"{value:.12g} {r} {g} {b} 255")

    for i in range(n):
        t = i / max(1, n - 1)
        rgba = cmap(t)
        r, g, b = (int(round(255 * x)) for x in rgba[:3])
        colors.append(f"#{r:02x}{g:02x}{b:02x}")

    lines.append("nv 0 0 0 0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return colors


def write_mask_color_file(path: Path, kind: str) -> List[str]:
    if kind == "hidden":
        # 0 = visible, 1 = hidden
        lines = ["0 170 175 180 150", "1 255 215 0 235", "nv 0 0 0 0"]
        colors = ["#aaafb4", "#ffd700"]
    elif kind == "loss":
        # 0 = excluded, 1 = included
        lines = ["0 215 48 39 190", "1 26 152 80 225", "nv 0 0 0 0"]
        colors = ["#d73027", "#1a9850"]
    else:
        raise ValueError(kind)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return colors


def colorize(gdaldem: str, source: Path, color_file: Path, output: Path, discrete: bool = False) -> None:
    cmd = [
        gdaldem, "color-relief", str(source), str(color_file), str(output),
        "-alpha", "-of", "GTiff",
        "-co", "TILED=YES", "-co", "BLOCKXSIZE=512", "-co", "BLOCKYSIZE=512",
        "-co", "COMPRESS=DEFLATE", "-co", "PHOTOMETRIC=RGB", "-co", "SPARSE_OK=TRUE", "-co", "BIGTIFF=YES",
    ]
    if discrete:
        cmd.insert(5, "-nearest_color_entry")
    run(cmd)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def auto_zoom(detail_res_m: float, requested_min: int, requested_max: int) -> Tuple[int, int]:
    if requested_max >= 0:
        max_zoom = requested_max
    else:
        max_zoom = int(math.floor(math.log2(WEB_MERCATOR_INITIAL_RES / detail_res_m)))
    max_zoom = max(1, min(22, max_zoom))
    min_zoom = requested_min if requested_min >= 0 else max(4, max_zoom - 8)
    min_zoom = max(0, min(min_zoom, max_zoom))
    return min_zoom, max_zoom


def mercator_tile_range(
    bounds: Tuple[float, float, float, float], zoom: int
) -> Tuple[int, int, int, int, Tuple[float, float, float, float]]:
    """Return inclusive XYZ tile range and its exact EPSG:3857 extent."""
    left, top, right, bottom = bounds
    n = 1 << zoom
    span = (2.0 * WEB_MERCATOR_HALF_WORLD) / n
    eps = span * 1e-10

    x_min = math.floor((left + WEB_MERCATOR_HALF_WORLD) / span)
    x_max = math.floor((right + WEB_MERCATOR_HALF_WORLD - eps) / span)
    y_min = math.floor((WEB_MERCATOR_HALF_WORLD - top) / span)
    y_max = math.floor((WEB_MERCATOR_HALF_WORLD - bottom - eps) / span)

    x_min = max(0, min(n - 1, int(x_min)))
    x_max = max(0, min(n - 1, int(x_max)))
    y_min = max(0, min(n - 1, int(y_min)))
    y_max = max(0, min(n - 1, int(y_max)))
    if x_max < x_min or y_max < y_min:
        raise RuntimeError(f"Empty Web Mercator tile range at zoom {zoom}: {bounds}")

    xmin = -WEB_MERCATOR_HALF_WORLD + x_min * span
    xmax = -WEB_MERCATOR_HALF_WORLD + (x_max + 1) * span
    ymax = WEB_MERCATOR_HALF_WORLD - y_min * span
    ymin = WEB_MERCATOR_HALF_WORLD - (y_max + 1) * span
    return x_min, x_max, y_min, y_max, (xmin, ymin, xmax, ymax)


def _save_png_tile(tile: Image.Image, path: Path) -> bool:
    """Save one non-empty RGBA tile; return False for fully transparent tiles."""
    if tile.mode != "RGBA":
        tile = tile.convert("RGBA")
    alpha = tile.getchannel("A")
    if alpha.getbbox() is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tile.save(path, format="PNG", compress_level=6, optimize=False)
    return True


def _parent_tile_from_children(
    out_dir: Path,
    child_zoom: int,
    parent_zoom: int,
    parent_x: int,
    parent_y: int,
    resampling: str,
) -> bool:
    """Build one 256x256 parent XYZ tile from up to four child tiles.

    The finest XYZ level is rendered through temporary RGBA GeoTIFF strips from the EPSG:3857 display raster.
    Lower zooms are display overviews only. Continuous-color layers use bilinear
    RGBA downsampling; categorical masks use nearest-neighbour downsampling.
    """
    canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    found_child = False

    for dy in (0, 1):
        for dx in (0, 1):
            child_x = parent_x * 2 + dx
            child_y = parent_y * 2 + dy
            child_path = out_dir / str(child_zoom) / str(child_x) / f"{child_y}.png"
            if not child_path.is_file():
                continue
            with Image.open(child_path) as child_opened:
                child = child_opened.convert("RGBA")
                canvas.paste(child, (dx * 256, dy * 256))
            found_child = True

    if not found_child:
        return False

    if hasattr(Image, "Resampling"):
        method = (
            Image.Resampling.NEAREST
            if resampling == "near"
            else Image.Resampling.BILINEAR
        )
    else:  # Pillow compatibility fallback
        method = Image.NEAREST if resampling == "near" else Image.BILINEAR

    parent = canvas.resize((256, 256), resample=method)
    parent_path = out_dir / str(parent_zoom) / str(parent_x) / f"{parent_y}.png"
    return _save_png_tile(parent, parent_path)


def make_xyz_tiles_maxzoom_parent_pyramid(
    rgba_source: Path,
    out_dir: Path,
    min_zoom: int,
    max_zoom: int,
    processes: int,
    resampling: str,
) -> int:
    """Create sparse XYZ PNG tiles without gdal2tiles/osgeo_utils.

    Robust strategy
    ---------------
    1. Render only the finest requested XYZ zoom from the EPSG:3857 RGBA
       display raster, one 256-pixel-high GeoTIFF strip at a time.
    2. Construct all lower zooms from already-created child PNG tiles.

    This avoids asking GDAL to downsample a very large source raster directly
    into a tiny low-zoom VRT. Some older GDAL builds overflow internal integer
    calculations in that operation (for example when a ~40,000 x 28,000 source
    is read for a 256 x 256 low-zoom tile).

    Scientific scope
    ----------------
    This function creates display-only web tiles. It never writes to or modifies
    F010/F020 source rasters, and official metrics are not calculated here.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = gdal_info(rgba_source)
    bounds = raster_bounds(info)
    total = 0

    work = out_dir.parent / f".{out_dir.name}_maxzoom_parent_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    try:
        # ------------------------------------------------------------------
        # Finest zoom: render directly from the source, one global XYZ row at
        # a time. At this level the source-to-output scale is close to 1:1,
        # so GDAL only needs a narrow source window for each strip.
        # ------------------------------------------------------------------
        x_min, x_max, y_min, y_max, extent = mercator_tile_range(bounds, max_zoom)
        nx = x_max - x_min + 1
        ny = y_max - y_min + 1
        width = nx * 256

        world_tiles = 1 << max_zoom
        tile_span = (2.0 * WEB_MERCATOR_HALF_WORLD) / world_tiles
        xmin = -WEB_MERCATOR_HALF_WORLD + x_min * tile_span
        xmax = -WEB_MERCATOR_HALF_WORLD + (x_max + 1) * tile_span

        print(
            f"[XYZ-FINEST] z={max_zoom} range=x{x_min}-{x_max},"
            f"y{y_min}-{y_max} strip={width}x256 "
            f"candidate_tiles={nx * ny:,}",
            flush=True,
        )

        for global_y in range(y_min, y_max + 1):
            ymax = WEB_MERCATOR_HALF_WORLD - global_y * tile_span
            ymin = ymax - tile_span
            # GDAL 2.3 on the HPC cannot create PNG directly with gdalwarp.
            # Warp each finest-level strip to an uncompressed RGBA GeoTIFF,
            # then let Pillow read the strip and write the final 256x256 PNG
            # tiles. This is display-only and does not alter source rasters.
            strip_tif = work / f"z{max_zoom}_row{global_y}.tif"
            strip_tif.unlink(missing_ok=True)

            run([
                "gdalwarp", "-q", "-overwrite", "-of", "GTiff",
                "-t_srs", "EPSG:3857", "-te_srs", "EPSG:3857",
                "-te", f"{xmin:.9f}", f"{ymin:.9f}",
                f"{xmax:.9f}", f"{ymax:.9f}",
                "-ts", str(width), "256",
                "-r", resampling,
                "-srcalpha", "-dstalpha",
                "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
                "-co", "TILED=YES",
                "-co", "BLOCKXSIZE=512",
                "-co", "BLOCKYSIZE=256",
                "-co", "COMPRESS=NONE",
                "-co", "BIGTIFF=IF_SAFER",
                str(rgba_source), str(strip_tif),
            ])

            with Image.open(strip_tif) as opened:
                strip = opened.convert("RGBA")
                if strip.getchannel("A").getbbox() is None:
                    strip_tif.unlink(missing_ok=True)
                    continue

                jobs = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(1, processes)
                ) as pool:
                    for local_x in range(nx):
                        global_x = x_min + local_x
                        tile = strip.crop(
                            (local_x * 256, 0, (local_x + 1) * 256, 256)
                        )
                        tile_path = (
                            out_dir / str(max_zoom) /
                            str(global_x) / f"{global_y}.png"
                        )
                        jobs.append(pool.submit(_save_png_tile, tile, tile_path))
                    total += sum(1 for job in jobs if job.result())

            strip_tif.unlink(missing_ok=True)

        finest_count = sum(
            1 for _ in (out_dir / str(max_zoom)).rglob("*.png")
        ) if (out_dir / str(max_zoom)).exists() else 0
        print(
            f"[XYZ-FINEST-DONE] z={max_zoom} nonempty_tiles={finest_count:,}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # Lower zooms: create parents from child PNG tiles. This is both
        # standard for web pyramids and immune to the low-zoom GDAL overflow.
        # ------------------------------------------------------------------
        for parent_zoom in range(max_zoom - 1, min_zoom - 1, -1):
            child_zoom = parent_zoom + 1
            px_min, px_max, py_min, py_max, _ = mercator_tile_range(
                bounds, parent_zoom
            )
            candidates = [
                (px, py)
                for py in range(py_min, py_max + 1)
                for px in range(px_min, px_max + 1)
            ]

            print(
                f"[XYZ-PARENT] z={parent_zoom} from z={child_zoom} "
                f"range=x{px_min}-{px_max},y{py_min}-{py_max} "
                f"candidate_tiles={len(candidates):,}",
                flush=True,
            )

            created = 0
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, processes)
            ) as pool:
                futures = [
                    pool.submit(
                        _parent_tile_from_children,
                        out_dir,
                        child_zoom,
                        parent_zoom,
                        px,
                        py,
                        resampling,
                    )
                    for px, py in candidates
                ]
                created = sum(1 for future in futures if future.result())

            total += created
            print(
                f"[XYZ-PARENT-DONE] z={parent_zoom} "
                f"nonempty_tiles={created:,}",
                flush=True,
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    return total


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten_json(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def first_metric(flat: Dict[str, Any], exact_suffixes: Sequence[str], contains: Sequence[str]) -> Optional[float]:
    for suffix in exact_suffixes:
        for key, value in flat.items():
            if key.lower().endswith(suffix.lower()):
                try:
                    return float(value)
                except Exception:
                    pass
    for key, value in flat.items():
        low = key.lower()
        if all(token in low for token in contains):
            try:
                return float(value)
            except Exception:
                pass
    return None


def extract_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    flat = flatten_json(summary)
    return {
        "n_pixels": first_metric(flat, ["unique_n_pixels", "n_pixels", "valid_n_pixels"], ["pixel"]),
        "rmse_m": first_metric(flat, ["unique_rmse_m", "rmse_m", "rmse"], ["rmse"]),
        "mae_m": first_metric(flat, ["unique_mae_m", "mae_m", "mae"], ["mae"]),
        "bias_m": first_metric(flat, ["unique_bias_m", "bias_m", "bias"], ["bias"]),
    }


def infer_error_definition(summary: Dict[str, Any]) -> str:
    text = json.dumps(summary, ensure_ascii=False).lower()
    if "gt_minus_pred" in text or "gt - pred" in text or "ground truth minus prediction" in text:
        return "GT - Prediction"
    if "pred_minus_gt" in text or "pred - gt" in text or "prediction minus gt" in text:
        return "Prediction - GT"
    return "Signed error from F020"


def layer_marker_matches(marker: Path, signature: Dict[str, Any]) -> bool:
    if not marker.is_file():
        return False
    try:
        old = json.loads(marker.read_text())
        return old == signature
    except Exception:
        return False


def source_signature(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {"path": str(path.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def process_one(
    src: RiverSources,
    args: argparse.Namespace,
    work_root: Path,
    tiles_root: Path,
    gdaldem: str,
    min_zoom: int,
    max_zoom: int,
) -> Dict[str, Any]:
    key = f"{src.experiment}::{src.river}"
    safe_key = slug(f"{src.experiment}__{src.river}")
    work = work_root / safe_key
    work.mkdir(parents=True, exist_ok=True)
    print(f"\n[RECORD] {key}")

    input_vrt = work / "input_bathy.vrt"
    hidden_vrt = work / "hidden_mask.vrt"
    loss_vrt = work / "loss_mask.vrt"
    build_tile_vrt(src.input_bathy_tiles, input_vrt, work / "input_bathy_files.txt", args.nodata, args.nodata)
    build_tile_vrt(src.hidden_mask_tiles, hidden_vrt, work / "hidden_files.txt", args.mask_nodata, args.mask_nodata)
    build_tile_vrt(src.loss_mask_tiles, loss_vrt, work / "loss_files.txt", args.mask_nodata, args.mask_nodata)

    target_vrt = work / "target_grid_3857.vrt"
    target = make_target_grid(src.pred_vrt, target_vrt, args.detail_res_m, args.nodata)
    left, top, right, bottom = raster_bounds(target)
    west, south = mercator_to_lonlat(left, bottom)
    east, north = mercator_to_lonlat(right, top)
    bounds = [[south, west], [north, east]]

    warped = {
        "prediction": work / "prediction_3857.tif",
        "gt": work / "gt_3857.tif",
        "error": work / "error_signed_3857.tif",
        "input_bathy": work / "input_bathy_3857.tif",
        "hidden": work / "hidden_3857.tif",
        "loss": work / "loss_3857.tif",
    }
    warp_aligned(src.pred_vrt, warped["prediction"], target, args.detail_res_m, gdal_info(src.pred_vrt).nodata or args.nodata, args.nodata, "bilinear")
    warp_aligned(src.gt_vrt, warped["gt"], target, args.detail_res_m, gdal_info(src.gt_vrt).nodata or args.nodata, args.nodata, "bilinear")
    if not src.error_vrt:
        raise RuntimeError(
            "F020 signed-error VRT is required by the core-GDAL version; "
            "it intentionally avoids gdal_calc.py/osgeo_utils."
        )
    warp_aligned(
        src.error_vrt, warped["error"], target, args.detail_res_m,
        gdal_info(src.error_vrt).nodata or args.nodata, args.nodata, "bilinear"
    )
    warp_aligned(input_vrt, warped["input_bathy"], target, args.detail_res_m, args.nodata, args.nodata, "bilinear")
    warp_aligned(hidden_vrt, warped["hidden"], target, args.detail_res_m, args.mask_nodata, args.mask_nodata, "near", "Byte")
    warp_aligned(loss_vrt, warped["loss"], target, args.detail_res_m, args.mask_nodata, args.mask_nodata, "near", "Byte")

    # Small samples are only for robust display limits; official metrics remain from F020 JSON.
    samples_dir = work / "samples"
    samples_dir.mkdir(exist_ok=True)
    pred_vals = valid_values(sample_raster(warped["prediction"], samples_dir / "pred.tif", args.stats_max_px, args.nodata), args.nodata, args.nodata_threshold)
    gt_vals = valid_values(sample_raster(warped["gt"], samples_dir / "gt.tif", args.stats_max_px, args.nodata), args.nodata, args.nodata_threshold)
    input_vals = valid_values(sample_raster(warped["input_bathy"], samples_dir / "input.tif", args.stats_max_px, args.nodata), args.nodata, args.nodata_threshold)
    err_vals = valid_values(sample_raster(warped["error"], samples_dir / "err.tif", args.stats_max_px, args.nodata), args.nodata, -1e30)
    hidden_vals = mask_values(sample_raster(warped["hidden"], samples_dir / "hidden.tif", args.stats_max_px, args.mask_nodata, "near"), args.mask_nodata)
    loss_vals = mask_values(sample_raster(warped["loss"], samples_dir / "loss.tif", args.stats_max_px, args.mask_nodata, "near"), args.mask_nodata)

    elev_min, elev_max = robust_limits([pred_vals, gt_vals], args.elev_low_pct, args.elev_high_pct)
    input_min, input_max = robust_limits([input_vals], args.elev_low_pct, args.elev_high_pct)
    err_max = error_limit(err_vals, args.error_abs_pct)

    colors_dir = work / "colors"
    colors_dir.mkdir(exist_ok=True)
    elev_colors = write_color_file(colors_dir / "elevation.txt", elev_min, elev_max, args.elev_cmap)
    input_colors = write_color_file(colors_dir / "input.txt", input_min, input_max, args.elev_cmap)
    error_colors = write_color_file(colors_dir / "error.txt", -err_max, err_max, args.error_cmap)
    abs_colors = write_abs_error_color_file(colors_dir / "abs_error.txt", err_max, args.abs_error_cmap)
    hidden_colors = write_mask_color_file(colors_dir / "hidden.txt", "hidden")
    loss_colors = write_mask_color_file(colors_dir / "loss.txt", "loss")

    colorized = {
        "prediction": work / "prediction_rgba.tif",
        "gt": work / "gt_rgba.tif",
        "error": work / "error_signed_rgba.tif",
        "abs_error": work / "error_absolute_rgba.tif",
        "input_bathy": work / "input_bathy_rgba.tif",
        "hidden": work / "hidden_rgba.tif",
        "loss": work / "loss_rgba.tif",
    }
    palette = {
        "prediction": (colors_dir / "elevation.txt", False),
        "gt": (colors_dir / "elevation.txt", False),
        "error": (colors_dir / "error.txt", False),
        "abs_error": (colors_dir / "abs_error.txt", False),
        "input_bathy": (colors_dir / "input.txt", False),
        "hidden": (colors_dir / "hidden.txt", True),
        "loss": (colors_dir / "loss.txt", True),
    }

    tile_urls: Dict[str, str] = {}
    tile_counts: Dict[str, int] = {}
    for layer_name in ("prediction", "gt", "error", "abs_error", "input_bathy", "hidden", "loss"):
        layer_tiles = tiles_root / safe_key / layer_name
        marker = layer_tiles / ".F045_complete.json"
        source_layer = "error" if layer_name == "abs_error" else layer_name
        pyramid_resampling = "near" if layer_name in ("hidden", "loss") else "bilinear"
        signature = {
            "version": 4,
            "tiler": "core_gdal_gtiff_strip_maxzoom_parent_xyz",
            "source": source_signature(warped[source_layer]),
            "detail_res_m": args.detail_res_m,
            "min_zoom": min_zoom,
            "max_zoom": max_zoom,
            "pyramid_resampling": pyramid_resampling,
            "scale": {
                "elev": [elev_min, elev_max],
                "input": [input_min, input_max],
                "error": [-err_max, err_max],
            },
            "layer": layer_name,
        }
        if layer_marker_matches(marker, signature):
            print(f"[RESUME] tiles already complete: {key} / {layer_name}")
            count = sum(1 for _ in layer_tiles.rglob("*.png"))
        else:
            colorize(
                gdaldem, warped[source_layer], palette[layer_name][0],
                colorized[layer_name], palette[layer_name][1]
            )
            count = make_xyz_tiles_maxzoom_parent_pyramid(
                colorized[layer_name], layer_tiles, min_zoom, max_zoom,
                args.tile_processes, pyramid_resampling
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        tile_urls[layer_name] = f"tiles/{safe_key}/{layer_name}/{{z}}/{{x}}/{{y}}.png"
        tile_counts[layer_name] = count
        print(f"[TILES] {layer_name}: {count:,}")

    metrics = extract_metrics(src.error_summary)
    record = {
        "key": key,
        "safe_key": safe_key,
        "experiment": src.experiment,
        "river": src.river,
        "bounds": bounds,
        "center": [(south + north) / 2.0, (west + east) / 2.0],
        "min_zoom": min_zoom,
        "max_zoom": max_zoom,
        "detail_res_m": args.detail_res_m,
        "xyz_finest_res_m": WEB_MERCATOR_INITIAL_RES / (2 ** max_zoom),
        "tile_urls": tile_urls,
        "tile_counts": tile_counts,
        "elev_min": elev_min,
        "elev_max": elev_max,
        "input_elev_min": input_min,
        "input_elev_max": input_max,
        "error_max": err_max,
        "elev_colors": elev_colors,
        "input_colors": input_colors,
        "error_colors": error_colors,
        "abs_error_colors": abs_colors,
        "hidden_colors": hidden_colors,
        "loss_colors": loss_colors,
        "error_definition": infer_error_definition(src.error_summary),
        "metrics": metrics,
        "input_stats": {
            "n_tiles": len(src.input_bathy_tiles),
            "hidden_fraction": float(np.nanmean(hidden_vals)) if hidden_vals.size else None,
            "loss_fraction": float(np.nanmean(loss_vals)) if loss_vals.size else None,
        },
        "source": {
            "prediction_vrt": str(src.pred_vrt),
            "gt_vrt": str(src.gt_vrt),
            "error_vrt": str(src.error_vrt) if src.error_vrt else None,
            "prediction_summary": str(src.pred_summary_path) if src.pred_summary_path else None,
            "error_summary": str(src.error_summary_path) if src.error_summary_path else None,
            "target_width": target.width,
            "target_height": target.height,
            "tile_root": str(args.tile_root.resolve()),
        },
        "visualization_policy": {
            "display_only": True,
            "source_rasters_modified": False,
            "target_crs": "EPSG:3857",
            "xyz_tiler": "built-in core-GDAL finest-zoom plus parent-pyramid tiler (no gdal2tiles/osgeo_utils)",
            "display_resolution_m": args.detail_res_m,
            "continuous_resampling": "bilinear",
            "binary_mask_resampling": "nearest-neighbour",
            "official_metrics_source": "F020 summary JSON at native/source resolution",
            "elevation_display_percentiles": [args.elev_low_pct, args.elev_high_pct],
            "absolute_error_display_percentile": args.error_abs_pct,
        },
    }

    if not args.keep_intermediate:
        # Tile package and manifest are retained; large warped/colorized work rasters are removed.
        shutil.rmtree(work, ignore_errors=True)
    return record


def gradient_css(colors: Sequence[str]) -> str:
    return "linear-gradient(to right," + ",".join(colors) + ")"


def html_template(records: List[Dict[str, Any]], manifest: Dict[str, Any], default_opacity: float) -> str:
    data_json = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    manifest_json = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Full-River MAE High-Detail Satellite Dashboard</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root {{ --bg:#eef1f4; --panel:#fff; --text:#17202a; --muted:#5d6d7e; --border:#ccd4dc; --accent:#165d9c; --header-h:236px; }}
* {{ box-sizing:border-box; }}
html,body {{ margin:0;width:100%;height:100%;overflow:hidden;font-family:Arial,Helvetica,sans-serif;color:var(--text);background:var(--bg); }}
header {{ height:var(--header-h);background:var(--panel);border-bottom:1px solid var(--border);padding:9px 12px;display:grid;grid-template-columns:minmax(430px,1fr) minmax(520px,1.35fr);grid-template-rows:auto auto;gap:8px 15px; }}
.title {{ font-size:20px;font-weight:700;margin-bottom:7px; }}
.controls {{ display:grid;grid-template-columns:1fr 1fr;gap:6px 9px; }}
label {{ display:block;font-size:10px;font-weight:700;color:var(--muted);margin-bottom:2px; }}
select,button,input[type=range] {{ width:100%;height:31px;border:1px solid var(--border);border-radius:5px;background:#fff;padding:3px 7px;font-size:12px; }}
button {{ cursor:pointer;font-weight:700; }} button:hover {{ background:#f3f7fb; }}
.tabs {{ display:flex;gap:7px;margin-top:7px; }} .tabBtn {{ width:auto;min-width:140px; }} .tabBtn.active {{ background:var(--accent);color:#fff;border-color:var(--accent); }}
.help {{ font-size:10px;color:var(--muted);margin-top:5px;line-height:1.3; }}
.displayNotice {{ grid-column:1 / -1;border:1px solid #8bb8df;border-left:5px solid #165d9c;border-radius:6px;background:#eef7ff;padding:7px 10px;font-size:11px;line-height:1.35;color:#20364a; }}
.displayNotice strong {{ color:#0f4f86; }}
.displayNotice .sub {{ color:#4f6578; }}
.summary {{ display:grid;grid-template-rows:auto auto 1fr;min-width:0; }}
.metrics {{ display:grid;grid-template-columns:repeat(4,minmax(90px,1fr));gap:7px; }}
.metric {{ border:1px solid var(--border);border-radius:5px;padding:5px 8px;background:#fafbfd; }}
.metric .name {{ font-size:10px;color:var(--muted);font-weight:700; }} .metric .value {{ font-size:15px;font-weight:700;margin-top:2px; }}
.scaleRows {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin-top:7px; }}
.scaleTitle {{ font-size:10px;font-weight:700;color:var(--muted);display:flex;justify-content:space-between;gap:7px; }}
.gradient {{ height:11px;border:1px solid #777;border-radius:2px;margin-top:2px; }}
#status {{ font-size:10px;color:var(--muted);margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }}
main {{ height:calc(100vh - var(--header-h));padding:7px; }}
.view {{ width:100%;height:100%;display:none; }} .view.active {{ display:block; }}
.compareGrid {{ display:grid;grid-template-columns:1fr 1fr;gap:7px;width:100%;height:100%; }}
.inputGrid {{ display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;width:100%;height:100%; }}
.panel {{ position:relative;background:#20252b;border:1px solid #69737e;border-radius:6px;overflow:hidden;min-width:0;min-height:0; }}
.map {{ width:100%;height:100%;background:#20252b; }}
.panelTitle {{ position:absolute;z-index:800;top:7px;left:48px;background:rgba(255,255,255,.94);padding:5px 8px;border-radius:4px;font-weight:700;font-size:12px;box-shadow:0 1px 5px rgba(0,0,0,.25);pointer-events:none; }}
.mapBadge {{ position:absolute;z-index:800;right:7px;bottom:7px;background:rgba(255,255,255,.92);padding:4px 7px;border-radius:4px;font-size:10px;font-weight:700;box-shadow:0 1px 5px rgba(0,0,0,.25);pointer-events:none; }}
.errorToolbar {{ position:absolute;z-index:850;top:7px;right:7px;display:flex;gap:6px;width:330px;max-width:calc(100% - 180px); }}
.errorToolbar select,.errorToolbar button {{ height:30px; }}
.leaflet-container {{ font-family:Arial,Helvetica,sans-serif; }}
.leaflet-tile-pane img {{ image-rendering:auto; }}
.pixelated .leaflet-overlay-pane img,.pixelated .leaflet-tile-pane img {{ image-rendering:pixelated;image-rendering:crisp-edges; }}
@media(max-width:950px) {{ :root{{--header-h:335px;}} header{{grid-template-columns:1fr;overflow-y:auto;}} .compareGrid{{grid-template-columns:1fr;grid-template-rows:1fr 1fr;}} .inputGrid{{grid-template-columns:1fr;grid-template-rows:repeat(3,1fr);}} }}
</style>
</head>
<body>
<header>
<section>
  <div class="title">Full-River MAE High-Detail Satellite Dashboard</div>
  <div class="controls">
    <div><label>Experiment</label><select id="experimentSelect"></select></div>
    <div><label>River</label><select id="riverSelect"></select></div>
    <div><label>Basemap</label><select id="basemapSelect"><option value="satellite">Satellite imagery</option><option value="light">Light gray map</option><option value="none">No basemap</option></select></div>
    <div><label>Raster opacity <span id="opacityValue"></span></label><input id="opacitySlider" type="range" min="0" max="1" step="0.02" value="{max(0.0,min(1.0,default_opacity)):.2f}" /></div>
    <div><button id="fitBtn">Fit current river</button></div>
    <div><button id="nativeBtn">Display-detail zoom</button></div>
  </div>
  <div class="tabs"><button class="tabBtn active" data-tab="compareView">Prediction + GT</button><button class="tabBtn" data-tab="errorView">Error</button><button class="tabBtn" data-tab="inputView">Input Tile + Masks</button></div>
  <div class="help">Maps in the current tab stay synchronized. Local MAE overlays work offline; satellite/light-gray basemaps require Internet. At display-detail zoom, Leaflet loads the finest generated local tiles rather than stretching one overview PNG.</div>
</section>
<section class="summary">
  <div class="metrics">
    <div class="metric"><div class="name">N PIXELS</div><div class="value" id="metricN">—</div></div>
    <div class="metric"><div class="name">RMSE (m)</div><div class="value" id="metricRMSE">—</div></div>
    <div class="metric"><div class="name">MAE (m)</div><div class="value" id="metricMAE">—</div></div>
    <div class="metric"><div class="name">BIAS (m)</div><div class="value" id="metricBias">—</div></div>
  </div>
  <div class="scaleRows">
    <div><div class="scaleTitle"><span>Prediction / GT elevation (m)</span><span id="elevRange">—</span></div><div id="elevGradient" class="gradient"></div></div>
    <div><div class="scaleTitle"><span id="errorScaleTitle">Signed error (m)</span><span id="errorRange">—</span></div><div id="errorGradient" class="gradient"></div></div>
    <div><div class="scaleTitle"><span>Raw input bathymetry (m)</span><span id="inputRange">—</span></div><div id="inputGradient" class="gradient"></div></div>
  </div>
  <div id="status"></div>
</section>
<div class="displayNotice">
  <strong>DISPLAY-ONLY RESAMPLING:</strong> MAE raster overlays are reprojected to an intermediate EPSG:3857 display grid of <strong>{manifest.get("detail_res_m", "—")} m/pixel</strong>. The finest XYZ web zoom is <strong>{manifest.get("xyz_finest_res_m", "—")} m/pixel</strong> at the equator. The original F010/F020 rasters and prediction values are not modified. <strong>RMSE, MAE, Bias, and N Pixels come directly from the native/source-resolution F020 summary JSON</strong> and are not recalculated from these display tiles.<br>
  <span class="sub">Continuous layers (Prediction, GT, Error, raw bathymetry) use bilinear resampling; binary Hidden/Loss masks use nearest-neighbour resampling. Display colors use robust percentile clipping ({manifest.get("color_scaling", {}).get("elevation_low_pct", "—")}–{manifest.get("color_scaling", {}).get("elevation_high_pct", "—")}% for elevation/input; {manifest.get("color_scaling", {}).get("error_abs_pct", "—")}th percentile of |error|). Values outside the display range are color-saturated only; source data remain unchanged. Satellite imagery is contextual background and is not model input or evaluation data.</span>
</div>
</header>
<main>
<div id="compareView" class="view active"><div class="compareGrid">
  <div class="panel"><div class="panelTitle">Prediction</div><div id="predMap" class="map"></div><div id="predBadge" class="mapBadge"></div></div>
  <div class="panel"><div class="panelTitle">Ground Truth</div><div id="gtMap" class="map"></div><div id="gtBadge" class="mapBadge"></div></div>
</div></div>
<div id="errorView" class="view"><div class="panel" style="width:100%;height:100%;">
  <div id="errorTitle" class="panelTitle">Signed Error</div>
  <div class="errorToolbar"><select id="errorMode"><option value="error">Signed error</option><option value="abs_error">Absolute error</option></select><button id="errorFitBtn">Fit</button></div>
  <div id="errorMap" class="map"></div><div id="errorBadge" class="mapBadge"></div>
</div></div>
<div id="inputView" class="view"><div class="inputGrid">
  <div class="panel"><div class="panelTitle">Raw Input Bathymetry</div><div id="inputMap" class="map"></div><div id="inputBadge" class="mapBadge"></div></div>
  <div class="panel"><div id="hiddenTitle" class="panelTitle">Hidden Mask</div><div id="hiddenMap" class="map"></div><div class="mapBadge">Gray = visible (0) · Yellow = hidden (1)</div></div>
  <div class="panel"><div id="lossTitle" class="panelTitle">Loss Mask Pixel</div><div id="lossMap" class="map"></div><div class="mapBadge">Red = excluded (0) · Green = included (1)</div></div>
</div></div>
</main>
<script>
const records={data_json};
const manifest={manifest_json};
const byKey=Object.fromEntries(records.map(r=>[r.key,r]));
const experimentSelect=document.getElementById('experimentSelect');
const riverSelect=document.getElementById('riverSelect');
const opacitySlider=document.getElementById('opacitySlider');
const opacityValue=document.getElementById('opacityValue');
let activeTab='compareView'; let currentRecord=null; let errorMode='error'; let syncing=false;

const mapIds=['predMap','gtMap','errorMap','inputMap','hiddenMap','lossMap'];
const maps=Object.fromEntries(mapIds.map(id=>[id,L.map(id,{{preferCanvas:true,zoomControl:true,attributionControl:true}})]));
const layerState=Object.fromEntries(mapIds.map(id=>[id,{{base:null,overlay:null}}]));

function satelliteLayer(){{return L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',{{maxZoom:20,attribution:'Tiles &copy; Esri'}});}}
function lightLayer(){{return L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{subdomains:'abcd',maxZoom:20,attribution:'&copy; OpenStreetMap contributors &copy; CARTO'}});}}
function makeBase(kind){{if(kind==='satellite')return satelliteLayer();if(kind==='light')return lightLayer();return null;}}
function applyBasemap(){{const kind=document.getElementById('basemapSelect').value;Object.entries(maps).forEach(([id,m])=>{{if(layerState[id].base)m.removeLayer(layerState[id].base);layerState[id].base=makeBase(kind);if(layerState[id].base)layerState[id].base.addTo(m);}});}}
function installOverlay(mapId,layerName){{const m=maps[mapId];if(layerState[mapId].overlay)m.removeLayer(layerState[mapId].overlay);if(!currentRecord)return;const url=currentRecord.tile_urls[layerName];const l=L.tileLayer(url,{{minZoom:currentRecord.min_zoom,maxNativeZoom:currentRecord.max_zoom,maxZoom:currentRecord.max_zoom+2,opacity:Number(opacitySlider.value),tms:false,noWrap:true,keepBuffer:3,updateWhenIdle:false,errorTileUrl:''}});l.addTo(m);layerState[mapId].overlay=l;}}
function installAllOverlays(){{installOverlay('predMap','prediction');installOverlay('gtMap','gt');installOverlay('errorMap',errorMode);installOverlay('inputMap','input_bathy');installOverlay('hiddenMap','hidden');installOverlay('lossMap','loss');}}
function setOpacity(){{const v=Number(opacitySlider.value);opacityValue.textContent=Math.round(v*100)+'%';Object.values(layerState).forEach(s=>{{if(s.overlay)s.overlay.setOpacity(v);}});}}

function syncGroup(ids){{ids.forEach(id=>{{maps[id].on('move zoom',()=>{{if(syncing)return;syncing=true;const c=maps[id].getCenter(),z=maps[id].getZoom();ids.forEach(other=>{{if(other!==id)maps[other].setView(c,z,{{animate:false,reset:false}});}});syncing=false;updateBadges();}});}});}}
syncGroup(['predMap','gtMap']);syncGroup(['inputMap','hiddenMap','lossMap']);
function activeMapIds(){{return activeTab==='compareView'?['predMap','gtMap']:(activeTab==='errorView'?['errorMap']:['inputMap','hiddenMap','lossMap']);}}
function fitActive(){{if(!currentRecord)return;activeMapIds().forEach(id=>maps[id].fitBounds(currentRecord.bounds,{{padding:[18,18],animate:false}}));setTimeout(()=>activeMapIds().forEach(id=>maps[id].invalidateSize()),20);}}
function nativeZoom(){{if(!currentRecord)return;const ids=activeMapIds();const center=maps[ids[0]].getCenter();ids.forEach(id=>maps[id].setView(center,currentRecord.max_zoom,{{animate:false}}));}}
function updateBadges(){{if(!currentRecord)return;const text=`GRID ${{currentRecord.detail_res_m}} m · XYZ finest ${{fmt(currentRecord.xyz_finest_res_m,3)}} m · resampled · z${{currentRecord.max_zoom}}`;document.getElementById('predBadge').textContent=text;document.getElementById('gtBadge').textContent=text;document.getElementById('errorBadge').textContent=text;document.getElementById('inputBadge').textContent=text;}}

function option(v,t=v){{const o=document.createElement('option');o.value=v;o.textContent=t;return o;}}
[...new Set(records.map(r=>r.experiment))].sort().forEach(e=>experimentSelect.appendChild(option(e)));
function populateRivers(){{const exp=experimentSelect.value;const old=riverSelect.value;riverSelect.innerHTML='';records.filter(r=>r.experiment===exp).sort((a,b)=>a.river.localeCompare(b.river)).forEach(r=>riverSelect.appendChild(option(r.river)));if([...riverSelect.options].some(o=>o.value===old))riverSelect.value=old;loadCurrent();}}
function fmt(x,d=3){{return x===null||x===undefined||!Number.isFinite(Number(x))?'—':Number(x).toFixed(d);}}
function fmtInt(x){{return x===null||x===undefined||!Number.isFinite(Number(x))?'—':Math.round(Number(x)).toLocaleString();}}
function setGradient(id,colors){{document.getElementById(id).style.background='linear-gradient(to right,'+colors.join(',')+')';}}
function loadCurrent(){{const r=byKey[`${{experimentSelect.value}}::${{riverSelect.value}}`];if(!r)return;currentRecord=r;installAllOverlays();Object.values(maps).forEach(m=>{{m.setMinZoom(r.min_zoom);m.setMaxZoom(r.max_zoom+2);m.fitBounds(r.bounds,{{padding:[18,18],animate:false}});setTimeout(()=>m.invalidateSize(),25);}});document.getElementById('metricN').textContent=fmtInt(r.metrics.n_pixels);document.getElementById('metricRMSE').textContent=fmt(r.metrics.rmse_m);document.getElementById('metricMAE').textContent=fmt(r.metrics.mae_m);document.getElementById('metricBias').textContent=fmt(r.metrics.bias_m);document.getElementById('elevRange').textContent=`${{fmt(r.elev_min,2)}} to ${{fmt(r.elev_max,2)}}`;document.getElementById('inputRange').textContent=`${{fmt(r.input_elev_min,2)}} to ${{fmt(r.input_elev_max,2)}}`;document.getElementById('errorRange').textContent=errorMode==='error'?`${{fmt(-r.error_max,2)}} to ${{fmt(r.error_max,2)}}`:`0 to ${{fmt(r.error_max,2)}}`;document.getElementById('errorScaleTitle').textContent=errorMode==='error'?`Signed error: ${{r.error_definition}} (m)`:'Absolute error (m)';setGradient('elevGradient',r.elev_colors);setGradient('inputGradient',r.input_colors);setGradient('errorGradient',errorMode==='error'?r.error_colors:r.abs_error_colors);document.getElementById('hiddenTitle').textContent=`Hidden Mask · mean hidden fraction ${{fmt(r.input_stats.hidden_fraction,3)}}`;document.getElementById('lossTitle').textContent=`Loss Mask Pixel · mean included fraction ${{fmt(r.input_stats.loss_fraction,3)}}`;const total=Object.values(r.tile_counts).reduce((a,b)=>a+b,0);document.getElementById('status').textContent=`${{r.experiment}} → ${{r.river}} · E001 tiles ${{r.input_stats.n_tiles.toLocaleString()}} · display grid ${{r.source.target_width.toLocaleString()}}×${{r.source.target_height.toLocaleString()}} at ${{r.detail_res_m}} m in EPSG:3857 (resampled) · local PNG tiles ${{total.toLocaleString()}}`;updateBadges();}}
function updateError(){{if(!currentRecord)return;installOverlay('errorMap',errorMode);document.getElementById('errorTitle').textContent=errorMode==='error'?`Signed Error (${{currentRecord.error_definition}})`:'Absolute Error';document.getElementById('errorRange').textContent=errorMode==='error'?`${{fmt(-currentRecord.error_max,2)}} to ${{fmt(currentRecord.error_max,2)}}`:`0 to ${{fmt(currentRecord.error_max,2)}}`;document.getElementById('errorScaleTitle').textContent=errorMode==='error'?`Signed error: ${{currentRecord.error_definition}} (m)`:'Absolute error (m)';setGradient('errorGradient',errorMode==='error'?currentRecord.error_colors:currentRecord.abs_error_colors);}}

experimentSelect.addEventListener('change',populateRivers);riverSelect.addEventListener('change',loadCurrent);document.getElementById('basemapSelect').addEventListener('change',applyBasemap);opacitySlider.addEventListener('input',setOpacity);document.getElementById('fitBtn').addEventListener('click',fitActive);document.getElementById('nativeBtn').addEventListener('click',nativeZoom);document.getElementById('errorFitBtn').addEventListener('click',()=>maps.errorMap.fitBounds(currentRecord.bounds,{{padding:[18,18]}}));document.getElementById('errorMode').addEventListener('change',e=>{{errorMode=e.target.value;updateError();}});
document.querySelectorAll('.tabBtn').forEach(btn=>btn.addEventListener('click',()=>{{activeTab=btn.dataset.tab;document.querySelectorAll('.tabBtn').forEach(x=>x.classList.toggle('active',x===btn));document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id===activeTab));setTimeout(()=>{{activeMapIds().forEach(id=>maps[id].invalidateSize());fitActive();}},30);}}));
applyBasemap();setOpacity();populateRivers();
</script>
</body>
</html>
"""


def write_readme(
    out_dir: Path,
    html_name: str,
    zip_name: str,
    detail_res_m: float,
    elev_low_pct: float,
    elev_high_pct: float,
    error_abs_pct: float,
    max_zoom: int,
) -> None:
    text = f"""F045 Full-River MAE High-Detail Satellite Dashboard

Open:
  {html_name}

IMPORTANT SCIENTIFIC INTERPRETATION NOTICE
------------------------------------------
This package is a DISPLAY-ONLY visualization product. It does not alter, replace,
or recompute the original model outputs.

1. Display reprojection/resampling
   The Prediction, GT, Error, and E001 input rasters are reprojected to EPSG:3857
   onto an intermediate {detail_res_m} metre display grid. The finest XYZ zoom is
   z={max_zoom}, corresponding to approximately {WEB_MERCATOR_INITIAL_RES / (2 ** max_zoom):.6f} metres per web pixel at the
   equator. Lower zooms are display overviews built from child PNG tiles.

2. Source rasters remain unchanged
   The original F010/F020 VRT/TIFF files are read-only inputs. This script never
   writes back to them and never changes their stored values.

3. Official metrics
   N Pixels, RMSE, MAE, and Bias shown in the HTML are read directly from the
   existing F020 summary JSON at native/source resolution. They are NOT recomputed
   from either the intermediate {detail_res_m} m display grid or the XYZ web tiles.

4. Resampling methods
   Continuous layers: Prediction, GT, signed/absolute Error, and raw bathymetry
     -> bilinear resampling for display.
   Binary masks: Hidden Mask and Loss Mask Pixel
     -> nearest-neighbour resampling to preserve classes 0 and 1.

5. Display color ranges
   Prediction/GT and input bathymetry use robust display ranges based on the
   {elev_low_pct:g}th to {elev_high_pct:g}th percentiles. Error uses the
   {error_abs_pct:g}th percentile of absolute error as the symmetric display limit.
   Values outside these ranges are only saturated to the end colors. Source raster
   values and official statistics remain unchanged.

6. Satellite imagery
   Satellite/light-gray basemaps are contextual online layers. They are not MAE
   inputs, are not used in prediction, and are not used in evaluation. Internet
   access is required for these basemaps. Local MAE overlay tiles work offline.

For a sharper but larger display package, rerun F044 with DETAIL_RES_M=2.
For a smaller display package, use DETAIL_RES_M=8.

Do not open the HTML while it is still inside the ZIP. Extract the complete folder
first so the relative tiles/ paths remain available.

Archive name:
  {zip_name}
"""
    (out_dir / "README_F045_SCIENTIFIC_NOTICE.txt").write_text(text, encoding="utf-8")


def make_zip(package_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    print(f"[ZIP] creating {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        base_parent = package_dir.parent
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(base_parent))
    print(f"[ZIP] size={zip_path.stat().st_size / (1024**3):.3f} GiB")


def main() -> None:
    args = parse_args()
    if args.detail_res_m <= 0:
        raise ValueError("detail_res_m must be positive")
    for exe in ("gdalinfo", "gdalwarp", "gdalbuildvrt", "gdal_translate"):
        find_executable([exe])
    gdaldem = find_executable(["gdaldem"])

    min_zoom, max_zoom = auto_zoom(args.detail_res_m, args.min_zoom, args.max_zoom)
    out_dir = args.out_dir.resolve()
    zip_path = out_dir.parent / args.zip_name
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and zip_path.exists():
        zip_path.unlink()

    tiles_root = out_dir / "tiles"
    work_root = out_dir / "_work"
    work_root.mkdir(parents=True, exist_ok=True)
    sources, discovery_warnings = discover_sources(args)

    print("============================================================")
    print("F045 high-detail holdout satellite dashboard with core-GDAL tiler and scientific resampling notice")
    print(f"records={len(sources)} detail_res_m={args.detail_res_m} zoom={min_zoom}-{max_zoom}")
    print(f"out_dir={out_dir}")
    print("============================================================")

    records: List[Dict[str, Any]] = []
    failures: List[str] = []
    for src in sources:
        try:
            records.append(process_one(src, args, work_root, tiles_root, gdaldem, min_zoom, max_zoom))
        except Exception as exc:
            msg = f"FAILED {src.experiment}/{src.river}: {exc}"
            failures.append(msg)
            print(f"[ERROR] {msg}", file=sys.stderr)
            raise

    manifest = {
        "generated_at_unix": time.time(),
        "generator": Path(__file__).name,
        "experiments": list(args.experiments),
        "detail_res_m": args.detail_res_m,
        "display_only": True,
        "source_rasters_modified": False,
        "target_crs": "EPSG:3857",
        "xyz_tiler": "built-in core-GDAL finest-zoom plus parent-pyramid tiler (no gdal2tiles/osgeo_utils)",
        "resampling": {
            "continuous_layers": "bilinear",
            "binary_masks": "nearest-neighbour",
        },
        "color_scaling": {
            "elevation_low_pct": args.elev_low_pct,
            "elevation_high_pct": args.elev_high_pct,
            "error_abs_pct": args.error_abs_pct,
            "values_outside_range": "color-saturated for display only; source values unchanged",
        },
        "official_metrics": {
            "source": "F020 summary JSON",
            "resolution": "native/source resolution",
            "recomputed_from_display_tiles": False,
        },
        "min_zoom": min_zoom,
        "max_zoom": max_zoom,
        "xyz_finest_res_m": WEB_MERCATOR_INITIAL_RES / (2 ** max_zoom),
        "records": records,
        "discovery_warnings": discovery_warnings,
        "failures": failures,
        "notes": [
            "The finest local XYZ zoom is rendered through temporary RGBA GeoTIFF strips from the intermediate EPSG:3857 display raster. Lower zooms are generated from child PNG tiles. Local overlay tiles are XYZ PNG tiles generated by the built-in core-GDAL finest-zoom plus parent-pyramid tiler; gdal2tiles/osgeo_utils is not required.",
            "Satellite/light-gray basemaps require Internet access in the browser.",
            "Official metrics come from native/source-resolution F020 summary JSON and are not recomputed from display tiles.",
            "The display overlays are EPSG:3857 resampled visualization products; original F010/F020 rasters are unchanged.",
            "Continuous rasters use bilinear resampling; binary masks use nearest-neighbour resampling.",
            "Robust percentile clipping affects colors only, not raster values or official metrics.",
        ],
    }
    html_path = out_dir / args.out_html
    html_path.write_text(html_template(records, manifest, args.overlay_opacity), encoding="utf-8")
    (out_dir / "F045_dashboard_manifest_with_scientific_notice.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(
        out_dir,
        args.out_html,
        args.zip_name,
        args.detail_res_m,
        args.elev_low_pct,
        args.elev_high_pct,
        args.error_abs_pct,
        max_zoom,
    )

    if not args.keep_intermediate:
        shutil.rmtree(work_root, ignore_errors=True)
    if not args.no_zip:
        make_zip(out_dir, zip_path)

    print("============================================================")
    print(f"HTML: {html_path}")
    print(f"ZIP : {zip_path if not args.no_zip else 'disabled'}")
    print("Extract the ZIP before opening the HTML.")
    print("============================================================")


if __name__ == "__main__":
    main()
