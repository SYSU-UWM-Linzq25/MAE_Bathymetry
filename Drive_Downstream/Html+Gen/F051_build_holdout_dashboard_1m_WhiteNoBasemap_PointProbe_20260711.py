#!/usr/bin/env python3
"""Build a high-detail Leaflet dashboard package for three MAE holdout experiments.

Why this version exists
-----------------------
F034 embedded one downsampled PNG per layer in a single HTML.  That design is
convenient, but no browser can recover details that were removed during the
quicklook downsampling.  It also used a checkerboard to indicate transparent
NoData pixels.

F051 creates local XYZ raster tiles and displays them in Leaflet over an
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
  visualization-only resampling setting. This revision defaults to 1 m for a
  high-detail trial. The resulting package is substantially larger than 4 m.
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
    "FullRiver_WebMap_F051_Holdout_1m_Probe"
)
DEFAULT_HTML_NAME = "F051_dashboard_1m_probe.html"
DEFAULT_ZIP_NAME = "F051_dashboard_1m_probe_package.zip"
DEFAULT_EXPERIMENTS = (
    "holdout_CA_D001NoDataSafe",
    "holdout_CO_D001NoDataSafe",
    "holdout_Santiam_D001NoDataSafe",
)
NODATA_DEFAULT = -999999.0
MASK_NODATA_DEFAULT = 255.0
WEB_MERCATOR_INITIAL_RES = 156543.03392804097
WEB_MERCATOR_HALF_WORLD = 20037508.342789244
PROBE_RGB_OFFSET = 1 << 23
PROBE_RGB_MIN = -(1 << 23)
PROBE_RGB_MAX = (1 << 23) - 1


@dataclass
class RasterInfo:
    path: Path
    width: int
    height: int
    nodata: Optional[float]
    geotransform: Optional[Tuple[float, float, float, float, float, float]]
    srs_wkt: Optional[str] = None
    srs_authority: Optional[str] = None


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
        default=1.0,
        help="Intermediate display-grid pixel size in EPSG:3857 metres. This does not alter source rasters.",
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
    p.add_argument("--overlay_opacity", type=float, default=0.90)
    p.add_argument("--probe_quantization_mm", type=float, default=1.0, help="Point-probe quantization in millimetres for continuous layers.")
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


def _extract_epsg_from_wkt(wkt: str) -> Optional[str]:
    """Best-effort EPSG extraction from GDAL WKT1/WKT2."""
    if not wkt:
        return None
    patterns = (
        r'AUTHORITY\s*\[\s*"EPSG"\s*,\s*"(\d+)"\s*\]',
        r'ID\s*\[\s*"EPSG"\s*,\s*(\d+)\s*\]',
    )
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, wkt, flags=re.IGNORECASE))
    if not matches:
        return None
    # In common GDAL WKT output, the projected CRS authority is the final EPSG
    # identifier. We still retain the full WKT as the authoritative fallback.
    return f"EPSG:{matches[-1]}"


def gdal_info(path: Path) -> RasterInfo:
    raw = json.loads(run(["gdalinfo", "-json", str(path)], capture=True))
    width, height = map(int, raw["size"])
    bands = raw.get("bands") or []
    nodata = bands[0].get("noDataValue") if bands else None
    gt = raw.get("geoTransform")

    cs = raw.get("coordinateSystem") or {}
    wkt = cs.get("wkt") or raw.get("wkt")
    wkt = str(wkt).strip() if wkt else None
    authority = _extract_epsg_from_wkt(wkt or "")

    return RasterInfo(
        path=path,
        width=width,
        height=height,
        nodata=float(nodata) if nodata is not None else None,
        geotransform=tuple(float(x) for x in gt) if gt and len(gt) == 6 else None,
        srs_wkt=wkt,
        srs_authority=authority,
    )


def resolve_authoritative_source_srs(tile_path: Path) -> Tuple[str, str]:
    """Read the river CRS from an actual E001 GeoTIFF, not the manual F010 VRT.

    F010/F020 VRTs were written manually and may omit <SRS>. Their coordinates
    are still in the original projected grid. An E001 bathymetry GeoTIFF carries
    the GeoTIFF CRS keys and is therefore used as the authoritative source SRS.
    """
    info = gdal_info(tile_path)
    if not info.srs_wkt:
        raise RuntimeError(
            "Cannot determine the source CRS from the authoritative E001 tile: "
            f"{tile_path}. gdalinfo -json returned no coordinateSystem.wkt."
        )

    # Prefer gdalsrsinfo because it resolves the top-level projected authority
    # more reliably than parsing nested WKT authorities.
    gdalsrsinfo = shutil.which("gdalsrsinfo")
    if gdalsrsinfo:
        try:
            epsg_text = run(
                [gdalsrsinfo, "-o", "epsg", str(tile_path)],
                capture=True,
            ).strip()
            match = re.search(r"EPSG\s*:\s*(\d+)", epsg_text, flags=re.IGNORECASE)
            if match:
                label = f"EPSG:{match.group(1)}"
                print(f"[CRS] authoritative E001 source CRS = {label}")
                return label, label
        except Exception as exc:
            print(f"[WARN] gdalsrsinfo EPSG lookup failed; using WKT: {exc}", file=sys.stderr)

    label = info.srs_authority or "WKT from E001 GeoTIFF"
    print(f"[CRS] authoritative E001 source CRS = {label}")
    return info.srs_wkt, label


def raster_bounds(info: RasterInfo) -> Tuple[float, float, float, float]:
    """Return left, top, right, bottom."""
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


def validate_conus_bounds(
    bounds: Sequence[Sequence[float]],
    experiment: str,
    river: str,
    source_srs_label: str,
) -> None:
    """Fail loudly if a US holdout was placed outside the conterminous US.

    This prevents a missing/incorrect source CRS from silently placing UTM
    eastings around longitude +3 to +5 degrees on the satellite basemap.
    """
    south, west = map(float, bounds[0])
    north, east = map(float, bounds[1])
    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0
    plausible = (
        -130.0 <= west <= -60.0
        and -130.0 <= east <= -60.0
        and 20.0 <= south <= 55.0
        and 20.0 <= north <= 55.0
        and south < north
        and west < east
    )
    if not plausible:
        raise RuntimeError(
            "CRS validation failed for the US holdout. "
            f"{experiment}/{river} transformed to bounds={bounds}, "
            f"center=({center_lat:.6f}, {center_lon:.6f}), "
            f"using source CRS {source_srs_label}. "
            "The dashboard is intentionally stopped so a scientifically false "
            "satellite alignment cannot be packaged."
        )
    print(
        f"[CRS-CHECK] PASS {experiment}/{river}: "
        f"center=({center_lat:.6f}, {center_lon:.6f}) "
        f"source={source_srs_label}"
    )


def build_tile_vrt(
    files: Sequence[Path],
    output: Path,
    file_list: Path,
    src_nodata: float,
    vrt_nodata: float,
    source_srs: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    file_list.write_text(
        "\n".join(str(p.resolve()) for p in files) + "\n",
        encoding="utf-8",
    )
    run([
        "gdalbuildvrt", "-q", "-overwrite", "-resolution", "highest",
        "-a_srs", source_srs,
        "-srcnodata", str(src_nodata), "-vrtnodata", str(vrt_nodata),
        "-input_file_list", str(file_list), str(output),
    ])


def make_target_grid(
    source: Path,
    output_vrt: Path,
    detail_res_m: float,
    nodata: float,
    source_srs: str,
) -> RasterInfo:
    output_vrt.parent.mkdir(parents=True, exist_ok=True)
    run([
        "gdalwarp", "-q", "-overwrite", "-of", "VRT",
        "-s_srs", source_srs,
        "-t_srs", "EPSG:3857",
        "-tr", str(detail_res_m), str(detail_res_m), "-tap",
        "-r", "near",
        "-srcnodata", str(nodata), "-dstnodata", str(nodata),
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
    source_srs: str,
    out_type: str = "Float32",
) -> None:
    left, top, right, bottom = raster_bounds(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdalwarp", "-q", "-overwrite", "-multi",
        "-wo", "NUM_THREADS=ALL_CPUS",
        "-of", "GTiff",
        "-s_srs", source_srs,
        "-t_srs", "EPSG:3857", "-te_srs", "EPSG:3857",
        "-te", f"{left:.6f}", f"{bottom:.6f}",
        f"{right:.6f}", f"{top:.6f}",
        "-tr", str(detail_res_m), str(detail_res_m), "-tap",
        "-r", resampling,
        "-srcnodata", str(src_nodata), "-dstnodata", str(dst_nodata),
        "-ot", out_type,
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=512", "-co", "BLOCKYSIZE=512",
        "-co", "COMPRESS=DEFLATE",
        "-co", "PREDICTOR=2" if out_type.lower().startswith("float") else "PREDICTOR=1",
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



def _encode_probe_tile(
    values: np.ndarray,
    valid: np.ndarray,
    output: Path,
    quantization_mm: float,
) -> bool:
    """Encode numeric values into an RGBA PNG for browser-side point queries.

    RGB stores a signed 24-bit fixed-point integer. Alpha is 255 for valid data
    and 0 for NoData. With the default 1 mm quantization, the supported range is
    approximately +/-8388.607 m. The probe is for the resampled web display grid;
    it does not alter or replace native/source raster values.
    """
    if values.shape != valid.shape:
        raise ValueError("Probe value/valid shapes differ")
    if not np.any(valid):
        return False
    step_m = float(quantization_mm) / 1000.0
    if not math.isfinite(step_m) or step_m <= 0:
        raise ValueError(f"Invalid probe quantization: {quantization_mm} mm")

    q = np.zeros(values.shape, dtype=np.int64)
    q[valid] = np.rint(values[valid].astype(np.float64) / step_m).astype(np.int64)
    qmin = int(q[valid].min())
    qmax = int(q[valid].max())
    if qmin < PROBE_RGB_MIN or qmax > PROBE_RGB_MAX:
        vmin = qmin * step_m
        vmax = qmax * step_m
        supported = PROBE_RGB_MAX * step_m
        raise RuntimeError(
            "Point-probe fixed-point range exceeded: "
            f"data=[{vmin:.6f}, {vmax:.6f}] m, supported approximately "
            f"[-{supported:.6f}, {supported:.6f}] m at {quantization_mm:g} mm. "
            "Increase --probe_quantization_mm rather than clipping values."
        )

    packed = q + PROBE_RGB_OFFSET
    rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = ((packed >> 16) & 255).astype(np.uint8)
    rgba[..., 1] = ((packed >> 8) & 255).astype(np.uint8)
    rgba[..., 2] = (packed & 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)

    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(
        output, format="PNG", compress_level=6, optimize=False
    )
    return True


def make_probe_tiles_maxzoom(
    scalar_source: Path,
    out_dir: Path,
    max_zoom: int,
    processes: int,
    resampling: str,
    nodata: float,
    valid_threshold: Optional[float],
    quantization_mm: float,
) -> int:
    """Create finest-zoom numeric probe PNGs aligned with the XYZ display grid.

    Only the finest XYZ level is required because a clicked latitude/longitude
    can always be converted to its finest-level tile and pixel. Lower-zoom
    visual overviews do not need duplicate numeric probe pyramids.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = gdal_info(scalar_source)
    bounds = raster_bounds(info)
    x_min, x_max, y_min, y_max, _ = mercator_tile_range(bounds, max_zoom)
    nx = x_max - x_min + 1
    width = nx * 256
    world_tiles = 1 << max_zoom
    tile_span = (2.0 * WEB_MERCATOR_HALF_WORLD) / world_tiles
    xmin = -WEB_MERCATOR_HALF_WORLD + x_min * tile_span
    xmax = -WEB_MERCATOR_HALF_WORLD + (x_max + 1) * tile_span

    work = out_dir.parent / f".{out_dir.name}_probe_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    total = 0

    print(
        f"[PROBE] z={max_zoom} range=x{x_min}-{x_max},y{y_min}-{y_max} "
        f"strip={width}x256 quantization={quantization_mm:g} mm",
        flush=True,
    )

    try:
        for global_y in range(y_min, y_max + 1):
            ymax = WEB_MERCATOR_HALF_WORLD - global_y * tile_span
            ymin = ymax - tile_span
            strip_tif = work / f"z{max_zoom}_row{global_y}.tif"
            strip_tif.unlink(missing_ok=True)

            run([
                "gdalwarp", "-q", "-overwrite", "-of", "GTiff",
                "-ot", "Float32",
                "-t_srs", "EPSG:3857", "-te_srs", "EPSG:3857",
                "-te", f"{xmin:.9f}", f"{ymin:.9f}",
                f"{xmax:.9f}", f"{ymax:.9f}",
                "-ts", str(width), "256",
                "-r", resampling,
                "-srcnodata", str(nodata), "-dstnodata", str(nodata),
                "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
                "-co", "TILED=YES",
                "-co", "BLOCKXSIZE=512",
                "-co", "BLOCKYSIZE=256",
                "-co", "COMPRESS=NONE",
                "-co", "BIGTIFF=IF_SAFER",
                str(scalar_source), str(strip_tif),
            ])

            arr = np.asarray(tifffile.imread(str(strip_tif)), dtype=np.float32)
            arr = np.squeeze(arr)
            if arr.ndim != 2 or arr.shape != (256, width):
                raise RuntimeError(
                    f"Unexpected probe strip shape {arr.shape}; expected (256, {width})"
                )
            valid = np.isfinite(arr) & (arr != np.float32(nodata))
            if valid_threshold is not None:
                valid &= arr > np.float32(valid_threshold)

            jobs = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, processes)
            ) as pool:
                for local_x in range(nx):
                    x0 = local_x * 256
                    x1 = x0 + 256
                    tile_valid = valid[:, x0:x1].copy()
                    if not np.any(tile_valid):
                        continue
                    tile_values = arr[:, x0:x1].copy()
                    global_x = x_min + local_x
                    tile_path = (
                        out_dir / str(max_zoom) /
                        str(global_x) / f"{global_y}.png"
                    )
                    jobs.append(
                        pool.submit(
                            _encode_probe_tile,
                            tile_values,
                            tile_valid,
                            tile_path,
                            quantization_mm,
                        )
                    )
                total += sum(1 for job in jobs if job.result())

            strip_tif.unlink(missing_ok=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"[PROBE-DONE] z={max_zoom} nonempty_tiles={total:,}", flush=True)
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

    # The manually written F010/F020 VRTs may not carry a usable <SRS>.
    # Read the authoritative CRS from a real E001 GeoTIFF and explicitly assign
    # it to every source before reprojection to EPSG:3857.
    source_srs, source_srs_label = resolve_authoritative_source_srs(
        src.input_bathy_tiles[0]
    )

    input_vrt = work / "input_bathy.vrt"
    hidden_vrt = work / "hidden_mask.vrt"
    loss_vrt = work / "loss_mask.vrt"
    build_tile_vrt(
        src.input_bathy_tiles, input_vrt, work / "input_bathy_files.txt",
        args.nodata, args.nodata, source_srs
    )
    build_tile_vrt(
        src.hidden_mask_tiles, hidden_vrt, work / "hidden_files.txt",
        args.mask_nodata, args.mask_nodata, source_srs
    )
    build_tile_vrt(
        src.loss_mask_tiles, loss_vrt, work / "loss_files.txt",
        args.mask_nodata, args.mask_nodata, source_srs
    )

    target_vrt = work / "target_grid_3857.vrt"
    target = make_target_grid(
        src.pred_vrt, target_vrt, args.detail_res_m, args.nodata, source_srs
    )
    left, top, right, bottom = raster_bounds(target)
    west, south = mercator_to_lonlat(left, bottom)
    east, north = mercator_to_lonlat(right, top)
    bounds = [[south, west], [north, east]]
    validate_conus_bounds(
        bounds, src.experiment, src.river, source_srs_label
    )

    warped = {
        "prediction": work / "prediction_3857.tif",
        "gt": work / "gt_3857.tif",
        "error": work / "error_signed_3857.tif",
        "input_bathy": work / "input_bathy_3857.tif",
        "hidden": work / "hidden_3857.tif",
        "loss": work / "loss_3857.tif",
    }
    pred_nd = gdal_info(src.pred_vrt).nodata
    gt_nd = gdal_info(src.gt_vrt).nodata
    warp_aligned(
        src.pred_vrt, warped["prediction"], target, args.detail_res_m,
        pred_nd if pred_nd is not None else args.nodata,
        args.nodata, "bilinear", source_srs
    )
    warp_aligned(
        src.gt_vrt, warped["gt"], target, args.detail_res_m,
        gt_nd if gt_nd is not None else args.nodata,
        args.nodata, "bilinear", source_srs
    )
    if not src.error_vrt:
        raise RuntimeError(
            "F020 signed-error VRT is required by the core-GDAL version; "
            "it intentionally avoids gdal_calc.py/osgeo_utils."
        )
    error_nd = gdal_info(src.error_vrt).nodata
    warp_aligned(
        src.error_vrt, warped["error"], target, args.detail_res_m,
        error_nd if error_nd is not None else args.nodata,
        args.nodata, "bilinear", source_srs
    )
    warp_aligned(
        input_vrt, warped["input_bathy"], target, args.detail_res_m,
        args.nodata, args.nodata, "bilinear", source_srs
    )
    warp_aligned(
        hidden_vrt, warped["hidden"], target, args.detail_res_m,
        args.mask_nodata, args.mask_nodata, "near", source_srs, "Byte"
    )
    warp_aligned(
        loss_vrt, warped["loss"], target, args.detail_res_m,
        args.mask_nodata, args.mask_nodata, "near", source_srs, "Byte"
    )

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
        marker = layer_tiles / ".F051_complete.json"
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

    # Numeric point-probe tiles are generated only at the finest XYZ zoom.
    # They are encoded independently from display colors, so percentile color
    # clipping and raster opacity cannot alter the reported probe values.
    probe_urls: Dict[str, str] = {}
    probe_counts: Dict[str, int] = {}
    probe_specs = {
        "prediction": (warped["prediction"], "bilinear", args.nodata, args.nodata_threshold),
        "gt": (warped["gt"], "bilinear", args.nodata, args.nodata_threshold),
        "error": (warped["error"], "bilinear", args.nodata, -1e30),
        "input_bathy": (warped["input_bathy"], "bilinear", args.nodata, args.nodata_threshold),
        "hidden": (warped["hidden"], "near", args.mask_nodata, None),
        "loss": (warped["loss"], "near", args.mask_nodata, None),
    }
    for layer_name, (probe_source, probe_resampling, probe_nodata, threshold) in probe_specs.items():
        probe_dir = tiles_root / safe_key / "probe" / layer_name
        marker = probe_dir / ".F051_probe_complete.json"
        signature = {
            "version": 1,
            "encoding": "signed_24bit_fixed_point_rgb_alpha_valid",
            "source": source_signature(probe_source),
            "detail_res_m": args.detail_res_m,
            "max_zoom": max_zoom,
            "resampling": probe_resampling,
            "nodata": probe_nodata,
            "valid_threshold": threshold,
            "quantization_mm": args.probe_quantization_mm,
            "layer": layer_name,
        }
        if layer_marker_matches(marker, signature):
            print(f"[RESUME] probe tiles already complete: {key} / {layer_name}")
            count = sum(1 for _ in probe_dir.rglob("*.png"))
        else:
            count = make_probe_tiles_maxzoom(
                probe_source, probe_dir, max_zoom, args.tile_processes,
                probe_resampling, probe_nodata, threshold,
                args.probe_quantization_mm,
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        probe_urls[layer_name] = (
            f"tiles/{safe_key}/probe/{layer_name}/{{z}}/{{x}}/{{y}}.png"
        )
        probe_counts[layer_name] = count
        print(f"[PROBE-TILES] {layer_name}: {count:,}")

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
        "probe_urls": probe_urls,
        "probe_counts": probe_counts,
        "probe_encoding": {
            "type": "signed_24bit_fixed_point_rgb_alpha_valid",
            "offset": PROBE_RGB_OFFSET,
            "quantization_mm": args.probe_quantization_mm,
            "value_step_m": args.probe_quantization_mm / 1000.0,
            "zoom": max_zoom,
            "scope": "display-grid sample, not native/source pixel",
        },
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
            "source_crs": source_srs_label,
            "source_crs_authority_tile": str(src.input_bathy_tiles[0]),
            "target_width": target.width,
            "target_height": target.height,
            "tile_root": str(args.tile_root.resolve()),
        },
        "visualization_policy": {
            "display_only": True,
            "source_rasters_modified": False,
            "source_crs": source_srs_label,
            "target_crs": "EPSG:3857",
            "xyz_tiler": "built-in core-GDAL finest-zoom plus parent-pyramid tiler (no gdal2tiles/osgeo_utils)",
            "display_resolution_m": args.detail_res_m,
            "continuous_resampling": "bilinear",
            "binary_mask_resampling": "nearest-neighbour",
            "point_probe": {
                "enabled": True,
                "layers": list(probe_specs.keys()),
                "quantization_mm": args.probe_quantization_mm,
                "sample_grid": "finest XYZ display grid",
                "native_source_pixel": False,
            },
            "basemap_layer_order": "basemap pane z=200; MAE raster pane z=400",
            "no_basemap_background": "white",
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
<title>Full-River MAE 1 m Display Dashboard with Point Probe</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" />
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
if (!window.L) {{
  document.write('<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"><\\/script>');
}}
</script>
<script>
if (!window.L) {{
  document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"><\\/script>');
}}
</script>
<style>
:root {{ --bg:#eef1f4; --panel:#fff; --text:#17202a; --muted:#5d6d7e; --border:#ccd4dc; --accent:#165d9c; --header-h:266px; }}
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
.panel {{ position:relative;background:#ffffff;border:1px solid #c7d0d8;border-radius:6px;overflow:hidden;min-width:0;min-height:0; }}
.map {{ width:100%;height:100%;background:#ffffff; }}
.leaflet-container {{ background:#ffffff !important; }}
.panelTitle {{ position:absolute;z-index:800;top:7px;left:48px;background:rgba(255,255,255,.94);padding:5px 8px;border-radius:4px;font-weight:700;font-size:12px;box-shadow:0 1px 5px rgba(0,0,0,.25);pointer-events:none; }}
.mapBadge {{ position:absolute;z-index:800;right:7px;bottom:7px;background:rgba(255,255,255,.92);padding:4px 7px;border-radius:4px;font-size:10px;font-weight:700;box-shadow:0 1px 5px rgba(0,0,0,.25);pointer-events:none; }}
.errorToolbar {{ position:absolute;z-index:850;top:7px;right:7px;display:flex;gap:6px;width:330px;max-width:calc(100% - 180px); }}
.errorToolbar select,.errorToolbar button {{ height:30px; }}
#fatalError {{ display:none;position:fixed;z-index:99999;inset:18px;overflow:auto;
background:#fff4f4;border:3px solid #b42318;border-radius:10px;padding:22px;
font-size:15px;line-height:1.55;color:#5f1712;box-shadow:0 6px 30px rgba(0,0,0,.3); }}
#fatalError h2 {{ margin-top:0; }}
#probePanel {{ display:none;position:fixed;z-index:5000;right:16px;bottom:16px;width:360px;max-width:calc(100vw - 32px);background:rgba(255,255,255,.97);border:1px solid #8ca1b3;border-left:5px solid #165d9c;border-radius:8px;box-shadow:0 4px 18px rgba(0,0,0,.35);padding:10px 12px;font-size:12px; }}
#probePanel .probeTitle {{ font-size:14px;font-weight:700;margin-bottom:4px;display:flex;justify-content:space-between;gap:10px; }}
#probePanel .probeMeta {{ color:var(--muted);font-size:10px;line-height:1.35;margin-bottom:6px; }}
#probePanel table {{ width:100%;border-collapse:collapse;font-size:11px; }}
#probePanel th,#probePanel td {{ padding:3px 5px;border-top:1px solid #e1e6ea;text-align:left; }}
#probePanel th {{ width:48%;color:#435466; }}
#probePanel .probeNote {{ margin-top:6px;color:#5b6875;font-size:9px;line-height:1.3; }}
.probe-crosshair {{ width:22px;height:22px;position:relative;filter:drop-shadow(0 0 2px #fff); }}
.probe-crosshair:before,.probe-crosshair:after {{ content:"";position:absolute;background:#ff2d20; }}
.probe-crosshair:before {{ left:10px;top:0;width:2px;height:22px; }}
.probe-crosshair:after {{ left:0;top:10px;width:22px;height:2px; }}
.probe-crosshair span {{ position:absolute;left:7px;top:7px;width:8px;height:8px;border:2px solid #fff;border-radius:50%;background:#ff2d20; }}
.leaflet-container {{ font-family:Arial,Helvetica,sans-serif; }}
.leaflet-tile-pane img {{ image-rendering:auto; }}
.pixelated .leaflet-overlay-pane img,.pixelated .leaflet-tile-pane img {{ image-rendering:pixelated;image-rendering:crisp-edges; }}
@media(max-width:950px) {{ :root{{--header-h:335px;}} header{{grid-template-columns:1fr;overflow-y:auto;}} .compareGrid{{grid-template-columns:1fr;grid-template-rows:1fr 1fr;}} .inputGrid{{grid-template-columns:1fr;grid-template-rows:repeat(3,1fr);}} }}
</style>
</head>
<body>
<div id="fatalError">
  <h2>Dashboard map library could not be loaded</h2>
  <p>The package itself may be intact, but Leaflet did not load from any of the
  three online CDNs. Fully extract the ZIP, then run <b>OPEN_DASHBOARD.bat</b>.
  Satellite and light-gray basemaps require Internet access.</p>
  <p>The local MAE PNG tiles remain unchanged. This error concerns only the
  browser map interface.</p>
</div>
<header>
<section>
  <div class="title">Full-River MAE 1 m Display Dashboard + Point Probe</div>
  <div class="controls">
    <div><label>Experiment</label><select id="experimentSelect"></select></div>
    <div><label>River</label><select id="riverSelect"></select></div>
    <div><label>Basemap</label><select id="basemapSelect"><option value="satellite">Satellite imagery</option><option value="light">Light gray map</option><option value="none">No basemap · white background</option></select></div>
    <div><label>Raster opacity <span id="opacityValue"></span></label><input id="opacitySlider" type="range" min="0" max="1" step="0.02" value="{max(0.0,min(1.0,default_opacity)):.2f}" /></div>
    <div><button id="fitBtn">Fit current river</button></div>
    <div><button id="nativeBtn">Display-detail zoom</button></div>
    <div><button id="clearProbeBtn">Clear point probe</button></div>
  </div>
  <div class="tabs"><button class="tabBtn active" data-tab="compareView">Prediction + GT</button><button class="tabBtn" data-tab="errorView">Error</button><button class="tabBtn" data-tab="inputView">Input Tile + Masks</button></div>
  <div class="help">Maps in the current tab stay synchronized. Click any map to place a linked crosshair and read numeric Prediction, GT, Error, input bathymetry, Hidden Mask, and Loss Mask values. Local MAE overlays work offline; satellite/light-gray basemaps require Internet.</div>
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
  <span class="sub">The basemap is forced into a lower Leaflet pane (z-index 200), while MAE result overlays are in a higher pane (z-index 400), so satellite/light-gray tiles cannot cover the data. Continuous layers (Prediction, GT, Error, raw bathymetry) use bilinear resampling; binary Hidden/Loss masks use nearest-neighbour resampling. Point-probe values come from the finest XYZ display grid and are independently encoded at {manifest.get("probe_quantization_mm", "—")} mm precision; they are not inferred from display colors. Display colors use robust percentile clipping ({manifest.get("color_scaling", {}).get("elevation_low_pct", "—")}–{manifest.get("color_scaling", {}).get("elevation_high_pct", "—")}% for elevation/input; {manifest.get("color_scaling", {}).get("error_abs_pct", "—")}th percentile of |error|). Values outside the display range are color-saturated only; source data remain unchanged. Satellite imagery is contextual background and is not model input or evaluation data.</span>
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
<div id="probePanel">
  <div class="probeTitle"><span>Point Probe</span><span id="probeStatus">Click a map</span></div>
  <div id="probeMeta" class="probeMeta"></div>
  <table>
    <tr><th>Prediction (m)</th><td id="probePrediction">—</td></tr>
    <tr><th>Ground Truth (m)</th><td id="probeGT">—</td></tr>
    <tr><th id="probeErrorLabel">Signed Error (m)</th><td id="probeError">—</td></tr>
    <tr><th>Absolute Error (m)</th><td id="probeAbsError">—</td></tr>
    <tr><th>Raw Input Bathymetry (m)</th><td id="probeInput">—</td></tr>
    <tr><th>Hidden Mask</th><td id="probeHidden">—</td></tr>
    <tr><th>Loss Mask Pixel</th><td id="probeLoss">—</td></tr>
  </table>
  <div class="probeNote">Probe values are sampled from the finest resampled XYZ display grid, not directly from the native source pixel. Official F020 metrics remain native/source-resolution values.</div>
</div>
<script>
if (!window.L) {{
  const fatal = document.getElementById('fatalError');
  fatal.style.display = 'block';
  document.querySelector('header').style.display = 'none';
  document.querySelector('main').style.display = 'none';
  throw new Error('Leaflet failed to load from all configured CDNs.');
}}
const records={data_json};
const manifest={manifest_json};
const byKey=Object.fromEntries(records.map(r=>[r.key,r]));
const experimentSelect=document.getElementById('experimentSelect');
const riverSelect=document.getElementById('riverSelect');
const opacitySlider=document.getElementById('opacitySlider');
const opacityValue=document.getElementById('opacityValue');
let activeTab='compareView'; let currentRecord=null; let errorMode='error'; let syncing=false;

const mapIds=['predMap','gtMap','errorMap','inputMap','hiddenMap','lossMap'];
function createDashboardMap(id){{
  const m=L.map(id,{{preferCanvas:true,zoomControl:true,attributionControl:true}});
  m.createPane('basemapPane');m.getPane('basemapPane').style.zIndex=200;m.getPane('basemapPane').style.pointerEvents='none';
  m.createPane('rasterPane');m.getPane('rasterPane').style.zIndex=400;m.getPane('rasterPane').style.pointerEvents='none';
  m.createPane('probePane');m.getPane('probePane').style.zIndex=650;m.getPane('probePane').style.pointerEvents='none';
  return m;
}}
const maps=Object.fromEntries(mapIds.map(id=>[id,createDashboardMap(id)]));
const layerState=Object.fromEntries(mapIds.map(id=>[id,{{base:null,overlay:null,probe:null}}]));

function satelliteLayer(){{return L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',{{pane:'basemapPane',maxZoom:20,attribution:'Tiles &copy; Esri'}});}}
function lightLayer(){{return L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{pane:'basemapPane',subdomains:'abcd',maxZoom:20,attribution:'&copy; OpenStreetMap contributors &copy; CARTO'}});}}
function makeBase(kind){{if(kind==='satellite')return satelliteLayer();if(kind==='light')return lightLayer();return null;}}
function applyBasemap(){{const kind=document.getElementById('basemapSelect').value;Object.entries(maps).forEach(([id,m])=>{{if(layerState[id].base)m.removeLayer(layerState[id].base);layerState[id].base=makeBase(kind);if(layerState[id].base){{layerState[id].base.addTo(m);if(layerState[id].base.bringToBack)layerState[id].base.bringToBack();}}}});}}
function installOverlay(mapId,layerName){{const m=maps[mapId];if(layerState[mapId].overlay)m.removeLayer(layerState[mapId].overlay);if(!currentRecord)return;const url=currentRecord.tile_urls[layerName];const l=L.tileLayer(url,{{pane:'rasterPane',zIndex:400,minZoom:currentRecord.min_zoom,maxNativeZoom:currentRecord.max_zoom,maxZoom:currentRecord.max_zoom+2,opacity:Number(opacitySlider.value),tms:false,noWrap:true,keepBuffer:3,updateWhenIdle:false,errorTileUrl:''}});l.addTo(m);if(l.bringToFront)l.bringToFront();layerState[mapId].overlay=l;}}
function installAllOverlays(){{installOverlay('predMap','prediction');installOverlay('gtMap','gt');installOverlay('errorMap',errorMode);installOverlay('inputMap','input_bathy');installOverlay('hiddenMap','hidden');installOverlay('lossMap','loss');}}
function setOpacity(){{const v=Number(opacitySlider.value);opacityValue.textContent=Math.round(v*100)+'%';Object.values(layerState).forEach(s=>{{if(s.overlay)s.overlay.setOpacity(v);}});}}

function syncGroup(ids){{ids.forEach(id=>{{maps[id].on('move zoom',()=>{{if(syncing)return;syncing=true;const c=maps[id].getCenter(),z=maps[id].getZoom();ids.forEach(other=>{{if(other!==id)maps[other].setView(c,z,{{animate:false,reset:false}});}});syncing=false;updateBadges();}});}});}}
syncGroup(['predMap','gtMap']);syncGroup(['inputMap','hiddenMap','lossMap']);
function activeMapIds(){{return activeTab==='compareView'?['predMap','gtMap']:(activeTab==='errorView'?['errorMap']:['inputMap','hiddenMap','lossMap']);}}
function fitActive(){{if(!currentRecord)return;activeMapIds().forEach(id=>maps[id].fitBounds(currentRecord.bounds,{{padding:[18,18],animate:false}}));setTimeout(()=>activeMapIds().forEach(id=>maps[id].invalidateSize()),20);}}
function nativeZoom(){{if(!currentRecord)return;const ids=activeMapIds();const center=maps[ids[0]].getCenter();ids.forEach(id=>maps[id].setView(center,currentRecord.max_zoom,{{animate:false}}));}}
function updateBadges(){{if(!currentRecord)return;const text=`GRID ${{currentRecord.detail_res_m}} m · XYZ finest ${{fmt(currentRecord.xyz_finest_res_m,3)}} m · resampled · z${{currentRecord.max_zoom}}`;document.getElementById('predBadge').textContent=text;document.getElementById('gtBadge').textContent=text;document.getElementById('errorBadge').textContent=text;document.getElementById('inputBadge').textContent=text;}}

const probePanel=document.getElementById('probePanel');
const probeTileCache=new Map();
let probeEpoch=0;
const probeIcon=L.divIcon({{className:'',html:'<div class="probe-crosshair"><span></span></div>',iconSize:[22,22],iconAnchor:[11,11]}});

function clearProbe(){{
  probeEpoch++;
  probePanel.style.display='none';
  Object.entries(maps).forEach(([id,m])=>{{if(layerState[id].probe){{m.removeLayer(layerState[id].probe);layerState[id].probe=null;}}}});
}}
function placeProbeMarkers(latlng){{
  Object.entries(maps).forEach(([id,m])=>{{
    if(layerState[id].probe)m.removeLayer(layerState[id].probe);
    layerState[id].probe=L.marker(latlng,{{icon:probeIcon,pane:'probePane',interactive:false,keyboard:false}}).addTo(m);
  }});
}}
function xyzPixel(latlng,z){{
  const lat=Math.max(-85.05112878,Math.min(85.05112878,latlng.lat));
  const n=Math.pow(2,z);
  const xf=(latlng.lng+180)/360*n;
  const rad=lat*Math.PI/180;
  const yf=(1-Math.asinh(Math.tan(rad))/Math.PI)/2*n;
  let x=Math.floor(xf),y=Math.floor(yf);
  let px=Math.floor((xf-x)*256),py=Math.floor((yf-y)*256);
  x=Math.max(0,Math.min(n-1,x));y=Math.max(0,Math.min(n-1,y));
  px=Math.max(0,Math.min(255,px));py=Math.max(0,Math.min(255,py));
  return {{x,y,px,py,z}};
}}
async function loadProbeTile(url){{
  if(probeTileCache.has(url))return probeTileCache.get(url);
  const promise=(async()=>{{
    const response=await fetch(url,{{cache:'force-cache'}});
    if(!response.ok)return null;
    const blob=await response.blob();
    let drawable=null,objectUrl=null;
    if(window.createImageBitmap){{drawable=await createImageBitmap(blob);}}
    else{{
      objectUrl=URL.createObjectURL(blob);
      drawable=await new Promise((resolve,reject)=>{{const img=new Image();img.onload=()=>resolve(img);img.onerror=reject;img.src=objectUrl;}});
    }}
    const canvas=document.createElement('canvas');canvas.width=256;canvas.height=256;
    const ctx=canvas.getContext('2d',{{willReadFrequently:true}});ctx.drawImage(drawable,0,0);
    const data=ctx.getImageData(0,0,256,256).data;
    if(drawable.close)drawable.close();if(objectUrl)URL.revokeObjectURL(objectUrl);
    return data;
  }})().catch(err=>{{console.warn('Probe tile load failed',url,err);return null;}});
  probeTileCache.set(url,promise);return promise;
}}
async function sampleProbe(layer,latlng){{
  if(!currentRecord||!currentRecord.probe_urls||!currentRecord.probe_urls[layer])return null;
  const c=xyzPixel(latlng,currentRecord.probe_encoding.zoom);
  const url=currentRecord.probe_urls[layer].replace('{{z}}',c.z).replace('{{x}}',c.x).replace('{{y}}',c.y);
  const data=await loadProbeTile(url);if(!data)return null;
  const i=(c.py*256+c.px)*4;const a=data[i+3];if(a===0)return null;
  const packed=data[i]*65536+data[i+1]*256+data[i+2];
  return (packed-currentRecord.probe_encoding.offset)*currentRecord.probe_encoding.value_step_m;
}}
function probeText(v,d=3){{return v===null||v===undefined||!Number.isFinite(v)?'NoData':Number(v).toFixed(d);}}
function maskText(v,kind){{
  if(v===null||v===undefined||!Number.isFinite(v))return 'NoData';
  const n=Math.round(v);
  if(kind==='hidden')return n===1?'1 · hidden':'0 · visible';
  return n===1?'1 · included in loss':'0 · excluded from loss';
}}
async function probeAt(latlng,mapId){{
  if(!currentRecord)return;
  const myEpoch=++probeEpoch;placeProbeMarkers(latlng);probePanel.style.display='block';
  document.getElementById('probeStatus').textContent='Loading…';
  document.getElementById('probeMeta').textContent=`Lat ${{latlng.lat.toFixed(7)}}, Lon ${{latlng.lng.toFixed(7)}} · clicked on ${{mapId}} · finest XYZ z${{currentRecord.probe_encoding.zoom}}`;
  ['probePrediction','probeGT','probeError','probeAbsError','probeInput','probeHidden','probeLoss'].forEach(id=>document.getElementById(id).textContent='…');
  document.getElementById('probeErrorLabel').textContent=`Signed Error (${{currentRecord.error_definition}}) (m)`;
  try{{
    const [pred,gt,err,input,hidden,loss]=await Promise.all([
      sampleProbe('prediction',latlng),sampleProbe('gt',latlng),sampleProbe('error',latlng),
      sampleProbe('input_bathy',latlng),sampleProbe('hidden',latlng),sampleProbe('loss',latlng)
    ]);
    if(myEpoch!==probeEpoch)return;
    document.getElementById('probePrediction').textContent=probeText(pred);
    document.getElementById('probeGT').textContent=probeText(gt);
    document.getElementById('probeError').textContent=probeText(err);
    document.getElementById('probeAbsError').textContent=err===null?'NoData':probeText(Math.abs(err));
    document.getElementById('probeInput').textContent=probeText(input);
    document.getElementById('probeHidden').textContent=maskText(hidden,'hidden');
    document.getElementById('probeLoss').textContent=maskText(loss,'loss');
    document.getElementById('probeStatus').textContent='Loaded';
  }}catch(err){{
    if(myEpoch!==probeEpoch)return;
    console.error(err);document.getElementById('probeStatus').textContent='Probe failed';
    document.getElementById('probeMeta').textContent+=' · Use OPEN_DASHBOARD.bat so local probe PNGs can be fetched.';
  }}
}}
mapIds.forEach(id=>maps[id].on('click',e=>probeAt(e.latlng,id)));

function option(v,t=v){{const o=document.createElement('option');o.value=v;o.textContent=t;return o;}}
[...new Set(records.map(r=>r.experiment))].sort().forEach(e=>experimentSelect.appendChild(option(e)));
function populateRivers(){{const exp=experimentSelect.value;const old=riverSelect.value;riverSelect.innerHTML='';records.filter(r=>r.experiment===exp).sort((a,b)=>a.river.localeCompare(b.river)).forEach(r=>riverSelect.appendChild(option(r.river)));if([...riverSelect.options].some(o=>o.value===old))riverSelect.value=old;loadCurrent();}}
function fmt(x,d=3){{return x===null||x===undefined||!Number.isFinite(Number(x))?'—':Number(x).toFixed(d);}}
function fmtInt(x){{return x===null||x===undefined||!Number.isFinite(Number(x))?'—':Math.round(Number(x)).toLocaleString();}}
function setGradient(id,colors){{document.getElementById(id).style.background='linear-gradient(to right,'+colors.join(',')+')';}}
function loadCurrent(){{const r=byKey[`${{experimentSelect.value}}::${{riverSelect.value}}`];if(!r)return;clearProbe();currentRecord=r;installAllOverlays();Object.values(maps).forEach(m=>{{m.setMinZoom(r.min_zoom);m.setMaxZoom(r.max_zoom+2);m.fitBounds(r.bounds,{{padding:[18,18],animate:false}});setTimeout(()=>m.invalidateSize(),25);}});document.getElementById('metricN').textContent=fmtInt(r.metrics.n_pixels);document.getElementById('metricRMSE').textContent=fmt(r.metrics.rmse_m);document.getElementById('metricMAE').textContent=fmt(r.metrics.mae_m);document.getElementById('metricBias').textContent=fmt(r.metrics.bias_m);document.getElementById('elevRange').textContent=`${{fmt(r.elev_min,2)}} to ${{fmt(r.elev_max,2)}}`;document.getElementById('inputRange').textContent=`${{fmt(r.input_elev_min,2)}} to ${{fmt(r.input_elev_max,2)}}`;document.getElementById('errorRange').textContent=errorMode==='error'?`${{fmt(-r.error_max,2)}} to ${{fmt(r.error_max,2)}}`:`0 to ${{fmt(r.error_max,2)}}`;document.getElementById('errorScaleTitle').textContent=errorMode==='error'?`Signed error: ${{r.error_definition}} (m)`:'Absolute error (m)';setGradient('elevGradient',r.elev_colors);setGradient('inputGradient',r.input_colors);setGradient('errorGradient',errorMode==='error'?r.error_colors:r.abs_error_colors);document.getElementById('hiddenTitle').textContent=`Hidden Mask · mean hidden fraction ${{fmt(r.input_stats.hidden_fraction,3)}}`;document.getElementById('lossTitle').textContent=`Loss Mask Pixel · mean included fraction ${{fmt(r.input_stats.loss_fraction,3)}}`;const total=Object.values(r.tile_counts).reduce((a,b)=>a+b,0);const probeTotal=Object.values(r.probe_counts||{{}}).reduce((a,b)=>a+b,0);document.getElementById('status').textContent=`${{r.experiment}} → ${{r.river}} · E001 tiles ${{r.input_stats.n_tiles.toLocaleString()}} · display grid ${{r.source.target_width.toLocaleString()}}×${{r.source.target_height.toLocaleString()}} at ${{r.detail_res_m}} m in EPSG:3857 (resampled) · source CRS ${{r.source.source_crs || "—"}} · display PNG tiles ${{total.toLocaleString()}} · probe PNG tiles ${{probeTotal.toLocaleString()}}`;updateBadges();}}
function updateError(){{if(!currentRecord)return;installOverlay('errorMap',errorMode);document.getElementById('errorTitle').textContent=errorMode==='error'?`Signed Error (${{currentRecord.error_definition}})`:'Absolute Error';document.getElementById('errorRange').textContent=errorMode==='error'?`${{fmt(-currentRecord.error_max,2)}} to ${{fmt(currentRecord.error_max,2)}}`:`0 to ${{fmt(currentRecord.error_max,2)}}`;document.getElementById('errorScaleTitle').textContent=errorMode==='error'?`Signed error: ${{currentRecord.error_definition}} (m)`:'Absolute error (m)';setGradient('errorGradient',errorMode==='error'?currentRecord.error_colors:currentRecord.abs_error_colors);}}

experimentSelect.addEventListener('change',populateRivers);riverSelect.addEventListener('change',loadCurrent);document.getElementById('basemapSelect').addEventListener('change',applyBasemap);opacitySlider.addEventListener('input',setOpacity);document.getElementById('fitBtn').addEventListener('click',fitActive);document.getElementById('nativeBtn').addEventListener('click',nativeZoom);document.getElementById('clearProbeBtn').addEventListener('click',clearProbe);document.getElementById('errorFitBtn').addEventListener('click',()=>maps.errorMap.fitBounds(currentRecord.bounds,{{padding:[18,18]}}));document.getElementById('errorMode').addEventListener('change',e=>{{errorMode=e.target.value;updateError();}});
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
    probe_quantization_mm: float,
) -> None:
    text = f"""F051 Full-River MAE High-Detail Satellite Dashboard

Recommended on Windows:
  1. Fully extract the ZIP.
  2. Double-click OPEN_DASHBOARD.bat.
  3. Keep the command window open while viewing.

Direct HTML fallback:
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

6. Coordinate reference system validation
   The source CRS is read from an actual E001 GeoTIFF and explicitly assigned
   to the manually written F010/F020 VRTs before reprojection. The script stops
   if a US holdout transforms outside the conterminous United States.

7. Basemap layer order
   Satellite/light-gray basemaps are forced into a lower Leaflet pane (z-index
   200). MAE result overlays are placed in a higher raster pane (z-index 400),
   so the basemap cannot cover Prediction, GT, Error, bathymetry, or masks.

8. Point probe
   Clicking any map places a synchronized crosshair on all panels. Numeric values
   are read from independently encoded finest-XYZ probe PNGs, not inferred from
   display colors. Continuous values are quantized to {probe_quantization_mm:g} mm.
   The probe reports values from the resampled finest XYZ display grid, not the
   exact native/source raster pixel. Official F020 metrics remain native-resolution.

9. Package size
   A 1 m intermediate display grid produces substantially more PNG files than
   the previous 4 m package. Generation, ZIP creation, download, and extraction
   will therefore take longer and require more storage.

For a smaller display package, rerun F050 with DETAIL_RES_M=2 or 4.

Do not open the HTML while it is still inside the ZIP. Extract the complete folder
first so the relative tiles/ paths remain available. OPEN_DASHBOARD.bat serves
the extracted folder through localhost and avoids file:// and temporary-ZIP issues.

Archive name:
  {zip_name}
"""
    (out_dir / "README_F051.txt").write_text(text, encoding="utf-8")



def write_browser_launchers(out_dir: Path, html_name: str) -> None:
    """Write a Windows-friendly local HTTP launcher.

    Serving the extracted folder over localhost avoids browsers opening only the
    HTML from inside a ZIP/temp directory and keeps all relative tiles/ paths
    available. The launcher does not upload or modify any data.
    """
    py_text = f"""#!/usr/bin/env python3
from __future__ import annotations

import http.server
import os
import socket
import socketserver
import threading
import urllib.parse
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML = {html_name!r}
os.chdir(ROOT)

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[HTTP]", fmt % args)

with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]

url = "http://127.0.0.1:" + str(port) + "/" + urllib.parse.quote(HTML)
print("Serving dashboard from:", ROOT)
print("Open:", url)
print("Keep this window open while using the dashboard. Press Ctrl+C to stop.")

threading.Timer(0.8, lambda: webbrowser.open(url)).start()
with socketserver.TCPServer(("127.0.0.1", port), QuietHandler) as server:
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nDashboard server stopped.")
"""
    (out_dir / "OPEN_DASHBOARD.py").write_text(py_text, encoding="utf-8")

    bat_text = """@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py OPEN_DASHBOARD.py
  goto :eof
)
where python >nul 2>nul
if %errorlevel%==0 (
  python OPEN_DASHBOARD.py
  goto :eof
)
echo Python was not found.
echo Fully extract this package, then open the HTML directly in Chrome or Edge.
echo If the page is blank, make sure the browser can access a Leaflet CDN.
pause
"""
    (out_dir / "OPEN_DASHBOARD.bat").write_text(bat_text, encoding="utf-8")


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
    if args.probe_quantization_mm <= 0:
        raise ValueError("probe_quantization_mm must be positive")
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
    print("F051 high-detail holdout satellite dashboard with CRS-validated browser-safe core-GDAL tiler")
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
        "probe_quantization_mm": args.probe_quantization_mm,
        "display_only": True,
        "source_rasters_modified": False,
        "target_crs": "EPSG:3857",
        "xyz_tiler": "built-in core-GDAL finest-zoom plus parent-pyramid tiler (no gdal2tiles/osgeo_utils)",
        "resampling": {
            "continuous_layers": "bilinear",
            "binary_masks": "nearest-neighbour",
        },
        "layer_order": {
            "basemap_pane_zindex": 200,
            "mae_raster_pane_zindex": 400,
            "point_probe_pane_zindex": 650,
        },
        "point_probe": {
            "enabled": True,
            "encoding": "signed 24-bit fixed-point RGB with alpha validity",
            "quantization_mm": args.probe_quantization_mm,
            "sample_grid": "finest XYZ display grid",
            "native_source_pixel": False,
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
            "Satellite/light-gray basemaps require Internet access in the browser. When No basemap is selected, the map background is white for easier identification of narrow river rasters.",
            "Leaflet is attempted from jsDelivr, unpkg, and cdnjs; a visible error panel replaces a blank page if all fail.",
            "OPEN_DASHBOARD.bat serves the extracted package over localhost.",
            "Each source CRS is read from an authoritative E001 GeoTIFF and validated after transformation to CONUS.",
            "Official metrics come from native/source-resolution F020 summary JSON and are not recomputed from display tiles.",
            "The display overlays are EPSG:3857 resampled visualization products; original F010/F020 rasters are unchanged.",
            "Continuous rasters use bilinear resampling; binary masks use nearest-neighbour resampling.",
            "Robust percentile clipping affects colors only, not raster values or official metrics.",
            "Basemaps are forced below MAE overlays using dedicated Leaflet panes.",
            "Point-probe values are independently encoded and sampled from the finest resampled XYZ display grid.",
        ],
    }
    html_path = out_dir / args.out_html
    html_path.write_text(html_template(records, manifest, args.overlay_opacity), encoding="utf-8")
    (out_dir / "F051_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(
        out_dir,
        args.out_html,
        args.zip_name,
        args.detail_res_m,
        args.elev_low_pct,
        args.elev_high_pct,
        args.error_abs_pct,
        max_zoom,
        args.probe_quantization_mm,
    )
    write_browser_launchers(out_dir, args.out_html)

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
