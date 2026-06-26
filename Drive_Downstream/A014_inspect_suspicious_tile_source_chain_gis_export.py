#!/usr/bin/env python3
"""Source-chain inspector for suspicious MAE bathymetry tiles.

Purpose
-------
For a list of suspicious tile filenames, this script reconstructs the same
patch-level masks used by the current NoData-safe core-loss MAE model and
compares them against the source data chain:

  * current model target tile / merged bathy+3DEP tile;
  * canonical bathy source crop;
  * resampled 3DEP source crop;
  * current final/LCC mask tile used by the model;
  * optional source-grid final mask crop;
  * model prediction patch mask;
  * model core loss patch mask;
  * E030-style core exact pixel mask;
  * WaterProb_1m.vrt;
  * USRiver_1m.vrt or USRiver_1m.tif;
  * source-origin map and GT jump/gradient diagnostics.

The model-mask reproduction follows the latest util/dem_dataset.py and
models_mae.py exact mode:

  candidate_patch = max(final_mask_pixel_in_patch) > lcc_patch_threshold
  valid_patch     = all(valid_pixels_in_patch)
  prediction_patch = candidate_patch AND valid_patch
  loss_patch       = prediction_patch AND centered_core_patch
  core_exact_pixel = loss_patch_pixels AND final_mask_pixel

NoData patches are not merely removed from the loss; they are excluded from
visible/prediction/decoder-valid patch sets in the model logic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


TILE_RE = re.compile(
    r"^Select_tile_Basin_(?P<res>\d+)m_(?P<river>.+)_ID(?P<tile_id>\d+)\.tif$",
    re.IGNORECASE,
)


def run(cmd: Sequence[str], *, capture: bool = True, check: bool = True, input_text: Optional[str] = None) -> str:
    res = subprocess.run(
        list(cmd),
        text=True,
        input=input_text,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=check,
    )
    return res.stdout if capture else ""


def gdalinfo_json(path: str | Path) -> dict:
    return json.loads(run(["gdalinfo", "-json", str(path)]))


def get_projwin(ref_path: str | Path) -> Tuple[float, float, float, float]:
    info = gdalinfo_json(ref_path)
    ulx, uly = info["cornerCoordinates"]["upperLeft"]
    lrx, lry = info["cornerCoordinates"]["lowerRight"]
    return float(ulx), float(uly), float(lrx), float(lry)


def get_band_nodata(path: str | Path) -> Optional[float]:
    try:
        info = gdalinfo_json(path)
        nd = info.get("bands", [{}])[0].get("noDataValue", None)
        return float(nd) if nd is not None else None
    except Exception:
        return None


def get_ref_geo(path: str | Path) -> Dict[str, Any]:
    info = gdalinfo_json(path)
    ulx, uly = info["cornerCoordinates"]["upperLeft"]
    lrx, lry = info["cornerCoordinates"]["lowerRight"]
    size = info.get("size", [336, 336])
    wkt = ""
    cs = info.get("coordinateSystem", {})
    if isinstance(cs, dict):
        wkt = cs.get("wkt", "") or cs.get("wkt2", "")
    gt = info.get("geoTransform", None)
    return {
        "ulx": float(ulx),
        "uly": float(uly),
        "lrx": float(lrx),
        "lry": float(lry),
        "width": int(size[0]),
        "height": int(size[1]),
        "wkt": wkt,
        "geoTransform": gt,
    }


def tile_center_xy(ref_geo: Dict[str, Any]) -> Tuple[float, float]:
    return (
        0.5 * (float(ref_geo["ulx"]) + float(ref_geo["lrx"])),
        0.5 * (float(ref_geo["uly"]) + float(ref_geo["lry"])),
    )


def center_lonlat(ref_geo: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    wkt = ref_geo.get("wkt", "")
    if not wkt:
        return None, None
    cx, cy = tile_center_xy(ref_geo)
    with tempfile.TemporaryDirectory(prefix="a013_srs_") as td:
        wkt_file = Path(td) / "source_srs.wkt"
        wkt_file.write_text(wkt)
        try:
            out = run(
                ["gdaltransform", "-s_srs", str(wkt_file), "-t_srs", "EPSG:4326"],
                input_text=f"{cx} {cy}\n",
            ).strip()
            if not out:
                return None, None
            parts = out.split()
            lon, lat = float(parts[0]), float(parts[1])
            return lon, lat
        except Exception:
            return None, None


def save_plain_tiff_with_world(
    arr: np.ndarray,
    out_tif: str | Path,
    ref_geo: Dict[str, Any],
    dtype: str = "float32",
    nodata: Optional[float] = None,
) -> Path:
    """Save an array as TIFF plus .tfw/.prj sidecars for ArcMap/ArcGIS.

    This avoids depending on Python GDAL bindings. The source and cropped rasters
    remain real GeoTIFFs; derived rasters are plain TIFFs with georeference sidecars.
    ArcGIS reads .tfw + .prj with the same basename.
    """
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr)
    if dtype == "uint8":
        a = np.where(np.isfinite(a), a, 0).astype(np.uint8)
    elif dtype == "int16":
        a = np.where(np.isfinite(a), a, 0).astype(np.int16)
    elif dtype == "float32":
        a = a.astype(np.float32)
        if nodata is not None:
            a = np.where(np.isfinite(a), a, np.float32(nodata))
    else:
        a = a.astype(dtype)
    Image.fromarray(a).save(out_tif)

    h, w = a.shape[:2]
    ulx, uly, lrx, lry = ref_geo["ulx"], ref_geo["uly"], ref_geo["lrx"], ref_geo["lry"]
    px = (lrx - ulx) / float(w)
    py = (lry - uly) / float(h)
    # World file uses center of upper-left pixel.
    tfw = out_tif.with_suffix(".tfw")
    tfw.write_text(
        f"{px:.12f}\n"
        f"0.000000000000\n"
        f"0.000000000000\n"
        f"{py:.12f}\n"
        f"{ulx + px/2.0:.12f}\n"
        f"{uly + py/2.0:.12f}\n"
    )
    wkt = ref_geo.get("wkt", "")
    if wkt:
        out_tif.with_suffix(".prj").write_text(wkt)
    return out_tif


def copy_if_exists(src: Optional[Path], dst: Path) -> Optional[Path]:
    if src is None or not Path(src).exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    # Copy sidecar aux/xml if present.
    for ext in [".aux.xml", ".ovr", ".tfw", ".prj"]:
        side = Path(str(src) + ext) if ext.startswith(".aux") else Path(src).with_suffix(ext)
        if side.exists():
            try:
                shutil.copy2(side, dst.with_suffix(ext))
            except Exception:
                pass
    return dst


def crop_like_tile(ref_tile: str | Path, src: str | Path, out: str | Path) -> Path:
    ulx, uly, lrx, lry = get_projwin(ref_tile)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdal_translate", "-q",
        "-projwin", str(ulx), str(uly), str(lrx), str(lry),
        "-outsize", "336", "336",
        str(src), str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def read_tif(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected single-band raster, got {arr.shape}: {path}")
    return arr.astype(np.float64, copy=False)


def valid_mask(arr: np.ndarray, nodata: Optional[float], nodata_threshold: float) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(a)
    if nodata is not None and math.isfinite(float(nodata)):
        nd = float(nodata)
        atol = max(1.0e-6, abs(nd) * 1.0e-7)
        valid &= ~np.isclose(a, nd, rtol=0.0, atol=atol)
    valid &= a > float(nodata_threshold)
    return valid


def binary_mask(arr: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    a = np.asarray(arr)
    a = np.where(np.isfinite(a), a, 0)
    return a > threshold


def parse_tile_name(tile_name: str) -> Tuple[int, str, int]:
    m = TILE_RE.match(Path(tile_name).name)
    if not m:
        raise ValueError(f"Unrecognized tile name: {tile_name}")
    return int(m.group("res")), m.group("river"), int(m.group("tile_id"))


def find_first_existing(candidates: Iterable[str | Path]) -> Optional[Path]:
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None


def source_path(root: str | Path | None, river: str, names: Sequence[str]) -> Optional[Path]:
    if not root:
        return None
    base = Path(root) / river
    return find_first_existing(base / name for name in names)


def crop_optional(
    ref_tile: Path,
    src: Optional[Path],
    tmp: Path,
    tag: str,
) -> Tuple[Optional[np.ndarray], Optional[Path], Optional[float]]:
    if src is None or not src.exists():
        return None, None, None
    out = tmp / f"{tag}.tif"
    crop_like_tile(ref_tile, src, out)
    return read_tif(out), out, get_band_nodata(out)


def patch_grid(mask: np.ndarray, valid: np.ndarray, patch_size: int, lcc_patch_threshold: float) -> Dict[str, np.ndarray]:
    if mask.shape != valid.shape:
        raise ValueError(f"mask/valid shape mismatch: {mask.shape} vs {valid.shape}")
    h, w = mask.shape
    p = int(patch_size)
    hh = (h // p) * p
    ww = (w // p) * p
    mask_blocks = mask[:hh, :ww].reshape(hh // p, p, ww // p, p)
    valid_blocks = valid[:hh, :ww].reshape(hh // p, p, ww // p, p)
    candidate = mask_blocks.max(axis=(1, 3)) > float(lcc_patch_threshold)
    valid_patch = valid_blocks.min(axis=(1, 3)) > 0
    prediction = candidate & valid_patch
    visible = (~candidate) & valid_patch
    ignored = ~valid_patch
    return {
        "candidate": candidate,
        "valid": valid_patch,
        "prediction": prediction,
        "visible": visible,
        "ignored": ignored,
    }


def center_core_patch_mask(grid_shape: Tuple[int, int], radius: int) -> np.ndarray:
    gh, gw = grid_shape
    out = np.zeros((gh, gw), dtype=bool)
    cy, cx = gh // 2, gw // 2
    y0, y1 = max(0, cy - radius), min(gh, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(gw, cx + radius + 1)
    out[y0:y1, x0:x1] = True
    return out


def expand_patch_mask(pm: np.ndarray, patch_size: int, shape: Tuple[int, int]) -> np.ndarray:
    px = np.kron(pm.astype(np.uint8), np.ones((patch_size, patch_size), dtype=np.uint8)).astype(bool)
    return px[: shape[0], : shape[1]]


def model_masks_exact(
    final_mask_pixel: np.ndarray,
    valid_pixel: np.ndarray,
    patch_size: int = 16,
    lcc_patch_threshold: float = 0.5,
    core_patch_radius: int = 3,
) -> Dict[str, Any]:
    status = patch_grid(final_mask_pixel.astype(np.uint8), valid_pixel.astype(np.uint8), patch_size, lcc_patch_threshold)
    core_patch = center_core_patch_mask(status["valid"].shape, core_patch_radius)
    prediction_patch = status["prediction"]
    loss_patch = prediction_patch & core_patch
    candidate_px = expand_patch_mask(status["candidate"], patch_size, final_mask_pixel.shape)
    valid_patch_px = expand_patch_mask(status["valid"], patch_size, final_mask_pixel.shape)
    prediction_px = expand_patch_mask(prediction_patch, patch_size, final_mask_pixel.shape)
    loss_patch_px = expand_patch_mask(loss_patch, patch_size, final_mask_pixel.shape)
    core_box_px = expand_patch_mask(core_patch, patch_size, final_mask_pixel.shape)
    core_exact_px = loss_patch_px & final_mask_pixel.astype(bool)
    visible_px = expand_patch_mask(status["visible"], patch_size, final_mask_pixel.shape)
    ignored_px = expand_patch_mask(status["ignored"], patch_size, final_mask_pixel.shape)
    return {
        "status": status,
        "core_patch": core_patch,
        "prediction_patch": prediction_patch,
        "loss_patch": loss_patch,
        "candidate_px": candidate_px,
        "valid_patch_px": valid_patch_px,
        "prediction_px": prediction_px,
        "loss_patch_px": loss_patch_px,
        "core_box_px": core_box_px,
        "core_exact_px": core_exact_px,
        "visible_px": visible_px,
        "ignored_px": ignored_px,
    }


def robust_limits(arr: Optional[np.ndarray], mask: Optional[np.ndarray] = None, pmin: float = 2, pmax: float = 98) -> Tuple[float, float]:
    if arr is None:
        return 0.0, 1.0
    a = np.asarray(arr, dtype=np.float64)
    vals = a[mask] if mask is not None else a[np.isfinite(a)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, pmin))
    hi = float(np.percentile(vals, pmax))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def gradient_mag(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64).copy()
    if valid.any():
        fill = float(np.nanmedian(a[valid]))
    else:
        fill = 0.0
    a[~valid] = fill
    gy, gx = np.gradient(a)
    g = np.sqrt(gx * gx + gy * gy)
    g[~valid] = np.nan
    return g


def near_boundary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Boundary-like pixels using simple 4-neighbor differences, dilated by radius."""
    m = mask.astype(bool)
    b = np.zeros_like(m, dtype=bool)
    b[:-1, :] |= m[:-1, :] != m[1:, :]
    b[1:, :] |= m[1:, :] != m[:-1, :]
    b[:, :-1] |= m[:, :-1] != m[:, 1:]
    b[:, 1:] |= m[:, 1:] != m[:, :-1]
    if radius <= 0:
        return b
    out = b.copy()
    for _ in range(radius):
        out2 = out.copy()
        out2[:-1, :] |= out[1:, :]
        out2[1:, :] |= out[:-1, :]
        out2[:, :-1] |= out[:, 1:]
        out2[:, 1:] |= out[:, :-1]
        out = out2
    return out


def tile_norm_stats(arr: np.ndarray, final_mask: np.ndarray, valid: np.ndarray, std_scale: float, eps: float) -> Dict[str, float]:
    known = (~final_mask.astype(bool)) & valid.astype(bool)
    vals = arr[known]
    used_known = True
    if vals.size < 2:
        vals = arr[valid.astype(bool)]
        used_known = False
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return {
            "tile_norm_used_known_visible": int(used_known),
            "tile_mean_m": float("nan"),
            "tile_std_m": float("nan"),
            "tile_std_safe": float("nan"),
            "known_visible_pixel_count": int(known.sum()),
        }
    std = float(np.std(vals))
    return {
        "tile_norm_used_known_visible": int(used_known),
        "tile_mean_m": float(np.mean(vals)),
        "tile_std_m": std,
        "tile_std_safe": max(std * std_scale, eps),
        "known_visible_pixel_count": int(known.sum()),
    }


def imshow(ax, arr, title, *, mask_invalid=None, cmap="viridis", vmin=None, vmax=None, colorbar=True):
    if arr is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=11)
        ax.set_title(title)
        ax.axis("off")
        return None
    data = np.asarray(arr, dtype=np.float64)
    if mask_invalid is not None:
        data = np.ma.array(data, mask=mask_invalid)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    return im


def overlay_box(ax, mask: np.ndarray, color="red", lw=1.5):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1,
                           fill=False, edgecolor=color, linewidth=lw))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--tile_names", nargs="*", default=[])
    ap.add_argument("--tile_list_txt", default="")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--bathy_root", required=True)
    ap.add_argument("--dep_root", required=True)
    ap.add_argument("--merged_root", required=True)
    ap.add_argument("--final_mask_root", default="")
    ap.add_argument("--aux_root", default="/tank/data/SFS/xinyis/data/bathymetry/Data_for_BetterMask/Auxiliary_ByRiver_1m")

    ap.add_argument("--bathy_names", nargs="*", default=["Bathy_1m.vrt", "Bathy_1m.tif"])
    ap.add_argument("--dep_names", nargs="*", default=["DEM_3DEP_1m_ResampleandClip.vrt", "DEM_3DEP_1m_ResampleandClip.tif"])
    ap.add_argument("--merged_names", nargs="*", default=["Combined_Bathy_Priority_1m.vrt", "Combined_Bathy_Priority_1m.tif"])
    ap.add_argument("--final_mask_names", nargs="*", default=["MAE_PredictionMask_1m.vrt", "MAE_PredictionMask_1m.tif"])
    ap.add_argument("--water_prob_names", nargs="*", default=["WaterProb_1m.vrt", "WaterProb_1m.tif"])
    ap.add_argument("--us_river_names", nargs="*", default=["USRiver_1m.vrt", "USRiver_1m.tif"])

    ap.add_argument("--nodata", type=float, default=-999999.0)
    ap.add_argument("--nodata_threshold", type=float, default=-9999.0)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--lcc_patch_threshold", type=float, default=0.5)
    ap.add_argument("--core_patch_radius", type=int, default=3)
    ap.add_argument("--tile_norm_std_scale", type=float, default=1.5)
    ap.add_argument("--tile_norm_eps", type=float, default=1e-3)
    ap.add_argument("--water_prob_threshold", type=float, default=0.0)
    ap.add_argument("--jump_percentile", type=float, default=95.0)
    ap.add_argument("--save_geotiffs", action="store_true", default=True,
                    help="Save all source crops and derived masks into a rasters/ folder for GIS inspection.")
    ap.add_argument("--no_save_geotiffs", dest="save_geotiffs", action="store_false",
                    help="Disable rasters/ GeoTIFF-sidecar outputs.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_names = [Path(x).name for x in args.tile_names]
    if args.tile_list_txt:
        with open(args.tile_list_txt, "r") as f:
            tile_names.extend(Path(line.strip()).name for line in f if line.strip() and not line.strip().startswith("#"))
    tile_names = list(dict.fromkeys(tile_names))
    if not tile_names:
        raise ValueError("No tiles provided. Use --tile_names or --tile_list_txt")

    rows: List[Dict[str, Any]] = []

    for tile_name in tile_names:
        res_m, river, tile_id = parse_tile_name(tile_name)
        tile_path = Path(args.tile_dir) / tile_name
        mask_name = f"Select_tile_{res_m}m_{river}_ID{tile_id}_LCC_Mask.tif"
        mask_path = Path(args.mask_dir) / mask_name
        if not tile_path.is_file():
            print(f"[WARN] missing tile: {tile_path}")
            continue
        if not mask_path.is_file():
            print(f"[WARN] missing mask: {mask_path}")
            continue

        bathy_src = source_path(args.bathy_root, river, args.bathy_names)
        dep_src = source_path(args.dep_root, river, args.dep_names)
        merged_src = source_path(args.merged_root, river, args.merged_names)
        final_src = source_path(args.final_mask_root, river, args.final_mask_names) if args.final_mask_root else None
        water_src = source_path(args.aux_root, river, args.water_prob_names)
        us_src = source_path(args.aux_root, river, args.us_river_names)

        tile_out_dir = out_dir / river / f"ID{tile_id}"
        tile_out_dir.mkdir(parents=True, exist_ok=True)

        ref_geo = get_ref_geo(tile_path)
        center_x, center_y = tile_center_xy(ref_geo)
        center_lon, center_lat = center_lonlat(ref_geo)

        with tempfile.TemporaryDirectory(prefix="a013_tile_") as td:
            tmp = Path(td)

            gt = read_tif(tile_path)
            gt_nd = get_band_nodata(tile_path)
            gt_valid = valid_mask(gt, gt_nd if gt_nd is not None else args.nodata, args.nodata_threshold)
            final_tile = binary_mask(read_tif(mask_path), 0.0)

            bathy, bathy_crop, bathy_nd = crop_optional(tile_path, bathy_src, tmp, "bathy")
            dep, dep_crop, dep_nd = crop_optional(tile_path, dep_src, tmp, "dep")
            merged, merged_crop, merged_nd = crop_optional(tile_path, merged_src, tmp, "merged")
            final_grid, final_grid_crop, _ = crop_optional(tile_path, final_src, tmp, "final_grid")
            water_prob, water_crop, _ = crop_optional(tile_path, water_src, tmp, "water_prob")
            us_river, us_crop, _ = crop_optional(tile_path, us_src, tmp, "us_river")

            if merged is None:
                merged = gt
                merged_nd = gt_nd
            merged_valid = valid_mask(merged, merged_nd if merged_nd is not None else args.nodata, args.nodata_threshold)

            bathy_valid = valid_mask(bathy, bathy_nd if bathy_nd is not None else args.nodata, args.nodata_threshold) if bathy is not None else np.zeros_like(gt_valid)
            dep_valid = valid_mask(dep, dep_nd if dep_nd is not None else args.nodata, args.nodata_threshold) if dep is not None else np.zeros_like(gt_valid)

            mm = model_masks_exact(
                final_tile,
                gt_valid,
                patch_size=args.patch_size,
                lcc_patch_threshold=args.lcc_patch_threshold,
                core_patch_radius=args.core_patch_radius,
            )

            source_origin = np.zeros_like(gt, dtype=np.int16)
            source_origin[bathy_valid] = 1
            source_origin[(~bathy_valid) & dep_valid] = 2
            source_origin[(~bathy_valid) & (~dep_valid)] = 0

            fill3dep = (~bathy_valid) & dep_valid
            source_boundary = near_boundary(source_origin == 1, radius=1) | near_boundary(source_origin == 2, radius=1)
            grad = gradient_mag(gt, gt_valid)
            if np.isfinite(grad).any():
                jump_thr = float(np.nanpercentile(grad, args.jump_percentile))
            else:
                jump_thr = float("nan")
            jump = np.isfinite(grad) & (grad >= jump_thr)

            water_mask = binary_mask(water_prob, args.water_prob_threshold) if water_prob is not None else np.zeros_like(final_tile)
            us_mask = binary_mask(us_river, 0.0) if us_river is not None else np.zeros_like(final_tile)

            core_exact = mm["core_exact_px"]
            loss_patch_px = mm["loss_patch_px"]
            pred_px = mm["prediction_px"]

            norm_stats = tile_norm_stats(
                gt,
                final_tile,
                gt_valid,
                std_scale=args.tile_norm_std_scale,
                eps=args.tile_norm_eps,
            )

            row: Dict[str, Any] = {
                "tile_name": tile_name,
                "river": river,
                "tile_id": tile_id,
                "tile_center_x": center_x,
                "tile_center_y": center_y,
                "tile_center_lon": center_lon if center_lon is not None else "",
                "tile_center_lat": center_lat if center_lat is not None else "",
                "tile_path": str(tile_path),
                "mask_path": str(mask_path),
                "bathy_src": str(bathy_src) if bathy_src else "",
                "dep_src": str(dep_src) if dep_src else "",
                "merged_src": str(merged_src) if merged_src else "",
                "final_mask_src": str(final_src) if final_src else "",
                "water_prob_src": str(water_src) if water_src else "",
                "us_river_src": str(us_src) if us_src else "",
                "gt_valid_pixels": int(gt_valid.sum()),
                "final_mask_pixels": int(final_tile.sum()),
                "candidate_patch_count": int(mm["status"]["candidate"].sum()),
                "valid_patch_count": int(mm["status"]["valid"].sum()),
                "prediction_patch_count": int(mm["prediction_patch"].sum()),
                "loss_patch_count": int(mm["loss_patch"].sum()),
                "core_exact_pixel_count": int(core_exact.sum()),
                "visible_patch_count": int(mm["status"]["visible"].sum()),
                "ignored_patch_count": int(mm["status"]["ignored"].sum()),
                "bathy_valid_pixels": int(bathy_valid.sum()),
                "dep_valid_pixels": int(dep_valid.sum()),
                "filled3dep_pixels": int(fill3dep.sum()),
                "filled3dep_in_final_pixels": int((fill3dep & final_tile).sum()),
                "filled3dep_in_loss_patch_pixels": int((fill3dep & loss_patch_px).sum()),
                "filled3dep_in_core_exact_pixels": int((fill3dep & core_exact).sum()),
                "frac_filled3dep_in_final": float((fill3dep & final_tile).sum() / max(1, final_tile.sum())),
                "frac_filled3dep_in_loss_patch": float((fill3dep & loss_patch_px).sum() / max(1, loss_patch_px.sum())),
                "frac_filled3dep_in_core_exact": float((fill3dep & core_exact).sum() / max(1, core_exact.sum())),
                "invalid_in_final_pixels": int(((~gt_valid) & final_tile).sum()),
                "invalid_in_loss_patch_pixels": int(((~gt_valid) & loss_patch_px).sum()),
                "jump_pixels_in_final": int((jump & final_tile).sum()),
                "jump_pixels_in_core_exact": int((jump & core_exact).sum()),
                "source_boundary_in_core_exact_pixels": int((source_boundary & core_exact).sum()),
                "waterprob_pixels": int(water_mask.sum()) if water_prob is not None else "",
                "usriver_pixels": int(us_mask.sum()) if us_river is not None else "",
                "final_not_waterprob_pixels": int((final_tile & ~water_mask).sum()) if water_prob is not None else "",
                "waterprob_not_final_pixels": int((water_mask & ~final_tile).sum()) if water_prob is not None else "",
                "final_not_usriver_pixels": int((final_tile & ~us_mask).sum()) if us_river is not None else "",
                "usriver_not_final_pixels": int((us_mask & ~final_tile).sum()) if us_river is not None else "",
                "waterprob_min": float(np.nanmin(water_prob)) if water_prob is not None and np.isfinite(water_prob).any() else "",
                "waterprob_max": float(np.nanmax(water_prob)) if water_prob is not None and np.isfinite(water_prob).any() else "",
                "usriver_min": float(np.nanmin(us_river)) if us_river is not None and np.isfinite(us_river).any() else "",
                "usriver_max": float(np.nanmax(us_river)) if us_river is not None and np.isfinite(us_river).any() else "",
                "jump_threshold": jump_thr,
            }
            row.update(norm_stats)
            rows.append(row)

            # Save GIS-readable rasters.
            if args.save_geotiffs:
                ras_dir = tile_out_dir / "rasters"
                ras_dir.mkdir(parents=True, exist_ok=True)

                copy_if_exists(tile_path, ras_dir / "01_current_model_target_tile.tif")
                copy_if_exists(mask_path, ras_dir / "05_current_model_final_mask_tile.tif")
                copy_if_exists(bathy_crop, ras_dir / "02_canonical_bathy_crop.tif")
                copy_if_exists(dep_crop, ras_dir / "03_resampled_3dep_crop.tif")
                copy_if_exists(merged_crop, ras_dir / "04_source_merged_crop.tif")
                copy_if_exists(final_grid_crop, ras_dir / "06_source_grid_final_mask_crop.tif")
                copy_if_exists(water_crop, ras_dir / "12_waterprob_crop.tif")
                copy_if_exists(us_crop, ras_dir / "13_usriver_crop.tif")

                save_plain_tiff_with_world(source_origin, ras_dir / "07_source_origin_0invalid_1bathy_2depfill.tif", ref_geo, dtype="int16")
                save_plain_tiff_with_world(mm["prediction_px"].astype(np.uint8), ras_dir / "08_model_prediction_patch_mask.tif", ref_geo, dtype="uint8")
                save_plain_tiff_with_world(mm["loss_patch_px"].astype(np.uint8), ras_dir / "09_model_core_loss_patch_mask.tif", ref_geo, dtype="uint8")
                save_plain_tiff_with_world(mm["core_exact_px"].astype(np.uint8), ras_dir / "10_e030_core_exact_pixel_mask.tif", ref_geo, dtype="uint8")
                save_plain_tiff_with_world(mm["visible_px"].astype(np.uint8), ras_dir / "11_model_visible_valid_patch_mask.tif", ref_geo, dtype="uint8")
                save_plain_tiff_with_world(mm["ignored_px"].astype(np.uint8), ras_dir / "14_model_ignored_nodata_patch_mask.tif", ref_geo, dtype="uint8")
                save_plain_tiff_with_world(fill3dep.astype(np.uint8), ras_dir / "15_filled_from_3dep_pixels.tif", ref_geo, dtype="uint8")
                save_plain_tiff_with_world(grad, ras_dir / "16_gt_gradient_magnitude.tif", ref_geo, dtype="float32", nodata=-999999.0)
                save_plain_tiff_with_world(jump.astype(np.uint8), ras_dir / "17_gt_jump_top_percentile_pixels.tif", ref_geo, dtype="uint8")
                save_plain_tiff_with_world(source_boundary.astype(np.uint8), ras_dir / "18_source_origin_boundary_pixels.tif", ref_geo, dtype="uint8")
                save_plain_tiff_with_world(water_mask.astype(np.uint8), ras_dir / "19_waterprob_binary_mask.tif", ref_geo, dtype="uint8")
                save_plain_tiff_with_world(us_mask.astype(np.uint8), ras_dir / "20_usriver_binary_mask.tif", ref_geo, dtype="uint8")

                (ras_dir / "README_layers.txt").write_text(
                    "Layer meanings\\n"
                    "01_current_model_target_tile.tif: tile actually used as model GT/target.\\n"
                    "02_canonical_bathy_crop.tif: original/canonical bathy source crop.\\n"
                    "03_resampled_3dep_crop.tif: resampled 3DEP source crop.\\n"
                    "04_source_merged_crop.tif: merged bathy+3DEP crop if found.\\n"
                    "05_current_model_final_mask_tile.tif: final mask tile used by model dataset.\\n"
                    "06_source_grid_final_mask_crop.tif: final mask cropped from source grid if found.\\n"
                    "07_source_origin_0invalid_1bathy_2depfill.tif: 0 invalid, 1 bathy-valid, 2 3DEP-filled where bathy invalid.\\n"
                    "08_model_prediction_patch_mask.tif: patch-expanded prediction area.\\n"
                    "09_model_core_loss_patch_mask.tif: patch-expanded core loss area.\\n"
                    "10_e030_core_exact_pixel_mask.tif: core loss patch area intersected with final mask pixels.\\n"
                    "11_model_visible_valid_patch_mask.tif: visible valid patch pixels used for context/tile-normalization logic.\\n"
                    "12_waterprob_crop.tif: WaterProb_1m.vrt/tif crop, VRT prioritized.\\n"
                    "13_usriver_crop.tif: USRiver_1m.vrt/tif crop, VRT prioritized.\\n"
                    "14_model_ignored_nodata_patch_mask.tif: patch-expanded NoData/invalid patch area.\\n"
                    "15_filled_from_3dep_pixels.tif: pixels where bathy is invalid but 3DEP is valid.\\n"
                    "16_gt_gradient_magnitude.tif: gradient magnitude of current GT tile.\\n"
                    "17_gt_jump_top_percentile_pixels.tif: high-gradient pixels above selected percentile.\\n"
                    "18_source_origin_boundary_pixels.tif: boundary around bathy/3DEP source-origin classes.\\n"
                    "19_waterprob_binary_mask.tif: WaterProb > threshold.\\n"
                    "20_usriver_binary_mask.tif: USRiver > 0.\\n"
                    f"center_x={center_x}\\ncenter_y={center_y}\\ncenter_lon={center_lon}\\ncenter_lat={center_lat}\\n"
                )

            # Save per-tile metrics.
            with (tile_out_dir / "metrics.json").open("w") as f:
                json.dump(row, f, indent=2)

            # Plot source-chain figure.
            vmin, vmax = robust_limits(gt, gt_valid)
            bvmin, bvmax = robust_limits(bathy, bathy_valid) if bathy is not None else (vmin, vmax)
            dvmin, dvmax = robust_limits(dep, dep_valid) if dep is not None else (vmin, vmax)
            gvmin, gvmax = robust_limits(grad, np.isfinite(grad))

            fig, axes = plt.subplots(5, 4, figsize=(24, 27))
            fig.suptitle(f"{tile_name} | source-chain + exact model masks", fontsize=15)

            imshow(axes[0, 0], gt, "Current target tile / GT", mask_invalid=~gt_valid, cmap="terrain", vmin=vmin, vmax=vmax)
            imshow(axes[0, 1], bathy, "Canonical bathy crop", mask_invalid=~bathy_valid if bathy is not None else None, cmap="terrain", vmin=bvmin, vmax=bvmax)
            imshow(axes[0, 2], dep, "Resampled 3DEP crop", mask_invalid=~dep_valid if dep is not None else None, cmap="terrain", vmin=dvmin, vmax=dvmax)

            cmap_src = ListedColormap(["black", "tab:blue", "tab:orange"])
            axes[0, 3].imshow(source_origin, cmap=cmap_src, vmin=0, vmax=2)
            axes[0, 3].set_title("Source origin\n0 invalid / 1 bathy / 2 3DEP fill", fontsize=9)
            axes[0, 3].axis("off")

            imshow(axes[1, 0], final_tile.astype(float), "Model final/LCC mask tile", cmap="viridis")
            imshow(axes[1, 1], final_grid if final_grid is not None else None, "Source-grid final mask crop", cmap="viridis")
            imshow(axes[1, 2], mm["prediction_px"].astype(float), "Model prediction patch mask", cmap="viridis")
            imshow(axes[1, 3], mm["loss_patch_px"].astype(float), "Model core loss patch mask", cmap="viridis")
            overlay_box(axes[1, 3], mm["core_box_px"], color="red")

            imshow(axes[2, 0], mm["core_exact_px"].astype(float), "E030 core exact pixel mask", cmap="viridis")
            imshow(axes[2, 1], mm["visible_px"].astype(float), "Visible valid patch pixels\nused for tile norm/encoder", cmap="viridis")
            imshow(axes[2, 2], mm["ignored_px"].astype(float), "Ignored NoData patch pixels", cmap="magma")
            imshow(axes[2, 3], fill3dep.astype(float), "3DEP-filled pixels\n(~bathy & 3DEP)", cmap="magma")

            imshow(axes[3, 0], water_prob, "WaterProb_1m", cmap="Blues")
            imshow(axes[3, 1], water_mask.astype(float) if water_prob is not None else None, f"WaterProb > {args.water_prob_threshold:g}", cmap="Blues")
            imshow(axes[3, 2], us_river, "USRiver_1m", cmap="Blues")
            imshow(axes[3, 3], us_mask.astype(float) if us_river is not None else None, "USRiver binary", cmap="Blues")

            imshow(axes[4, 0], grad, "GT gradient magnitude", mask_invalid=~np.isfinite(grad), cmap="inferno", vmin=gvmin, vmax=gvmax)
            imshow(axes[4, 1], jump.astype(float), f"Top {100-args.jump_percentile:g}% GT jump pixels", cmap="magma")
            imshow(axes[4, 2], (jump & core_exact).astype(float), "GT jumps inside core exact", cmap="magma")

            ax = axes[4, 3]
            ax.axis("off")
            key_lines = [
                f"river={river}",
                f"tile_id={tile_id}",
                f"final_px={row['final_mask_pixels']}",
                f"pred_patch={row['prediction_patch_count']}",
                f"loss_patch={row['loss_patch_count']}",
                f"core_exact_px={row['core_exact_pixel_count']}",
                f"3DEPfill/final={row['frac_filled3dep_in_final']:.3f}",
                f"3DEPfill/loss={row['frac_filled3dep_in_loss_patch']:.3f}",
                f"3DEPfill/core={row['frac_filled3dep_in_core_exact']:.3f}",
                f"jump_core_px={row['jump_pixels_in_core_exact']}",
                f"source_boundary_core_px={row['source_boundary_in_core_exact_pixels']}",
                f"tile_mean={row['tile_mean_m']:.3f}",
                f"tile_std_safe={row['tile_std_safe']:.3f}",
            ]
            ax.text(0.02, 0.98, "\n".join(key_lines), va="top", ha="left", family="monospace", fontsize=10)

            fig.tight_layout(rect=[0, 0.02, 1, 0.975])
            fig.savefig(tile_out_dir / "source_chain_exactmask.png", dpi=170)
            plt.close(fig)

            print(f"[DONE] {tile_name} -> {tile_out_dir}")

    write_csv(out_dir / "source_chain_exactmask_summary.csv", rows)
    print(f"[SUMMARY] {out_dir / 'source_chain_exactmask_summary.csv'}")


if __name__ == "__main__":
    main()
