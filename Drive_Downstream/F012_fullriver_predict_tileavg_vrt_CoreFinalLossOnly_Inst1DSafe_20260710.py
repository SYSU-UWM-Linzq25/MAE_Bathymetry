#!/usr/bin/env python3
"""F010 full-river MAE inference by per-tile prediction + sparse overlap averaging + VRT.

This version intentionally avoids allocating a dense full-river array.  Some
rivers have a huge diagonal bounding box, so a dense mosaic can require hundreds
of GB even though the actually reconstructed river pixels are sparse.

Workflow
--------
1. Read E001 full-river tiles:
     FullRiver_tile + Hidden_Mask + Loss_Mask_Pixel + Core_Loss_Mask_Pixel
2. Run the downstream dual-mask MAE model using Hidden_Mask as model visibility.
3. Keep ONLY pixels in Core_Loss_Mask_Pixel as final reconstruction footprint.
4. Convert each tile prediction back to meters using tile-wise visible valid norm.
5. Detect overlap by exact georeferenced pixel coordinates.
6. Average overlapping pixels.
7. Write small averaged per-tile GeoTIFFs and a VRT mosaic that references them.

Important
---------
Hidden_Mask is only model input visibility.  It is NOT the final output mask.
Final output footprint is strictly Core_Loss_Mask_Pixel.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except Exception:
    tifffile = None
    TIFFFILE_AVAILABLE = False


@dataclass(frozen=True)
class SimpleAffine:
    """Minimal north-up affine: x = a*col + c, y = e*row + f."""
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


@dataclass
class TileItem:
    key: str
    river: str
    tile_id: int
    tile_path: Path
    hidden_path: Path
    loss_path: Optional[Path]
    core_path: Optional[Path]
    core_loss_path: Optional[Path]


@dataclass
class SparseTileRecord:
    river: str
    tile_id: int
    key: str
    meta: Dict[str, Any]
    local_flat: np.ndarray
    global_key: np.ndarray
    row0: int
    col0: int
    h: int
    w: int
    tile_path: str
    hidden_path: str
    loss_path: str
    core_loss_path: str


TILE_RE = re.compile(
    r"^E001_FullRiver_tile_(?P<res>\d+m)_(?P<river>.+)_ID(?P<id>\d+)\.tif$",
    re.IGNORECASE,
)

DEFAULT_RIVERS = [
    "BadgerFinNull",
    "Estabrook_Combined",
    "KewaFix2Null",
    "Kletzch_Combined_UpMax3Null",
    "CA_KlamathRiver_TopoBathy_2018_D18",
    "CO_UpperColorado_Topobathy_1_2020",
    "MD_PotomacRiver_Bathy_2019",
    "NE_Niobrara_Topobathy_2018",
    "OR_MKRC_Topobathy_2021",
    "OR_SantiamRiverTB_Topobathy_1_D23",
    "WA_ChehalisRiverTB_Topobathy_1_D23",
    "WA_Nisqually_Bathymetric_2020",
]

HOLDOUT_TO_RIVERS = {
    "CO": ["CO_UpperColorado_Topobathy_1_2020"],
    "CA": ["CA_KlamathRiver_TopoBathy_2018_D18"],
    "Santiam": ["OR_SantiamRiverTB_Topobathy_1_D23"],
    "NE": ["NE_Niobrara_Topobathy_2018"],
    "OR_MKRC": ["OR_MKRC_Topobathy_2021"],
    "Nisqually": ["WA_Nisqually_Bathymetric_2020"],
    "MD": ["MD_PotomacRiver_Bathy_2019"],
    "Chehalis": ["WA_ChehalisRiverTB_Topobathy_1_D23"],
    "MilwaukeeGroup": ["BadgerFinNull", "Estabrook_Combined", "KewaFix2Null", "Kletzch_Combined_UpMax3Null"],
}


def add_code_path(code_dir: str) -> None:
    p = str(Path(code_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _norm_tag_value(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return [_norm_tag_value(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_norm_tag_value(x) for x in v]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    if isinstance(v, np.generic):
        return v.item()
    return v


def _tag_value(tags, code_or_name, default=None):
    tag = tags.get(code_or_name)
    if tag is None:
        return default
    return tag.value


def _parse_nodata(tags) -> Optional[float]:
    val = _tag_value(tags, 42113, None)
    if val is None:
        val = _tag_value(tags, "GDAL_NODATA", None)
    if val is None:
        return None
    try:
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="ignore")
        if isinstance(val, (tuple, list)):
            val = val[0]
        return float(str(val).strip().strip("\x00"))
    except Exception:
        return None


def _geo_tags_from_tifffile(tags) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for code in (34735, 34736, 34737):
        val = _tag_value(tags, code, None)
        if val is not None:
            out[str(code)] = _norm_tag_value(val)
    return out


def _transform_from_tiff_tags(tags) -> SimpleAffine:
    scale = _tag_value(tags, 33550, None) or _tag_value(tags, "ModelPixelScaleTag", None)
    tie = _tag_value(tags, 33922, None) or _tag_value(tags, "ModelTiepointTag", None)
    matrix = _tag_value(tags, 34264, None) or _tag_value(tags, "ModelTransformationTag", None)

    if scale is not None and tie is not None:
        scale = tuple(float(x) for x in scale)
        tie = tuple(float(x) for x in tie)
        if len(scale) < 2 or len(tie) < 6:
            raise RuntimeError("Invalid GeoTIFF ModelPixelScale/ModelTiepoint tags.")
        sx, sy = abs(scale[0]), abs(scale[1])
        raster_x, raster_y = tie[0], tie[1]
        model_x, model_y = tie[3], tie[4]
        c = model_x - raster_x * sx
        f = model_y + raster_y * sy
        return SimpleAffine(sx, 0.0, c, 0.0, -sy, f)

    if matrix is not None:
        m = tuple(float(x) for x in matrix)
        if len(m) != 16:
            raise RuntimeError("Invalid GeoTIFF ModelTransformationTag.")
        return SimpleAffine(float(m[0]), float(m[1]), float(m[3]), float(m[4]), float(m[5]), float(m[7]))

    raise RuntimeError("Missing GeoTIFF georeference tags.")


def _crs_wkt_from_tags(tags) -> str:
    val = _tag_value(tags, 34737, "")
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore").strip("\x00")
    return str(val).strip("\x00") if val is not None else ""


def _extratags_for_geotiff(transform: SimpleAffine, crs_tags: Dict[str, Any], nodata: Optional[float]):
    extratags = [
        (33550, "d", 3, (abs(float(transform.a)), abs(float(transform.e)), 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, float(transform.c), float(transform.f), 0.0), False),
    ]

    if "34735" in crs_tags:
        v = tuple(int(x) for x in np.asarray(crs_tags["34735"]).ravel().tolist())
        extratags.append((34735, "H", len(v), v, False))

    if "34736" in crs_tags:
        v = tuple(float(x) for x in np.asarray(crs_tags["34736"]).ravel().tolist())
        extratags.append((34736, "d", len(v), v, False))

    if "34737" in crs_tags:
        v = crs_tags["34737"]
        if isinstance(v, (list, tuple)):
            v = "".join(str(x) for x in v)
        else:
            v = str(v)
        if not v.endswith("\x00"):
            v += "\x00"
        extratags.append((34737, "s", len(v), v, False))

    if nodata is not None:
        nd = str(nodata)
        if not nd.endswith("\x00"):
            nd += "\x00"
        extratags.append((42113, "s", len(nd), nd, False))

    return extratags


def _write_world_file(path: Path, transform: SimpleAffine) -> None:
    world = path.with_suffix(".tfw")
    x_center = float(transform.c) + float(transform.a) / 2.0
    y_center = float(transform.f) + float(transform.e) / 2.0
    world.write_text(
        f"{float(transform.a):.12f}\n"
        f"{float(transform.d):.12f}\n"
        f"{float(transform.b):.12f}\n"
        f"{float(transform.e):.12f}\n"
        f"{x_center:.12f}\n"
        f"{y_center:.12f}\n"
    )


def _write_crs_sidecar(path: Path, crs_wkt: str) -> None:
    if crs_wkt:
        path.with_suffix(".prj").write_text(str(crs_wkt))


def read_one(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    path = Path(path)
    if not TIFFFILE_AVAILABLE:
        raise RuntimeError("tifffile is required.")

    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        tags = page.tags
        transform = _transform_from_tiff_tags(tags)
        nodata = _parse_nodata(tags)
        crs_tags = _geo_tags_from_tifffile(tags)
        crs_wkt = _crs_wkt_from_tags(tags)

    meta = {
        "transform": transform,
        "crs_key": json.dumps(crs_tags, sort_keys=True),
        "crs_tags": crs_tags,
        "crs_wkt": crs_wkt,
        "nodata": nodata,
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "dtype": str(arr.dtype),
    }
    return arr, meta


def write_tif(path: Path, arr: np.ndarray, transform: SimpleAffine, meta_or_crs: Dict[str, Any], nodata, dtype: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not TIFFFILE_AVAILABLE:
        raise RuntimeError("tifffile is required.")

    arr_out = arr.astype(dtype, copy=False)
    crs_tags = meta_or_crs.get("crs_tags", {}) if isinstance(meta_or_crs, dict) else {}
    crs_wkt = meta_or_crs.get("crs_wkt", "") if isinstance(meta_or_crs, dict) else ""
    extratags = _extratags_for_geotiff(transform, crs_tags, nodata)

    tifffile.imwrite(
        str(path),
        arr_out,
        dtype=arr_out.dtype,
        bigtiff=False,
        photometric="minisblack",
        metadata=None,
        extratags=extratags,
    )
    _write_world_file(path, transform)
    _write_crs_sidecar(path, crs_wkt)


def parse_tile_path(path: Path) -> Tuple[str, str, int]:
    m = TILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Cannot parse E001 full-river tile name: {path.name}")
    return m.group("res"), m.group("river"), int(m.group("id"))


def make_key(res: str, river: str, tile_id: int) -> str:
    return f"{res}_{river}_ID{tile_id}"


def block_any(mask: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = mask.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"shape {mask.shape} not divisible by patch_size={patch_size}")
    return mask.reshape(h // patch_size, patch_size, w // patch_size, patch_size).any(axis=(1, 3))


def block_all(mask: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = mask.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"shape {mask.shape} not divisible by patch_size={patch_size}")
    return mask.reshape(h // patch_size, patch_size, w // patch_size, patch_size).all(axis=(1, 3))


def expand_patch_mask(mask21: np.ndarray, patch_size: int) -> np.ndarray:
    return np.repeat(np.repeat(mask21.astype(bool), patch_size, axis=0), patch_size, axis=1)


def make_core_mask(tile_size: int, patch_size: int, radius: int) -> np.ndarray:
    n = tile_size // patch_size
    cy = n // 2
    cx = n // 2
    y0 = max(0, cy - radius)
    y1 = min(n, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(n, cx + radius + 1)
    patch = np.zeros((n, n), dtype=bool)
    patch[y0:y1, x0:x1] = True
    return expand_patch_mask(patch, patch_size)


def is_valid_dem(arr: np.ndarray, nodata: float, threshold: float, src_nodata: Optional[float]) -> np.ndarray:
    a = arr.astype(np.float64, copy=False)
    valid = np.isfinite(a) & (a > float(threshold)) & (a != float(nodata))
    if src_nodata is not None and math.isfinite(float(src_nodata)) and abs(float(src_nodata)) > 1e-100:
        valid &= (a != float(src_nodata))
    return valid


def compute_tile_norm(dem: np.ndarray, valid: np.ndarray, hidden: np.ndarray, std_scale: float, eps: float):
    visible = valid & (~hidden)
    use = visible
    if int(use.sum()) < 2:
        use = valid
    vals = dem.astype(np.float64)[use]
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan"), float("nan"), float("nan"), int(visible.sum()), int(valid.sum())
    mean_m = float(vals.mean())
    std_m = float(vals.std())
    std_safe = float(max(std_m * float(std_scale), float(eps)))
    return mean_m, std_m, std_safe, int(visible.sum()), int(valid.sum())


def collect_tiles(tile_root: Path, rivers: Sequence[str], res: str) -> List[TileItem]:
    full_dir = tile_root / "FullRiver_tile"
    hidden_dir = tile_root / "Hidden_Mask"
    loss_dir = tile_root / "Loss_Mask_Pixel"
    core_dir = tile_root / "Core_Mask_Pixel"
    core_loss_dir = tile_root / "Core_Loss_Mask_Pixel"

    for d in [full_dir, hidden_dir, loss_dir, core_loss_dir]:
        if not d.is_dir():
            raise FileNotFoundError(f"Missing required E001 directory: {d}")

    river_set = set(rivers)
    items: List[TileItem] = []
    for tp in sorted(full_dir.glob(f"E001_FullRiver_tile_{res}_*_ID*.tif")):
        rres, river, tile_id = parse_tile_path(tp)
        if rres != res or river not in river_set:
            continue
        key = make_key(rres, river, tile_id)
        hp = hidden_dir / f"E001_tile_{key}_HiddenMask.tif"
        lp = loss_dir / f"E001_tile_{key}_LossMaskPixel.tif"
        cp = core_dir / f"E001_tile_{key}_CoreMaskPixel.tif"
        clp = core_loss_dir / f"E001_tile_{key}_CoreLossMaskPixel.tif"
        for p in [hp, lp, clp]:
            if not p.exists():
                raise FileNotFoundError(f"Missing paired mask for {tp.name}: {p}")
        items.append(TileItem(
            key=key,
            river=river,
            tile_id=tile_id,
            tile_path=tp,
            hidden_path=hp,
            loss_path=lp,
            core_path=cp if cp.exists() else None,
            core_loss_path=clp,
        ))
    if not items:
        raise RuntimeError(f"No E001 full-river tiles found under {full_dir} for rivers={rivers}")
    return items


def load_model(args, device: torch.device):
    add_code_path(args.code_dir)
    import models_mae

    model = models_mae.__dict__[args.model](
        norm_pix_loss=False,
        img_size=args.input_size,
        in_chans=args.in_chans,
        bottleneck_norm=args.bottleneck_norm,
        loss_mode="mse",
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    msg = model.load_state_dict(state, strict=False)
    print(f"[CKPT] loaded: {args.ckpt}")
    print(f"[CKPT] missing_keys={msg.missing_keys}")
    print(f"[CKPT] unexpected_keys={msg.unexpected_keys}")
    model.to(device)
    model.eval()
    return model


def prepare_sample(item: TileItem, args) -> Dict[str, Any]:
    dem, meta = read_one(item.tile_path)
    hidden_arr, _ = read_one(item.hidden_path)
    loss_arr, _ = read_one(item.loss_path)
    core_loss_arr, _ = read_one(item.core_loss_path)

    dem = dem.astype(np.float32)
    hidden = (hidden_arr.astype(np.float32) > 0.5) & np.isfinite(hidden_arr)
    loss = (loss_arr.astype(np.float32) > 0.5) & np.isfinite(loss_arr)
    core_loss = (core_loss_arr.astype(np.float32) > 0.5) & np.isfinite(core_loss_arr)

    valid = is_valid_dem(dem, args.nodata, args.nodata_threshold, meta.get("nodata"))

    mean_m, std_m, std_safe, n_visible, n_valid = compute_tile_norm(
        dem, valid, hidden, args.tile_norm_std_scale, args.tile_norm_eps
    )
    if not np.isfinite(mean_m) or not np.isfinite(std_safe):
        return {
            "skip_reason": "not_enough_valid_pixels_for_tile_norm",
            "item": item,
            "n_visible_pixels": n_visible,
            "n_valid_pixels": n_valid,
        }

    valid_patch = block_all(valid, args.patch_size)
    hidden_patch = block_any(hidden, args.patch_size)
    visible_patch_count = int((valid_patch & (~hidden_patch)).sum())
    prediction_patch_count = int((valid_patch & hidden_patch).sum())
    core_loss_valid_pixel_count = int((core_loss & valid).sum())

    # Exact-LCC inference can leave a different number of visible tokens per tile.
    # InstanceNorm1d with track_running_stats=False still computes per-instance
    # statistics during model.eval(), and therefore requires token length >= 2.
    # A tile with exactly one visible patch would otherwise fail inside
    # _apply_bottleneck_norm with input shape [1, embed_dim, 1].
    min_visible_patches = 2 if args.bottleneck_norm == "inst1d" else 1
    if visible_patch_count < min_visible_patches:
        reason = (
            "fewer_than_2_valid_visible_patches_for_inst1d_exact_encoder"
            if args.bottleneck_norm == "inst1d"
            else "zero_valid_visible_patches_for_exact_encoder"
        )
        return {
            "skip_reason": reason,
            "item": item,
            "n_visible_pixels": n_visible,
            "n_valid_pixels": n_valid,
            "visible_patch_count": visible_patch_count,
            "min_required_visible_patches": min_visible_patches,
            "prediction_patch_count": prediction_patch_count,
            "core_loss_valid_pixel_count": core_loss_valid_pixel_count,
        }

    sample_norm = (dem.astype(np.float32) - np.float32(mean_m)) / np.float32(std_safe)
    sample_norm[~valid] = 0.0

    # Final footprint is strictly Core_Loss_Mask_Pixel; Hidden_Mask is not used here.
    mosaic_mask = core_loss & valid

    return {
        "item": item,
        "sample_norm": sample_norm,
        "hidden": hidden.astype(np.float32),
        "valid": valid.astype(np.float32),
        "loss": loss.astype(np.float32),
        "core_loss": core_loss,
        "mosaic_mask": mosaic_mask,
        "meta": meta,
        "tile_mean_m": mean_m,
        "tile_std_m": std_m,
        "tile_std_safe": std_safe,
        "n_visible_pixels": n_visible,
        "n_valid_pixels": n_valid,
        "visible_patch_count": visible_patch_count,
        "prediction_patch_count": prediction_patch_count,
        "core_loss_valid_pixel_count": core_loss_valid_pixel_count,
    }


def union_grid(tile_metas: Sequence[Dict[str, Any]]) -> Tuple[SimpleAffine, Dict[str, Any], int, int]:
    if not tile_metas:
        raise ValueError("No tile metadata for union grid.")
    crs_key = tile_metas[0]["crs_key"]
    crs_out = {"crs_tags": tile_metas[0].get("crs_tags", {}), "crs_wkt": tile_metas[0].get("crs_wkt", "")}
    t0 = tile_metas[0]["transform"]
    resx = abs(float(t0.a))
    resy = abs(float(t0.e))
    lefts, rights, tops, bottoms = [], [], [], []
    for m in tile_metas:
        if m["crs_key"] != crs_key:
            raise ValueError("CRS mismatch among tiles.")
        t = m["transform"]
        if abs(abs(float(t.a)) - resx) > 1e-6 or abs(abs(float(t.e)) - resy) > 1e-6:
            raise ValueError("Resolution mismatch among tiles.")
        h, w = int(m["height"]), int(m["width"])
        left = float(t.c)
        top = float(t.f)
        right = left + w * resx
        bottom = top - h * resy
        lefts.append(left); rights.append(right); tops.append(top); bottoms.append(bottom)
    left = min(lefts)
    right = max(rights)
    top = max(tops)
    bottom = min(bottoms)
    width = int(round((right - left) / resx))
    height = int(round((top - bottom) / resy))
    transform = SimpleAffine(resx, 0.0, left, 0.0, -resy, top)
    return transform, crs_out, height, width


def mosaic_offset(tile_transform: SimpleAffine, mosaic_transform: SimpleAffine) -> Tuple[int, int]:
    resx = abs(float(mosaic_transform.a))
    resy = abs(float(mosaic_transform.e))
    col = int(round((float(tile_transform.c) - float(mosaic_transform.c)) / resx))
    row = int(round((float(mosaic_transform.f) - float(tile_transform.f)) / resy))
    return row, col


def vrt_dtype(dtype: str) -> str:
    d = str(dtype).lower()
    if d in ("float32", "single"):
        return "Float32"
    if d in ("uint16",):
        return "UInt16"
    if d in ("uint8", "byte"):
        return "Byte"
    return "Float32"


def write_vrt(
    vrt_path: Path,
    sources: Sequence[Dict[str, Any]],
    mosaic_transform: SimpleAffine,
    width: int,
    height: int,
    nodata: float,
    dtype: str,
    crs_wkt: str = "",
) -> None:
    """Write a simple VRT.  Overlapping source pixels already contain identical averaged values."""
    vrt_path = Path(vrt_path)
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    gt = (
        f"{mosaic_transform.c:.12f}, {mosaic_transform.a:.12f}, {mosaic_transform.b:.12f}, "
        f"{mosaic_transform.f:.12f}, {mosaic_transform.d:.12f}, {mosaic_transform.e:.12f}"
    )
    lines = [
        f'<VRTDataset rasterXSize="{int(width)}" rasterYSize="{int(height)}">',
        f'  <GeoTransform>{gt}</GeoTransform>',
    ]
    if crs_wkt and any(crs_wkt.startswith(prefix) for prefix in ("PROJCS", "GEOGCS", "PROJCRS", "GEOGCRS")):
        lines.append(f'  <SRS>{html.escape(crs_wkt)}</SRS>')
    lines += [
        f'  <VRTRasterBand dataType="{vrt_dtype(dtype)}" band="1">',
        f'    <NoDataValue>{nodata}</NoDataValue>',
    ]
    for s in sources:
        src = html.escape(str(Path(s["path"]).resolve()))
        lines += [
            "    <ComplexSource>",
            f'      <SourceFilename relativeToVRT="0">{src}</SourceFilename>',
            "      <SourceBand>1</SourceBand>",
            f'      <SrcRect xOff="0" yOff="0" xSize="{int(s["w"])}" ySize="{int(s["h"])}"/>',
            f'      <DstRect xOff="{int(s["col0"])}" yOff="{int(s["row0"])}" xSize="{int(s["w"])}" ySize="{int(s["h"])}"/>',
            f'      <NODATA>{nodata}</NODATA>',
            "    </ComplexSource>",
        ]
    lines += ["  </VRTRasterBand>", "</VRTDataset>", ""]
    vrt_path.write_text("\n".join(lines))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def infer_river(model, items: Sequence[TileItem], river: str, args, device: torch.device) -> Dict[str, Any]:
    out_river = Path(args.output_dir) / river
    avg_tile_dir = out_river / "tile_predictions_core_final_loss_avg"
    count_tile_dir = out_river / "tile_overlap_count_core_final_loss"
    out_river.mkdir(parents=True, exist_ok=True)
    avg_tile_dir.mkdir(parents=True, exist_ok=True)
    count_tile_dir.mkdir(parents=True, exist_ok=True)

    prepared: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for item in items:
        rec = prepare_sample(item, args)
        if "skip_reason" in rec:
            skipped.append({
                "river": item.river,
                "tile_id": item.tile_id,
                "key": item.key,
                "tile_path": str(item.tile_path),
                **{k: v for k, v in rec.items() if k != "item"},
            })
        else:
            prepared.append(rec)

    if not prepared:
        raise RuntimeError(f"All tiles skipped for river={river}. See skip reasons.")

    mosaic_transform, mosaic_crs, height, width = union_grid([r["meta"] for r in prepared])
    print(f"[RIVER] {river}: prepared={len(prepared)}, skipped={len(skipped)}, virtual_grid={height}x{width}, dense_alloc=NO")

    sparse_records: List[SparseTileRecord] = []
    key_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []

    batch_size = int(args.batch_size)
    n = len(prepared)

    for start in range(0, n, batch_size):
        batch = prepared[start:start + batch_size]
        xb = torch.from_numpy(np.stack([r["sample_norm"][None, :, :] for r in batch], axis=0)).to(device=device, dtype=torch.float32)
        hb = torch.from_numpy(np.stack([r["hidden"][None, :, :] for r in batch], axis=0)).to(device=device, dtype=torch.float32)
        vb = torch.from_numpy(np.stack([r["valid"][None, :, :] for r in batch], axis=0)).to(device=device, dtype=torch.float32)
        lb = torch.from_numpy(np.stack([r["loss"][None, :, :] for r in batch], axis=0)).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                _, pred, _, _ = model(
                    xb,
                    mask_ratio=args.mask_ratio,
                    lcc_mask=hb,
                    valid_mask=vb,
                    loss_pixel_mask=lb,
                    loss_on_lcc_only=False,
                    lcc_mask_mode="exact",
                    lcc_patch_threshold=args.lcc_patch_threshold,
                    loss_region_mode=args.loss_region_mode,
                    core_patch_radius=args.core_patch_radius,
                    return_aux_masks=True,
                )
            pred_img = model.unpatchify(pred).detach().float().cpu().numpy()[:, 0]

        for j, rec in enumerate(batch):
            item: TileItem = rec["item"]
            pred_norm = pred_img[j]
            pred_m = (pred_norm * float(rec["tile_std_safe"]) + float(rec["tile_mean_m"])).astype(np.float32)

            mask = rec["mosaic_mask"].astype(bool) & np.isfinite(pred_m)
            local_flat = np.flatnonzero(mask.ravel()).astype(np.int32)
            if local_flat.size == 0:
                skipped.append({
                    "river": item.river,
                    "tile_id": item.tile_id,
                    "key": item.key,
                    "tile_path": str(item.tile_path),
                    "skip_reason": "zero_core_final_loss_pixels_after_valid",
                })
                continue

            h, w = pred_m.shape
            row0, col0 = mosaic_offset(rec["meta"]["transform"], mosaic_transform)
            rr = (local_flat // w).astype(np.int64)
            cc = (local_flat % w).astype(np.int64)
            global_key = (np.int64(row0) + rr) * np.int64(width) + (np.int64(col0) + cc)
            vals = pred_m.ravel()[local_flat].astype(np.float32)

            key_parts.append(global_key)
            val_parts.append(vals)
            sparse_records.append(SparseTileRecord(
                river=river,
                tile_id=item.tile_id,
                key=item.key,
                meta=rec["meta"],
                local_flat=local_flat,
                global_key=global_key,
                row0=row0,
                col0=col0,
                h=h,
                w=w,
                tile_path=str(item.tile_path),
                hidden_path=str(item.hidden_path),
                loss_path=str(item.loss_path),
                core_loss_path=str(item.core_loss_path),
            ))

        done = min(start + batch_size, n)
        if start == 0 or done == n or done % max(batch_size * 20, 1) == 0:
            print(f"  [{river}] predicted sparse tiles {done}/{n}")

    if not sparse_records:
        raise RuntimeError(f"No sparse predictions produced for river={river}.")

    print(f"[AVG] {river}: concatenating sparse pixels...")
    all_keys = np.concatenate(key_parts).astype(np.int64, copy=False)
    all_vals = np.concatenate(val_parts).astype(np.float32, copy=False)
    print(f"[AVG] {river}: sparse_input_pixels={all_keys.size}")

    order = np.argsort(all_keys, kind="mergesort")
    keys_sorted = all_keys[order]
    vals_sorted = all_vals[order].astype(np.float64, copy=False)
    uniq_keys, starts = np.unique(keys_sorted, return_index=True)
    sums = np.add.reduceat(vals_sorted, starts)
    counts64 = np.diff(np.r_[starts, keys_sorted.size]).astype(np.int64)
    avg_vals = (sums / counts64).astype(np.float32)
    counts = np.minimum(counts64, np.iinfo(np.uint16).max).astype(np.uint16)

    del all_keys, all_vals, order, keys_sorted, vals_sorted, sums

    print(f"[AVG] {river}: unique_output_pixels={uniq_keys.size}, max_overlap={int(counts.max())}")

    manifest_rows: List[Dict[str, Any]] = []
    avg_vrt_sources: List[Dict[str, Any]] = []
    count_vrt_sources: List[Dict[str, Any]] = []

    for rec in sparse_records:
        pos = np.searchsorted(uniq_keys, rec.global_key)
        ok = (pos < uniq_keys.size) & (uniq_keys[pos] == rec.global_key)
        if not bool(np.all(ok)):
            raise RuntimeError(f"Internal sparse lookup failed for {rec.key}")

        tile_avg = np.full((rec.h, rec.w), float(args.nodata), dtype=np.float32)
        tile_count = np.zeros((rec.h, rec.w), dtype=np.uint16)
        tile_avg.ravel()[rec.local_flat] = avg_vals[pos]
        tile_count.ravel()[rec.local_flat] = counts[pos]

        avg_path = avg_tile_dir / f"F010_avg_pred_m_{rec.key}_core_final_loss.tif"
        cnt_path = count_tile_dir / f"F010_overlap_count_{rec.key}_core_final_loss.tif"
        write_tif(avg_path, tile_avg, rec.meta["transform"], rec.meta, args.nodata, dtype="float32")
        write_tif(cnt_path, tile_count, rec.meta["transform"], rec.meta, 0, dtype="uint16")

        avg_vrt_sources.append({"path": str(avg_path), "row0": rec.row0, "col0": rec.col0, "h": rec.h, "w": rec.w})
        count_vrt_sources.append({"path": str(cnt_path), "row0": rec.row0, "col0": rec.col0, "h": rec.h, "w": rec.w})

        this_counts = counts[pos]
        manifest_rows.append({
            "river": river,
            "tile_id": rec.tile_id,
            "key": rec.key,
            "tile_path": rec.tile_path,
            "hidden_path": rec.hidden_path,
            "loss_path": rec.loss_path,
            "core_loss_path": rec.core_loss_path,
            "avg_pred_tile_path": str(avg_path),
            "overlap_count_tile_path": str(cnt_path),
            "mosaic_row0": rec.row0,
            "mosaic_col0": rec.col0,
            "tile_height": rec.h,
            "tile_width": rec.w,
            "core_final_loss_pixels": int(rec.local_flat.size),
            "tile_overlap_min": int(this_counts.min()) if this_counts.size else 0,
            "tile_overlap_mean": float(this_counts.mean()) if this_counts.size else 0.0,
            "tile_overlap_max": int(this_counts.max()) if this_counts.size else 0,
        })

    pred_vrt = out_river / f"F010_fullriver_pred_m_{river}_core_final_loss_avg_tiles.vrt"
    count_vrt = out_river / f"F010_fullriver_overlap_count_{river}_core_final_loss_avg_tiles.vrt"
    write_vrt(pred_vrt, avg_vrt_sources, mosaic_transform, width, height, args.nodata, "float32", mosaic_crs.get("crs_wkt", ""))
    write_vrt(count_vrt, count_vrt_sources, mosaic_transform, width, height, 0, "uint16", mosaic_crs.get("crs_wkt", ""))

    write_csv(out_river / "F010_tileavg_prediction_manifest.csv", manifest_rows)
    write_csv(out_river / "F010_skipped_tiles.csv", skipped)

    skip_reason_counts: Dict[str, int] = {}
    skipped_core_loss_valid_pixels = 0
    for row in skipped:
        reason = str(row.get("skip_reason", "unknown"))
        skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
        try:
            skipped_core_loss_valid_pixels += int(row.get("core_loss_valid_pixel_count", 0) or 0)
        except Exception:
            pass

    summary = {
        "river": river,
        "checkpoint": str(args.ckpt),
        "n_tiles_total": len(items),
        "n_tiles_prepared": len(prepared),
        "n_tiles_written": len(sparse_records),
        "n_tiles_skipped": len(skipped),
        "skip_reason_counts": skip_reason_counts,
        "skipped_core_loss_valid_pixels_before_overlap_recovery": skipped_core_loss_valid_pixels,
        "dense_mosaic_allocated": False,
        "virtual_mosaic_height": height,
        "virtual_mosaic_width": width,
        "mosaic_mask": "Core_Loss_Mask_Pixel only",
        "hidden_mask_role": "model input visibility only",
        "sparse_input_pixels_before_averaging": int(sum(len(x) for x in key_parts)),
        "unique_output_pixels_after_averaging": int(uniq_keys.size),
        "mean_overlap_count": float(counts.mean()) if counts.size else None,
        "max_overlap_count": int(counts.max()) if counts.size else 0,
        "pred_vrt_path": str(pred_vrt),
        "count_vrt_path": str(count_vrt),
        "avg_tile_dir": str(avg_tile_dir),
        "count_tile_dir": str(count_tile_dir),
    }
    (out_river / "F010_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def resolve_rivers(args) -> List[str]:
    if args.rivers:
        rivers = list(args.rivers)
    elif args.holdout_preset:
        if args.holdout_preset not in HOLDOUT_TO_RIVERS:
            raise ValueError(f"Unknown holdout_preset={args.holdout_preset}. Allowed={sorted(HOLDOUT_TO_RIVERS)}")
        rivers = HOLDOUT_TO_RIVERS[args.holdout_preset]
    else:
        raise ValueError("Set --rivers or --holdout_preset.")
    for r in rivers:
        if r not in DEFAULT_RIVERS:
            raise ValueError(f"Unknown river={r}")
    return rivers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tile_root", default="/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_FullRiver_E001/Tiles_1m")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing output directory: skip a river only when its "
            "F010_summary.json and both VRT outputs are complete; otherwise "
            "remove only that incomplete river directory and rebuild it."
        ),
    )
    ap.add_argument("--rivers", nargs="*", default=[])
    ap.add_argument("--holdout_preset", default="")
    ap.add_argument("--res", default="1m")

    ap.add_argument("--model", default="mae_vit_large_patch16")
    ap.add_argument("--input_size", type=int, default=336)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--in_chans", type=int, default=1)
    ap.add_argument("--bottleneck_norm", default="inst1d", choices=["none", "inst1d"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--nodata", type=float, default=-999999.0)
    ap.add_argument("--nodata_threshold", type=float, default=-9999.0)
    ap.add_argument("--tile_norm_eps", type=float, default=1e-3)
    ap.add_argument("--tile_norm_std_scale", type=float, default=1.5)

    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--lcc_patch_threshold", type=float, default=0.5)
    ap.add_argument("--loss_region_mode", choices=["all", "core"], default="core")
    ap.add_argument("--core_patch_radius", type=int, default=3)
    args = ap.parse_args()

    if args.input_size % args.patch_size:
        raise ValueError("input_size must be divisible by patch_size")
    if not TIFFFILE_AVAILABLE:
        raise RuntimeError("tifffile is required for this F010 tile-average VRT workflow.")

    rivers = resolve_rivers(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    args_payload = {**vars(args), "resolved_rivers": rivers}
    # Preserve the original invocation metadata when resuming only a subset.
    args_file = out / "F010_args.json"
    if not args_file.exists():
        args_file.write_text(json.dumps(args_payload, indent=2))
    (out / "F010_args_last_invocation.json").write_text(json.dumps(args_payload, indent=2))

    print("[IO] TileAvgVRT=True dense_mosaic_allocated=False tifffile_only=True")
    print("[MASK] final footprint = Core_Loss_Mask_Pixel only; Hidden_Mask = model input visibility only")
    min_vis = 2 if args.bottleneck_norm == "inst1d" else 1
    print(f"[GUARD] exact encoder requires >= {min_vis} valid visible patch(es) for bottleneck_norm={args.bottleneck_norm}; smaller tiles will be logged and skipped")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    model = load_model(args, device)
    all_items = collect_tiles(Path(args.tile_root), rivers, args.res)
    summaries = []

    def completed_summary(summary_path: Path) -> Optional[Dict[str, Any]]:
        if not summary_path.is_file():
            return None
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            print(f"[RESUME] Invalid summary JSON, will rebuild: {summary_path}: {exc}")
            return None
        required = [summary.get("pred_vrt_path"), summary.get("count_vrt_path")]
        if not all(p and Path(p).is_file() for p in required):
            print(f"[RESUME] Summary exists but required VRT is missing; will rebuild: {summary_path.parent}")
            return None
        return summary

    for river in rivers:
        river_out = out / river
        summary_path = river_out / "F010_summary.json"

        if args.resume:
            previous = completed_summary(summary_path)
            if previous is not None:
                print(f"[RESUME] complete river skipped: {river}")
                summaries.append(previous)
                continue

            if river_out.exists():
                print(f"[RESUME] removing incomplete river output before rebuild: {river_out}")
                shutil.rmtree(river_out)
        elif river_out.exists() and any(river_out.iterdir()):
            raise RuntimeError(
                f"River output already exists and is non-empty: {river_out}. "
                "Use --resume to preserve completed rivers and rebuild only incomplete rivers."
            )

        items = [it for it in all_items if it.river == river]
        summaries.append(infer_river(model, items, river, args, device))

    # Rebuild the root summary from every completed river currently present, not
    # only from the subset requested by this resume invocation.
    all_completed: List[Dict[str, Any]] = []
    for summary_path in sorted(out.glob("*/F010_summary.json")):
        summary = completed_summary(summary_path)
        if summary is not None:
            all_completed.append(summary)
    (out / "F010_all_rivers_summary.json").write_text(json.dumps(all_completed, indent=2))
    print(f"[SUMMARY] completed_rivers_in_output={len(all_completed)}")
    print("[DONE]", out)


if __name__ == "__main__":
    main()
