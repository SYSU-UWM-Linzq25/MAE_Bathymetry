#!/usr/bin/env python3
# NUMBER-ALIGNED NEW FAMILY COPY: F027_fullriver_gt_error_NormThenMeter_UniquePixel_20260727.py
# TEMPLATE SOURCE: F032_fullriver_gt_error_MeterOnly_UniquePixel_20260713.py
# Reads legacy F060 manifest/summary product names written by F025 for compatibility.
# NUMBER-ALIGNED NAME: F032_fullriver_gt_error_MeterOnly_UniquePixel_20260713.py
# ORIGINAL BACKUP NAME: F027_fullriver_gt_error_uniquePixel_MeterMAE_20260713.py
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
"""F027: build GT/error GeoTIFFs and exact unique-pixel metrics for F025 NormThenMeter full-river predictions.

Inputs
------
Per-river F025 TileAvgVRT outputs:
  - F025_tileavg_prediction_manifest.csv
  - F025_summary.json
  - tile_predictions_core_final_loss_avg/*.tif

For each averaged prediction tile, this script reads:
  - original E001 FullRiver_tile as ground truth
  - E001 Core_Loss_Mask_Pixel as final comparison footprint
  - F025 averaged prediction tile

Outputs
-------
Per river:
  gt_core_final_loss_tile/
    F027_gt_m_<key>_core_final_loss.tif
  error_signed_m_tile/
    F027_err_signed_m_<key>_core_final_loss.tif
  error_abs_m_tile/
    F027_err_abs_m_<key>_core_final_loss.tif

  F027_fullriver_gt_m_<river>_core_final_loss_tiles.vrt
  F027_fullriver_err_signed_m_<river>_core_final_loss_tiles.vrt
  F027_fullriver_err_abs_m_<river>_core_final_loss_tiles.vrt
  F027_tile_error_metrics.csv
  F027_summary.json

Mask rule
---------
Final comparison footprint is strictly:
    Core_Loss_Mask_Pixel AND valid_GT AND valid_prediction

Hidden_Mask is not used in F027. It was only used in F025 model inference.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except Exception:
    tifffile = None
    TIFFFILE_AVAILABLE = False


@dataclass(frozen=True)
class SimpleAffine:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


def _norm_tag_value(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return [_norm_tag_value(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_norm_tag_value(x) for x in v]
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='ignore')
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
        val = _tag_value(tags, 'GDAL_NODATA', None)
    if val is None:
        return None
    try:
        if isinstance(val, bytes):
            val = val.decode('utf-8', errors='ignore')
        if isinstance(val, (tuple, list)):
            val = val[0]
        return float(str(val).strip().strip('\x00'))
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
    scale = _tag_value(tags, 33550, None) or _tag_value(tags, 'ModelPixelScaleTag', None)
    tie = _tag_value(tags, 33922, None) or _tag_value(tags, 'ModelTiepointTag', None)
    matrix = _tag_value(tags, 34264, None) or _tag_value(tags, 'ModelTransformationTag', None)

    if scale is not None and tie is not None:
        scale = tuple(float(x) for x in scale)
        tie = tuple(float(x) for x in tie)
        if len(scale) < 2 or len(tie) < 6:
            raise RuntimeError('Invalid GeoTIFF ModelPixelScale/ModelTiepoint tags.')
        sx, sy = abs(scale[0]), abs(scale[1])
        raster_x, raster_y = tie[0], tie[1]
        model_x, model_y = tie[3], tie[4]
        c = model_x - raster_x * sx
        f = model_y + raster_y * sy
        return SimpleAffine(sx, 0.0, c, 0.0, -sy, f)

    if matrix is not None:
        m = tuple(float(x) for x in matrix)
        if len(m) != 16:
            raise RuntimeError('Invalid GeoTIFF ModelTransformationTag.')
        return SimpleAffine(float(m[0]), float(m[1]), float(m[3]), float(m[4]), float(m[5]), float(m[7]))

    raise RuntimeError('Missing GeoTIFF georeference tags.')


def _crs_wkt_from_tags(tags) -> str:
    val = _tag_value(tags, 34737, '')
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='ignore').strip('\x00')
    return str(val).strip('\x00') if val is not None else ''


def _extratags_for_geotiff(transform: SimpleAffine, crs_tags: Dict[str, Any], nodata: Optional[float]):
    extratags = [
        (33550, 'd', 3, (abs(float(transform.a)), abs(float(transform.e)), 0.0), False),
        (33922, 'd', 6, (0.0, 0.0, 0.0, float(transform.c), float(transform.f), 0.0), False),
    ]

    if '34735' in crs_tags:
        v = tuple(int(x) for x in np.asarray(crs_tags['34735']).ravel().tolist())
        extratags.append((34735, 'H', len(v), v, False))

    if '34736' in crs_tags:
        v = tuple(float(x) for x in np.asarray(crs_tags['34736']).ravel().tolist())
        extratags.append((34736, 'd', len(v), v, False))

    if '34737' in crs_tags:
        v = crs_tags['34737']
        if isinstance(v, (list, tuple)):
            v = ''.join(str(x) for x in v)
        else:
            v = str(v)
        if not v.endswith('\x00'):
            v += '\x00'
        extratags.append((34737, 's', len(v), v, False))

    if nodata is not None:
        nd = str(nodata)
        if not nd.endswith('\x00'):
            nd += '\x00'
        extratags.append((42113, 's', len(nd), nd, False))

    return extratags


def _write_world_file(path: Path, transform: SimpleAffine) -> None:
    world = path.with_suffix('.tfw')
    x_center = float(transform.c) + float(transform.a) / 2.0
    y_center = float(transform.f) + float(transform.e) / 2.0
    world.write_text(
        f'{float(transform.a):.12f}\n'
        f'{float(transform.d):.12f}\n'
        f'{float(transform.b):.12f}\n'
        f'{float(transform.e):.12f}\n'
        f'{x_center:.12f}\n'
        f'{y_center:.12f}\n'
    )


def _write_crs_sidecar(path: Path, crs_wkt: str) -> None:
    if crs_wkt:
        path.with_suffix('.prj').write_text(str(crs_wkt))


def read_one(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    path = Path(path)
    if not TIFFFILE_AVAILABLE:
        raise RuntimeError('tifffile is required.')

    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        tags = page.tags
        transform = _transform_from_tiff_tags(tags)
        nodata = _parse_nodata(tags)
        crs_tags = _geo_tags_from_tifffile(tags)
        crs_wkt = _crs_wkt_from_tags(tags)

    meta = {
        'transform': transform,
        'crs_tags': crs_tags,
        'crs_wkt': crs_wkt,
        'nodata': nodata,
        'height': int(arr.shape[0]),
        'width': int(arr.shape[1]),
        'dtype': str(arr.dtype),
    }
    return arr, meta


def write_tif(path: Path, arr: np.ndarray, ref_meta: Dict[str, Any], nodata: float, dtype: str = 'float32') -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_out = arr.astype(dtype, copy=False)

    extratags = _extratags_for_geotiff(
        ref_meta['transform'],
        ref_meta.get('crs_tags', {}),
        nodata,
    )
    tifffile.imwrite(
        str(path),
        arr_out,
        dtype=arr_out.dtype,
        bigtiff=False,
        photometric='minisblack',
        metadata=None,
        extratags=extratags,
    )
    _write_world_file(path, ref_meta['transform'])
    _write_crs_sidecar(path, ref_meta.get('crs_wkt', ''))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def valid_gt_mask(arr: np.ndarray, nodata: float, threshold: float, src_nodata: Optional[float]) -> np.ndarray:
    a = arr.astype(np.float64, copy=False)
    valid = np.isfinite(a) & (a > float(threshold)) & (a != float(nodata))
    if src_nodata is not None and math.isfinite(float(src_nodata)) and abs(float(src_nodata)) > 1e-100:
        valid &= (a != float(src_nodata))
    return valid


def valid_pred_mask(arr: np.ndarray, nodata: float) -> np.ndarray:
    a = arr.astype(np.float64, copy=False)
    return np.isfinite(a) & (a != float(nodata))


def stats_signed_error(err: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    vals = np.asarray(err, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            'n_pixels': 0,
            'sse_m2': 0.0,
            'rmse_m': float('nan'),
            'mae_m': float('nan'),
            'bias_m': float('nan'),
            'median_abs_error_m': float('nan'),
            'p90_abs_error_m': float('nan'),
            'p95_abs_error_m': float('nan'),
            'max_abs_error_m': float('nan'),
            'min_signed_error_m': float('nan'),
            'max_signed_error_m': float('nan'),
        }
    av = np.abs(vals)
    sse = float(np.square(vals).sum(dtype=np.float64))
    return {
        'n_pixels': int(vals.size),
        'sse_m2': sse,
        'rmse_m': float(np.sqrt(sse / vals.size)),
        'mae_m': float(av.mean()),
        'bias_m': float(vals.mean()),
        'median_abs_error_m': float(np.median(av)),
        'p90_abs_error_m': float(np.percentile(av, 90)),
        'p95_abs_error_m': float(np.percentile(av, 95)),
        'max_abs_error_m': float(av.max()),
        'min_signed_error_m': float(vals.min()),
        'max_signed_error_m': float(vals.max()),
    }


def parse_vrt_basic(vrt_path: Path) -> Dict[str, Any]:
    root = ET.parse(str(vrt_path)).getroot()
    width = int(root.attrib['rasterXSize'])
    height = int(root.attrib['rasterYSize'])
    gt_txt = root.findtext('GeoTransform')
    if gt_txt:
        vals = [float(x.strip()) for x in gt_txt.split(',')]
        transform = SimpleAffine(a=vals[1], b=vals[2], c=vals[0], d=vals[4], e=vals[5], f=vals[3])
    else:
        transform = None
    srs = root.findtext('SRS') or ''
    return {'width': width, 'height': height, 'transform': transform, 'srs': srs}


def vrt_dtype(dtype: str) -> str:
    d = str(dtype).lower()
    if d in ('float32', 'single'):
        return 'Float32'
    if d in ('float64', 'double'):
        return 'Float64'
    if d in ('uint16',):
        return 'UInt16'
    if d in ('uint8', 'byte'):
        return 'Byte'
    return 'Float32'


def write_vrt(
    vrt_path: Path,
    sources: Sequence[Dict[str, Any]],
    width: int,
    height: int,
    transform: Optional[SimpleAffine],
    nodata: float,
    dtype: str,
    srs: str = '',
) -> None:
    vrt_path = Path(vrt_path)
    vrt_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f'<VRTDataset rasterXSize="{int(width)}" rasterYSize="{int(height)}">',
    ]
    if transform is not None:
        gt = (
            f'{transform.c:.12f}, {transform.a:.12f}, {transform.b:.12f}, '
            f'{transform.f:.12f}, {transform.d:.12f}, {transform.e:.12f}'
        )
        lines.append(f'  <GeoTransform>{gt}</GeoTransform>')
    if srs:
        lines.append(f'  <SRS>{html.escape(srs)}</SRS>')
    lines += [
        f'  <VRTRasterBand dataType="{vrt_dtype(dtype)}" band="1">',
        f'    <NoDataValue>{nodata}</NoDataValue>',
    ]
    for s in sources:
        src = html.escape(str(Path(s['path']).resolve()))
        lines += [
            '    <ComplexSource>',
            f'      <SourceFilename relativeToVRT="0">{src}</SourceFilename>',
            '      <SourceBand>1</SourceBand>',
            f'      <SrcRect xOff="0" yOff="0" xSize="{int(s["w"])}" ySize="{int(s["h"])}"/>',
            f'      <DstRect xOff="{int(s["col0"])}" yOff="{int(s["row0"])}" xSize="{int(s["w"])}" ySize="{int(s["h"])}"/>',
            f'      <NODATA>{nodata}</NODATA>',
            '    </ComplexSource>',
        ]
    lines += ['  </VRTRasterBand>', '</VRTDataset>', '']
    vrt_path.write_text('\n'.join(lines))


def find_river_dirs(f060_out_dir: Path, rivers: Sequence[str]) -> List[Path]:
    f060_out_dir = Path(f060_out_dir)
    if rivers:
        dirs = [f060_out_dir / r for r in rivers]
    else:
        dirs = [p for p in sorted(f060_out_dir.iterdir()) if p.is_dir() and (p / 'F025_summary.json').exists()]
    missing = [str(d) for d in dirs if not (d / 'F025_summary.json').exists()]
    if missing:
        raise FileNotFoundError('Missing F025_summary.json for river dirs:\n' + '\n'.join(missing))
    return dirs


def process_river(river_dir: Path, out_base: Path, args) -> Dict[str, Any]:
    river_dir = Path(river_dir)
    river = river_dir.name
    out_river = Path(out_base) / river
    gt_dir = out_river / 'gt_core_final_loss_tile'
    err_signed_dir = out_river / 'error_signed_m_tile'
    err_abs_dir = out_river / 'error_abs_m_tile'
    gt_dir.mkdir(parents=True, exist_ok=True)
    err_signed_dir.mkdir(parents=True, exist_ok=True)
    err_abs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = river_dir / 'F025_summary.json'
    manifest_path = river_dir / 'F025_tileavg_prediction_manifest.csv'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing F025 manifest: {manifest_path}')

    f060_summary = json.loads(summary_path.read_text())
    pred_vrt = Path(f060_summary.get('pred_vrt_path', ''))
    vrt_info = parse_vrt_basic(pred_vrt) if pred_vrt.exists() else {
        'width': int(f060_summary['virtual_mosaic_width']),
        'height': int(f060_summary['virtual_mosaic_height']),
        'transform': None,
        'srs': '',
    }

    rows = read_csv(manifest_path)
    metric_rows: List[Dict[str, Any]] = []
    gt_sources: List[Dict[str, Any]] = []
    err_signed_sources: List[Dict[str, Any]] = []
    err_abs_sources: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    # Exact final-river metrics: map each comparison pixel to its virtual
    # geospatial mosaic key, then count each overlap-averaged pixel once.
    unique_key_parts: List[np.ndarray] = []
    unique_err_parts: List[np.ndarray] = []

    print(f'[RIVER] {river}: manifest_rows={len(rows)}')

    for idx, r in enumerate(rows, start=1):
        key = r.get('key') or f'tile_{idx:06d}'
        tile_path = Path(r['tile_path'])
        pred_path = Path(r['avg_pred_tile_path'])
        core_loss_path = Path(r['core_loss_path'])

        try:
            gt, gt_meta = read_one(tile_path)
            pred, _ = read_one(pred_path)
            core_loss, _ = read_one(core_loss_path)
        except Exception as exc:
            skipped.append({'key': key, 'tile_path': str(tile_path), 'skip_reason': f'read_error: {exc}'})
            continue

        gt = gt.astype(np.float32, copy=False)
        pred = pred.astype(np.float32, copy=False)
        core_mask = (core_loss.astype(np.float32) > 0.5) & np.isfinite(core_loss)

        vgt = valid_gt_mask(gt, args.nodata, args.nodata_threshold, gt_meta.get('nodata'))
        vpred = valid_pred_mask(pred, args.nodata)
        mask = core_mask & vgt & vpred

        if int(mask.sum()) == 0:
            skipped.append({'key': key, 'tile_path': str(tile_path), 'skip_reason': 'zero_common_pixels_core_loss_gt_pred'})
            continue

        gt_out = np.full(gt.shape, float(args.nodata), dtype=np.float32)
        signed_out = np.full(gt.shape, float(args.nodata), dtype=np.float32)
        abs_out = np.full(gt.shape, float(args.nodata), dtype=np.float32)

        err = pred - gt
        gt_out[mask] = gt[mask]
        signed_out[mask] = err[mask]
        abs_out[mask] = np.abs(err[mask])

        gt_out_path = gt_dir / f'F027_gt_m_{key}_core_final_loss.tif'
        err_signed_path = err_signed_dir / f'F027_err_signed_m_{key}_core_final_loss.tif'
        err_abs_path = err_abs_dir / f'F027_err_abs_m_{key}_core_final_loss.tif'

        write_tif(gt_out_path, gt_out, gt_meta, args.nodata, dtype='float32')
        write_tif(err_signed_path, signed_out, gt_meta, args.nodata, dtype='float32')
        write_tif(err_abs_path, abs_out, gt_meta, args.nodata, dtype='float32')

        h, w = gt.shape
        row0 = int(r['mosaic_row0'])
        col0 = int(r['mosaic_col0'])

        local_flat = np.flatnonzero(mask.ravel()).astype(np.int32)
        rr = (local_flat // w).astype(np.int64)
        cc = (local_flat % w).astype(np.int64)
        global_key = (
            (np.int64(row0) + rr) * np.int64(vrt_info['width'])
            + (np.int64(col0) + cc)
        )
        unique_key_parts.append(global_key)
        unique_err_parts.append(err.ravel()[local_flat].astype(np.float32))

        src_entry = {'row0': row0, 'col0': col0, 'h': h, 'w': w}
        gt_sources.append({**src_entry, 'path': str(gt_out_path)})
        err_signed_sources.append({**src_entry, 'path': str(err_signed_path)})
        err_abs_sources.append({**src_entry, 'path': str(err_abs_path)})

        stats = stats_signed_error(err, mask)
        metric_rows.append({
            'river': river,
            'key': key,
            'tile_id': r.get('tile_id', ''),
            'tile_path': str(tile_path),
            'pred_tile_path': str(pred_path),
            'core_loss_path': str(core_loss_path),
            'gt_tile_out': str(gt_out_path),
            'err_signed_tile_out': str(err_signed_path),
            'err_abs_tile_out': str(err_abs_path),
            'mosaic_row0': row0,
            'mosaic_col0': col0,
            'tile_height': h,
            'tile_width': w,
            **stats,
        })

        if idx == 1 or idx == len(rows) or idx % max(int(args.progress_every), 1) == 0:
            print(f'  [{river}] processed {idx}/{len(rows)} tiles')

    if not metric_rows:
        raise RuntimeError(f'No F027 outputs produced for river={river}.')

    gt_vrt = out_river / f'F027_fullriver_gt_m_{river}_core_final_loss_tiles.vrt'
    err_signed_vrt = out_river / f'F027_fullriver_err_signed_m_{river}_core_final_loss_tiles.vrt'
    err_abs_vrt = out_river / f'F027_fullriver_err_abs_m_{river}_core_final_loss_tiles.vrt'

    write_vrt(gt_vrt, gt_sources, vrt_info['width'], vrt_info['height'], vrt_info['transform'], args.nodata, 'float32', vrt_info.get('srs', ''))
    write_vrt(err_signed_vrt, err_signed_sources, vrt_info['width'], vrt_info['height'], vrt_info['transform'], args.nodata, 'float32', vrt_info.get('srs', ''))
    write_vrt(err_abs_vrt, err_abs_sources, vrt_info['width'], vrt_info['height'], vrt_info['transform'], args.nodata, 'float32', vrt_info.get('srs', ''))

    write_csv(out_river / 'F027_tile_error_metrics.csv', metric_rows)
    write_csv(out_river / 'F027_skipped_tiles.csv', skipped)

    total_n = int(sum(int(r['n_pixels']) for r in metric_rows))
    total_sse = float(sum(float(r['sse_m2']) for r in metric_rows))
    mae_num = 0.0
    bias_num = 0.0
    for r in metric_rows:
        n = int(r['n_pixels'])
        mae_num += float(r['mae_m']) * n
        bias_num += float(r['bias_m']) * n

    if not unique_key_parts:
        raise RuntimeError(f'No unique comparison pixels collected for river={river}.')

    all_keys = np.concatenate(unique_key_parts).astype(np.int64, copy=False)
    all_err = np.concatenate(unique_err_parts).astype(np.float64, copy=False)
    order = np.argsort(all_keys, kind='mergesort')
    keys_sorted = all_keys[order]
    err_sorted = all_err[order]
    uniq_keys, starts = np.unique(keys_sorted, return_index=True)
    err_sums = np.add.reduceat(err_sorted, starts)
    duplicate_counts = np.diff(np.r_[starts, keys_sorted.size]).astype(np.int64)
    unique_err = err_sums / duplicate_counts
    unique_stats = stats_signed_error(
        unique_err,
        np.ones(unique_err.shape, dtype=bool),
    )

    summary = {
        'river': river,
        'source_f060_dir': str(river_dir),
        'n_manifest_tiles': len(rows),
        'n_tiles_written': len(metric_rows),
        'n_tiles_skipped': len(skipped),
        'comparison_mask': 'Core_Loss_Mask_Pixel AND valid_GT AND valid_prediction',
        'gt_vrt_path': str(gt_vrt),
        'err_signed_vrt_path': str(err_signed_vrt),
        'err_abs_vrt_path': str(err_abs_vrt),
        'gt_tile_dir': str(gt_dir),
        'err_signed_tile_dir': str(err_signed_dir),
        'err_abs_tile_dir': str(err_abs_dir),
        'source_checkpoint': f060_summary.get('checkpoint'),
        'tile_footprint_weighted_n_pixels': total_n,
        'tile_footprint_weighted_rmse_m': float(math.sqrt(total_sse / total_n)) if total_n > 0 else None,
        'tile_footprint_weighted_mae_m': float(mae_num / total_n) if total_n > 0 else None,
        'tile_footprint_weighted_bias_m': float(bias_num / total_n) if total_n > 0 else None,
        'unique_geospatial_n_pixels': int(unique_stats['n_pixels']),
        'unique_geospatial_rmse_m': unique_stats['rmse_m'],
        'unique_geospatial_mae_m': unique_stats['mae_m'],
        'unique_geospatial_bias_m': unique_stats['bias_m'],
        'unique_geospatial_median_abs_error_m': unique_stats['median_abs_error_m'],
        'unique_geospatial_p90_abs_error_m': unique_stats['p90_abs_error_m'],
        'unique_geospatial_p95_abs_error_m': unique_stats['p95_abs_error_m'],
        'unique_geospatial_max_abs_error_m': unique_stats['max_abs_error_m'],
        'tile_footprint_to_unique_pixel_ratio': float(total_n / unique_stats['n_pixels']) if unique_stats['n_pixels'] > 0 else None,
        'max_overlap_multiplicity_in_metric_footprint': int(duplicate_counts.max()) if duplicate_counts.size else 0,
        'primary_metric_note': 'unique_geospatial_* counts each final overlap-averaged full-river pixel exactly once.',
        'legacy_comparison_note': 'tile_footprint_weighted_* preserves the earlier F020 convention and may double-count overlap pixels.',
    }
    (out_river / 'F027_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--f060_out_dir', required=True, help='F025 experiment output folder, e.g. holdout_CO_D001NoDataSafe')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--rivers', nargs='*', default=[])
    ap.add_argument('--nodata', type=float, default=-999999.0)
    ap.add_argument('--nodata_threshold', type=float, default=-9999.0)
    ap.add_argument('--progress_every', type=int, default=200)
    args = ap.parse_args()

    if not TIFFFILE_AVAILABLE:
        raise RuntimeError('tifffile is required.')

    f060_out = Path(args.f060_out_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    river_dirs = find_river_dirs(f060_out, args.rivers)

    (out / 'F027_args.json').write_text(json.dumps({
        **vars(args),
        'resolved_river_dirs': [str(d) for d in river_dirs],
    }, indent=2))

    print('[F027] Build GT/error VRTs plus legacy tile-footprint and exact unique-pixel metrics')
    print(f'[F027] f060_out_dir={f060_out}')
    print(f'[F027] output_dir={out}')
    print(f'[F027] rivers={[d.name for d in river_dirs]}')

    summaries = []
    for d in river_dirs:
        summaries.append(process_river(d, out, args))

    (out / 'F027_all_rivers_summary.json').write_text(json.dumps(summaries, indent=2))
    print('[DONE]', out)


if __name__ == '__main__':
    main()
