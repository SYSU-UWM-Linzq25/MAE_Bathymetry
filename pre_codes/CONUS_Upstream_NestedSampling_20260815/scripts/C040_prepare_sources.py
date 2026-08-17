#!/usr/bin/env python3
"""Extract selected TNM products and create EPSG:5070, 1 m prepared rasters or Warped VRTs."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from osgeo import gdal

from conus_common import NODATA, TARGET_CRS, write_csv


LOG = logging.getLogger("prepare-sources")
gdal.UseExceptions()


INDEX_FIELDS = (
    "download_key", "source_path", "prepared_path", "format", "width", "height",
    "min_x", "min_y", "max_x", "max_y", "pixel_x", "pixel_y", "nodata",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download-manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--format", choices=("VRT", "GTiff"), default="VRT")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


def read_download_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or ())
        required = {"download_key", "url", "local_relpath"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                f"Download manifest must be a tab-separated TSV with fields {sorted(required)}; "
                f"missing={missing}, found={sorted(fieldnames)}: {manifest_path}"
            )
        rows = []
        for line_number, row in enumerate(reader, start=2):
            clean = {
                key: (value.rstrip("\r") if isinstance(value, str) else value)
                for key, value in row.items()
            }
            if not any(clean.get(name, "") for name in required):
                continue
            empty = sorted(name for name in required if not clean.get(name))
            if empty:
                raise ValueError(f"Empty required fields {empty} at {manifest_path}:{line_number}")
            rows.append(clean)
    if not rows:
        raise ValueError(f"Download manifest contains no data rows: {manifest_path}")
    return rows


def safe_extract_zip(source: Path, target: Path) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member.is_dir() or member_path.suffix.lower() not in (".tif", ".tiff"):
                continue
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe path in ZIP: {member.filename}")
            dst = target / member_path.name
            if not dst.exists() or dst.stat().st_size != member.file_size:
                with archive.open(member) as src, dst.open("wb") as out:
                    shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
            output.append(dst)
    return output


def source_tiffs(download: Path, extract_dir: Path) -> list[Path]:
    lower = download.name.lower()
    if lower.endswith(".zip"):
        return safe_extract_zip(download, extract_dir)
    if lower.endswith((".tif", ".tiff")):
        return [download]
    # TNM URLs occasionally omit/alter the suffix; inspect as ZIP before giving up.
    if zipfile.is_zipfile(download):
        return safe_extract_zip(download, extract_dir)
    ds = gdal.Open(str(download))
    if ds is not None:
        ds = None
        return [download]
    raise RuntimeError(f"Downloaded item is neither a readable raster nor ZIP: {download}")


def raster_index(download_key: str, source: Path, prepared: Path, output_format: str) -> dict:
    ds = gdal.Open(str(prepared))
    if ds is None:
        raise RuntimeError(f"Cannot open prepared raster: {prepared}")
    gt = ds.GetGeoTransform()
    if abs(gt[2]) > 1e-9 or abs(gt[4]) > 1e-9:
        raise RuntimeError(f"Prepared raster is rotated: {prepared}")
    min_x = gt[0]
    max_y = gt[3]
    max_x = min_x + ds.RasterXSize * gt[1]
    min_y = max_y + ds.RasterYSize * gt[5]
    nd = ds.GetRasterBand(1).GetNoDataValue()
    row = {
        "download_key": download_key,
        "source_path": str(source.resolve()),
        "prepared_path": str(prepared.resolve()),
        "format": output_format,
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "pixel_x": gt[1],
        "pixel_y": abs(gt[5]),
        "nodata": nd if nd is not None else "",
    }
    ds = None
    return row


def prepare_one(task: tuple[dict, str, str, str, bool]) -> tuple[list[dict], str | None]:
    row, data_root_text, out_dir_text, output_format, overwrite = task
    data_root = Path(data_root_text)
    out_dir = Path(out_dir_text)
    key = row["download_key"]
    download = data_root / row["local_relpath"]
    if not download.exists():
        return [], f"{key}: missing download {download}"
    try:
        tiffs = source_tiffs(download, out_dir / "extracted" / key)
        if not tiffs:
            raise RuntimeError("ZIP contains no GeoTIFF")
        results: list[dict] = []
        for idx, source in enumerate(tiffs, start=1):
            stem = f"{key}_{idx:02d}_epsg5070_1m"
            suffix = ".vrt" if output_format == "VRT" else ".tif"
            prepared = out_dir / "prepared" / key[:7] / f"{stem}{suffix}"
            prepared.parent.mkdir(parents=True, exist_ok=True)
            if overwrite and prepared.exists():
                prepared.unlink()
            if not prepared.exists():
                options = gdal.WarpOptions(
                    format=output_format,
                    dstSRS=TARGET_CRS,
                    xRes=1.0,
                    yRes=1.0,
                    targetAlignedPixels=True,
                    resampleAlg="bilinear",
                    dstNodata=NODATA,
                    multithread=True,
                    creationOptions=(
                        ["TILED=YES", "COMPRESS=ZSTD", "PREDICTOR=3", "BIGTIFF=IF_SAFER"]
                        if output_format == "GTiff" else []
                    ),
                    # Multiple products are prepared concurrently; cap per-warp
                    # threads to avoid workers each claiming the whole node.
                    warpOptions=["NUM_THREADS=2"],
                )
                warped = gdal.Warp(str(prepared), str(source), options=options)
                if warped is None:
                    raise RuntimeError(f"gdal.Warp returned None for {source}")
                warped.FlushCache()
                warped = None
            results.append(raster_index(key, source, prepared, output_format))
        return results, None
    except Exception as exc:  # Worker returns an auditable error row instead of hiding it.
        return [], f"{key}: {type(exc).__name__}: {exc}"


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    manifest = read_download_manifest(args.download_manifest)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (row, str(Path(args.data_root).resolve()), str(out_dir), args.format, args.overwrite)
        for row in manifest
    ]
    results: list[dict] = []
    errors: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(prepare_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            rows, error = future.result()
            results.extend(rows)
            if error:
                errors.append({"error": error})
                LOG.error(error)
            if index % 25 == 0 or index == len(futures):
                LOG.info("prepared jobs=%d/%d rasters=%d errors=%d", index, len(futures), len(results), len(errors))

    results.sort(key=lambda r: (r["download_key"], r["prepared_path"]))
    write_csv(out_dir / "source_index.csv", results, INDEX_FIELDS)
    write_csv(out_dir / "prepare_errors.csv", errors, ("error",))
    if errors:
        LOG.error("Preparation incomplete: %d errors", len(errors))
        return 3
    if not results:
        LOG.error("No prepared rasters were produced")
        return 4
    LOG.info("Preparation complete: %d rasters", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
