#!/usr/bin/env python3
"""Collect upstream and downstream MAE tile elevation statistics.

Upstream mode
-------------
Statistics are computed from ALL valid elevation pixels in every tile listed
in the supplied split files.  Per-tile normalization uses that tile's full
valid-pixel mean and population standard deviation (ddof=0).

Downstream mode
---------------
The script scans one MAE-v2 tile root containing:
  Train_tile/ + Hidden_Mask/ + Loss_Mask_Pixel/

Statistics are computed separately for:
  all     : every valid DEM pixel with a defined Hidden_Mask value;
  known   : valid pixels with Hidden_Mask == 0;
  masked  : valid pixels with Hidden_Mask == 1 (the model-hidden region);
  loss    : valid pixels with Loss_Mask_Pixel == 1 (supervised subset).

Downstream normalization is deliberately training-consistent: its center and
scale come from KNOWN pixels only.  The same known_mean and
max(known_std * std_scale, eps) are then used to normalize known, masked, and
loss pixels.

Main outputs
------------
  *_tile_stats.csv                 one row per tile
  *_group_summary.csv              pixel-pooled summary by state or river
  by_state/*.csv or by_river/*.csv one detailed CSV per group
  errors.csv                       unreadable or invalid items, when present
  run_config.json                  exact arguments and counts

The script does not run the MAE model and does not compute prediction error.
It only characterizes the elevation distributions presented to the model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


TILE_RE = re.compile(
    r"^Select_tile_(?:Basin_)?(?P<res>\d+)m_(?P<river>.+)_ID(?P<id>\d+)(?:_(?P<suffix>[^.]+))?\.tif$",
    re.IGNORECASE,
)

US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP",
}

RAW_STAT_NAMES = ("count", "min", "max", "mean", "std", "p01", "p05", "p50", "p95", "p99")


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def read_raster(path: Path) -> np.ndarray:
    """Read band 1 without changing values."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.read(1)
    except ImportError:
        import tifffile
        return tifffile.imread(path)


def is_valid_dem(a: np.ndarray, nodata: float, nodata_threshold: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    valid = np.isfinite(a) & (a > nodata_threshold)
    if math.isfinite(nodata):
        tol = max(1e-6, abs(nodata) * 1e-7)
        valid &= np.abs(a - nodata) > tol
    return valid


def is_defined_byte_mask(a: np.ndarray, mask_nodata: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    defined = np.isfinite(a)
    if math.isfinite(mask_nodata):
        tol = max(1e-6, abs(mask_nodata) * 1e-7)
        defined &= np.abs(a - mask_nodata) > tol
    return defined


def values_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "min": float("nan"), "max": float("nan"),
            "mean": float("nan"), "std": float("nan"),
            "p01": float("nan"), "p05": float("nan"),
            "p50": float("nan"), "p95": float("nan"),
            "p99": float("nan"),
        }
    q = np.percentile(values, [1, 5, 50, 95, 99])
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        # Population standard deviation, matching NumPy's default ddof=0.
        "std": float(values.std(ddof=0)),
        "p01": float(q[0]), "p05": float(q[1]), "p50": float(q[2]),
        "p95": float(q[3]), "p99": float(q[4]),
    }


def add_region_stats(
    row: Dict[str, Any],
    prefix: str,
    dem: np.ndarray,
    mask: np.ndarray,
    norm_mean: Optional[float],
    norm_scale: Optional[float],
) -> None:
    vals = np.asarray(dem, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    raw = values_stats(vals)
    for key, value in raw.items():
        row[f"{prefix}_raw_{key}"] = value

    if norm_mean is None or norm_scale is None or not math.isfinite(norm_scale) or norm_scale <= 0:
        norm = values_stats(np.asarray([], dtype=np.float64))
    else:
        norm = values_stats((vals - norm_mean) / norm_scale)
    for key, value in norm.items():
        row[f"{prefix}_norm_{key}"] = value


def parse_state(path: Path) -> str:
    """Infer a state code from directory components or the file name."""
    candidates: List[str] = []
    for part in path.parts:
        token = part.upper()
        if token in US_STATE_CODES:
            candidates.append(token)
        for m in re.finditer(r"(?:^|[^A-Z])([A-Z]{2})(?:[^A-Z]|$)", token):
            code = m.group(1)
            if code in US_STATE_CODES:
                candidates.append(code)
    return candidates[-1] if candidates else "UNKNOWN"


def parse_river_and_id(path: Path) -> Tuple[str, Optional[int], str]:
    m = TILE_RE.match(path.name)
    if not m:
        return "UNKNOWN", None, ""
    return m.group("river"), int(m.group("id")), f"{int(m.group('res'))}m"


def resolve_list_entry(text: str, data_root: Path, list_path: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(text.strip())))
    if p.is_absolute():
        return p
    candidates = [data_root / p, list_path.parent / p, Path.cwd() / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def read_split_list(label: str, list_path: Path, data_root: Path) -> List[Tuple[str, Path]]:
    if not list_path.is_file():
        raise FileNotFoundError(f"Split list does not exist: {list_path}")
    out: List[Tuple[str, Path]] = []
    with list_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            out.append((label, resolve_list_entry(text, data_root, list_path)))
    return out


def parse_labeled_list(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Use LABEL=/path/to/list.txt")
    label, value = spec.split("=", 1)
    label = label.strip()
    value = value.strip()
    if not label or not value:
        raise argparse.ArgumentTypeError("Use LABEL=/path/to/list.txt")
    return label, Path(value)


def upstream_worker(task: Tuple[str, str, float, float, float, float]) -> Dict[str, Any]:
    split, path_text, nodata, nodata_threshold, std_scale, eps = task
    path = Path(path_text)
    row: Dict[str, Any] = {
        "status": "OK", "split": split, "state": parse_state(path),
        "tile_path": str(path), "tile_name": path.name,
    }
    try:
        dem = read_raster(path).astype(np.float64, copy=False)
        valid = is_valid_dem(dem, nodata, nodata_threshold)
        n_total = int(dem.size)
        n_valid = int(valid.sum())
        row["height"] = int(dem.shape[-2]) if dem.ndim >= 2 else 1
        row["width"] = int(dem.shape[-1]) if dem.ndim >= 2 else int(dem.size)
        row["total_pixel_count"] = n_total
        row["valid_pixel_count"] = n_valid
        row["valid_pixel_fraction"] = n_valid / n_total if n_total else float("nan")
        if n_valid < 2:
            raise ValueError(f"Fewer than 2 valid pixels: {n_valid}")

        raw = values_stats(dem[valid])
        norm_mean = raw["mean"]
        norm_scale = max(float(raw["std"]) * std_scale, eps)
        row["normalization_source"] = "all_valid"
        row["normalization_mean_m"] = norm_mean
        row["normalization_raw_std_m"] = raw["std"]
        row["normalization_std_scale"] = std_scale
        row["normalization_denominator_m"] = norm_scale
        add_region_stats(row, "all", dem, valid, norm_mean, norm_scale)
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def downstream_worker(task: Tuple[str, str, str, str, float, float, float, float, float, float]) -> Dict[str, Any]:
    key, dem_text, hidden_text, loss_text, nodata, nodata_threshold, mask_nodata, mask_threshold, std_scale, eps = task
    dem_path = Path(dem_text)
    hidden_path = Path(hidden_text)
    loss_path = Path(loss_text)
    river, tile_id, resolution = parse_river_and_id(dem_path)
    row: Dict[str, Any] = {
        "status": "OK", "key": key, "river": river, "tile_id": tile_id,
        "resolution": resolution, "tile_path": str(dem_path),
        "hidden_mask_path": str(hidden_path), "loss_mask_path": str(loss_path),
        "tile_name": dem_path.name,
    }
    try:
        dem = read_raster(dem_path).astype(np.float64, copy=False)
        hidden = read_raster(hidden_path).astype(np.float64, copy=False)
        loss = read_raster(loss_path).astype(np.float64, copy=False)
        if dem.shape != hidden.shape or dem.shape != loss.shape:
            raise ValueError(f"Shape mismatch: DEM={dem.shape}, hidden={hidden.shape}, loss={loss.shape}")

        valid = is_valid_dem(dem, nodata, nodata_threshold)
        hidden_defined = is_defined_byte_mask(hidden, mask_nodata)
        loss_defined = is_defined_byte_mask(loss, mask_nodata)
        hidden_positive = hidden_defined & (hidden >= mask_threshold)
        loss_positive = loss_defined & (loss >= mask_threshold)

        # Exclude undefined Hidden_Mask pixels from all known/masked comparisons.
        all_mask = valid & hidden_defined
        known_mask = all_mask & ~hidden_positive
        masked_mask = all_mask & hidden_positive
        loss_mask = valid & loss_positive

        n_total = int(dem.size)
        row["height"] = int(dem.shape[-2]) if dem.ndim >= 2 else 1
        row["width"] = int(dem.shape[-1]) if dem.ndim >= 2 else int(dem.size)
        row["total_pixel_count"] = n_total
        row["valid_dem_pixel_count"] = int(valid.sum())
        row["hidden_mask_defined_count"] = int(hidden_defined.sum())
        row["loss_mask_defined_count"] = int(loss_defined.sum())
        row["known_pixel_count"] = int(known_mask.sum())
        row["masked_pixel_count"] = int(masked_mask.sum())
        row["loss_pixel_count"] = int(loss_mask.sum())
        row["known_pixel_fraction_of_defined_valid"] = float(known_mask.sum() / all_mask.sum()) if all_mask.any() else float("nan")
        row["masked_pixel_fraction_of_defined_valid"] = float(masked_mask.sum() / all_mask.sum()) if all_mask.any() else float("nan")
        row["loss_pixel_fraction_of_valid"] = float(loss_mask.sum() / valid.sum()) if valid.any() else float("nan")
        row["masked_loss_overlap_count"] = int((masked_mask & loss_mask).sum())
        row["masked_nonloss_count"] = int((masked_mask & ~loss_mask).sum())
        row["loss_outside_masked_count"] = int((loss_mask & ~masked_mask).sum())
        row["valid_dem_with_undefined_hidden_count"] = int((valid & ~hidden_defined).sum())

        known_raw = values_stats(dem[known_mask])
        if known_raw["count"] >= 2:
            norm_mask = known_mask
            norm_source = "known"
            norm_raw = known_raw
        else:
            # This mirrors the defensive fallback used by the dataset helper.
            norm_mask = all_mask
            norm_source = "fallback_all_valid_defined"
            norm_raw = values_stats(dem[norm_mask])
        if norm_raw["count"] < 2:
            raise ValueError(
                f"Fewer than 2 pixels for normalization: known={known_raw['count']}, all={norm_raw['count']}"
            )

        norm_mean = float(norm_raw["mean"])
        norm_scale = max(float(norm_raw["std"]) * std_scale, eps)
        row["normalization_source"] = norm_source
        row["normalization_mean_m"] = norm_mean
        row["normalization_raw_std_m"] = norm_raw["std"]
        row["normalization_std_scale"] = std_scale
        row["normalization_denominator_m"] = norm_scale

        add_region_stats(row, "all", dem, all_mask, norm_mean, norm_scale)
        add_region_stats(row, "known", dem, known_mask, norm_mean, norm_scale)
        add_region_stats(row, "masked", dem, masked_mask, norm_mean, norm_scale)
        add_region_stats(row, "loss", dem, loss_mask, norm_mean, norm_scale)
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_group_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "UNKNOWN"


def finite_number(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def pooled_region_summary(rows: Sequence[Mapping[str, Any]], region: str) -> Dict[str, Any]:
    components: List[Tuple[int, float, float, float, float]] = []
    tile_stds: List[float] = []
    tile_means: List[float] = []
    norm_components: List[Tuple[int, float, float, float, float]] = []
    norm_tile_stds: List[float] = []
    norm_tile_means: List[float] = []

    for row in rows:
        n = finite_number(row.get(f"{region}_raw_count"))
        mn = finite_number(row.get(f"{region}_raw_min"))
        mx = finite_number(row.get(f"{region}_raw_max"))
        mean = finite_number(row.get(f"{region}_raw_mean"))
        std = finite_number(row.get(f"{region}_raw_std"))
        if n is not None and n > 0 and None not in (mn, mx, mean, std):
            components.append((int(n), float(mn), float(mx), float(mean), float(std)))
            tile_stds.append(float(std))
            tile_means.append(float(mean))

        nn = finite_number(row.get(f"{region}_norm_count"))
        nmn = finite_number(row.get(f"{region}_norm_min"))
        nmx = finite_number(row.get(f"{region}_norm_max"))
        nmean = finite_number(row.get(f"{region}_norm_mean"))
        nstd = finite_number(row.get(f"{region}_norm_std"))
        if nn is not None and nn > 0 and None not in (nmn, nmx, nmean, nstd):
            norm_components.append((int(nn), float(nmn), float(nmx), float(nmean), float(nstd)))
            norm_tile_stds.append(float(nstd))
            norm_tile_means.append(float(nmean))

    def pool(parts: Sequence[Tuple[int, float, float, float, float]]) -> Dict[str, Any]:
        if not parts:
            return {"count": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
        total = sum(p[0] for p in parts)
        pooled_mean = sum(p[0] * p[3] for p in parts) / total
        pooled_var = sum(p[0] * (p[4] ** 2 + (p[3] - pooled_mean) ** 2) for p in parts) / total
        return {
            "count": int(total),
            "min": min(p[1] for p in parts),
            "max": max(p[2] for p in parts),
            "mean": float(pooled_mean),
            "std": float(math.sqrt(max(0.0, pooled_var))),
        }

    raw_pool = pool(components)
    norm_pool = pool(norm_components)
    result: Dict[str, Any] = {
        "region": region,
        "n_tiles_with_pixels": len(components),
        "raw_pixel_count": raw_pool["count"],
        "raw_pooled_min": raw_pool["min"],
        "raw_pooled_max": raw_pool["max"],
        "raw_pooled_mean": raw_pool["mean"],
        "raw_pooled_std": raw_pool["std"],
        "raw_tile_mean_median": float(np.median(tile_means)) if tile_means else float("nan"),
        "raw_tile_std_min": min(tile_stds) if tile_stds else float("nan"),
        "raw_tile_std_mean": float(np.mean(tile_stds)) if tile_stds else float("nan"),
        "raw_tile_std_median": float(np.median(tile_stds)) if tile_stds else float("nan"),
        "raw_tile_std_max": max(tile_stds) if tile_stds else float("nan"),
        "norm_pixel_count": norm_pool["count"],
        "norm_pooled_min": norm_pool["min"],
        "norm_pooled_max": norm_pool["max"],
        "norm_pooled_mean": norm_pool["mean"],
        "norm_pooled_std": norm_pool["std"],
        "norm_tile_mean_median": float(np.median(norm_tile_means)) if norm_tile_means else float("nan"),
        "norm_tile_std_min": min(norm_tile_stds) if norm_tile_stds else float("nan"),
        "norm_tile_std_mean": float(np.mean(norm_tile_stds)) if norm_tile_stds else float("nan"),
        "norm_tile_std_median": float(np.median(norm_tile_stds)) if norm_tile_stds else float("nan"),
        "norm_tile_std_max": max(norm_tile_stds) if norm_tile_stds else float("nan"),
    }
    return result


def write_group_outputs(
    good_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    group_field: str,
    group_folder: str,
    regions: Sequence[str],
    summary_name: str,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in good_rows:
        group = str(row.get(group_field, "UNKNOWN") or "UNKNOWN")
        groups.setdefault(group, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for group in sorted(groups):
        rows = groups[group]
        write_csv(output_dir / group_folder / f"{safe_group_filename(group)}_tile_stats.csv", rows)
        for region in regions:
            summary = pooled_region_summary(rows, region)
            summary = {
                group_field: group,
                "n_tiles_total": len(rows),
                **summary,
            }
            summary_rows.append(summary)
    write_csv(output_dir / summary_name, summary_rows)
    return summary_rows


def key_from_path(path: Path) -> str:
    river, tile_id, resolution = parse_river_and_id(path)
    if river == "UNKNOWN" or tile_id is None:
        raise ValueError(f"Unrecognized MAE-v2 filename: {path.name}")
    return f"{resolution}_{river}_ID{tile_id}"


def index_folder(folder: Path) -> Tuple[Dict[str, Path], List[Dict[str, Any]]]:
    index: Dict[str, Path] = {}
    problems: List[Dict[str, Any]] = []
    if not folder.is_dir():
        raise FileNotFoundError(f"Required folder not found: {folder}")
    for path in sorted(folder.rglob("*.tif")):
        try:
            key = key_from_path(path)
        except Exception as exc:
            problems.append({"status": "UNRECOGNIZED_FILENAME", "folder": str(folder), "path": str(path), "error": str(exc)})
            continue
        if key in index:
            problems.append({
                "status": "DUPLICATE_KEY", "folder": str(folder), "key": key,
                "path": str(path), "other_path": str(index[key]),
            })
        else:
            index[key] = path.resolve()
    return index, problems


def run_tasks(worker, tasks: Sequence[Tuple[Any, ...]], workers: int, progress_every: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    total = len(tasks)
    if workers <= 1:
        iterator = map(worker, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(worker, tasks, chunksize=max(1, min(64, total // max(1, workers * 8))))
    try:
        for i, row in enumerate(iterator, 1):
            results.append(row)
            if i == 1 or i == total or i % progress_every == 0:
                print(f"[PROGRESS] {i}/{total}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)
    return results


def upstream_main(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items: List[Tuple[str, Path]] = []
    for label, list_path in args.list:
        items.extend(read_split_list(label, list_path.resolve(), args.data_root.resolve()))

    duplicate_rows: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    unique_items: List[Tuple[str, Path]] = []
    for split, path in items:
        key = str(path)
        if key in seen:
            duplicate_rows.append({"status": "DUPLICATE_PATH", "tile_path": key, "first_split": seen[key], "duplicate_split": split})
        else:
            seen[key] = split
            unique_items.append((split, path))

    tasks = [
        (split, str(path), args.nodata, args.nodata_threshold, args.std_scale, args.eps)
        for split, path in unique_items
    ]
    print(f"[UPSTREAM] list entries={len(items)}, unique tiles={len(tasks)}, duplicates={len(duplicate_rows)}")
    rows = run_tasks(upstream_worker, tasks, args.workers, args.progress_every)
    good = [r for r in rows if r.get("status") == "OK"]
    errors = duplicate_rows + [r for r in rows if r.get("status") != "OK"]

    write_csv(output_dir / "upstream_tile_stats.csv", good)
    write_group_outputs(
        good, output_dir, group_field="state", group_folder="by_state",
        regions=("all",), summary_name="upstream_state_summary.csv",
    )
    # Also preserve split-specific summaries.
    write_group_outputs(
        good, output_dir, group_field="split", group_folder="by_split",
        regions=("all",), summary_name="upstream_split_summary.csv",
    )
    if errors:
        write_csv(output_dir / "errors.csv", errors)

    config = {
        "mode": "upstream",
        "data_root": str(args.data_root.resolve()),
        "lists": [{"label": label, "path": str(path.resolve())} for label, path in args.list],
        "nodata": args.nodata,
        "nodata_threshold": args.nodata_threshold,
        "std_scale": args.std_scale,
        "eps": args.eps,
        "std_ddof": 0,
        "input_entries": len(items),
        "unique_tiles": len(tasks),
        "successful_tiles": len(good),
        "error_count": len(errors),
        "definition": "all valid pixels per upstream tile",
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))
    return 2 if errors and args.fail_on_error else 0


def downstream_main(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_root = args.tile_root.resolve()

    dem_index, p1 = index_folder(tile_root / args.dem_folder)
    hidden_index, p2 = index_folder(tile_root / args.hidden_folder)
    loss_index, p3 = index_folder(tile_root / args.loss_folder)
    pairing_problems: List[Dict[str, Any]] = p1 + p2 + p3

    all_keys = sorted(set(dem_index) | set(hidden_index) | set(loss_index))
    matched_keys: List[str] = []
    for key in all_keys:
        missing = []
        if key not in dem_index:
            missing.append("DEM")
        if key not in hidden_index:
            missing.append("Hidden_Mask")
        if key not in loss_index:
            missing.append("Loss_Mask_Pixel")
        if missing:
            pairing_problems.append({
                "status": "MISSING_PAIR", "key": key, "missing": ";".join(missing),
                "dem_path": str(dem_index.get(key, "")),
                "hidden_path": str(hidden_index.get(key, "")),
                "loss_path": str(loss_index.get(key, "")),
            })
        else:
            matched_keys.append(key)

    tasks = [
        (
            key, str(dem_index[key]), str(hidden_index[key]), str(loss_index[key]),
            args.nodata, args.nodata_threshold, args.mask_nodata,
            args.mask_threshold, args.std_scale, args.eps,
        )
        for key in matched_keys
    ]
    print(
        f"[DOWNSTREAM] DEM={len(dem_index)}, hidden={len(hidden_index)}, loss={len(loss_index)}, "
        f"matched={len(tasks)}, pairing problems={len(pairing_problems)}"
    )
    rows = run_tasks(downstream_worker, tasks, args.workers, args.progress_every)
    good = [r for r in rows if r.get("status") == "OK"]
    errors = pairing_problems + [r for r in rows if r.get("status") != "OK"]

    write_csv(output_dir / "downstream_tile_stats.csv", good)
    write_group_outputs(
        good, output_dir, group_field="river", group_folder="by_river",
        regions=("all", "known", "masked", "loss"),
        summary_name="downstream_river_summary.csv",
    )
    if errors:
        write_csv(output_dir / "errors.csv", errors)

    config = {
        "mode": "downstream",
        "tile_root": str(tile_root),
        "dem_folder": args.dem_folder,
        "hidden_folder": args.hidden_folder,
        "loss_folder": args.loss_folder,
        "nodata": args.nodata,
        "nodata_threshold": args.nodata_threshold,
        "mask_nodata": args.mask_nodata,
        "mask_threshold": args.mask_threshold,
        "std_scale": args.std_scale,
        "eps": args.eps,
        "std_ddof": 0,
        "matched_tiles": len(tasks),
        "successful_tiles": len(good),
        "error_count": len(errors),
        "known_definition": "valid DEM AND defined Hidden_Mask AND Hidden_Mask < threshold",
        "masked_definition": "valid DEM AND defined Hidden_Mask AND Hidden_Mask >= threshold",
        "loss_definition": "valid DEM AND defined Loss_Mask_Pixel AND Loss_Mask_Pixel >= threshold",
        "normalization_definition": "known-pixel mean and max(known-pixel std * std_scale, eps)",
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))
    return 2 if errors and args.fail_on_error else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    up = sub.add_parser("upstream", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    up.add_argument("--data-root", type=Path, required=True, help="Root used to resolve relative paths in split lists")
    up.add_argument(
        "--list", type=parse_labeled_list, action="append", required=True,
        metavar="LABEL=FILE", help="Repeat for train, val, holdout, etc.",
    )
    up.add_argument("--output-dir", type=Path, required=True)
    up.add_argument("--nodata", type=float, default=-9999.0)
    up.add_argument("--nodata-threshold", type=float, default=-9999.0)
    up.add_argument("--std-scale", type=float, default=1.0)
    up.add_argument("--eps", type=float, default=1e-3)
    up.add_argument("--workers", type=int, default=8)
    up.add_argument("--progress-every", type=int, default=250)
    up.add_argument("--fail-on-error", action="store_true")
    up.set_defaults(func=upstream_main)

    down = sub.add_parser("downstream", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    down.add_argument("--tile-root", type=Path, required=True)
    down.add_argument("--output-dir", type=Path, required=True)
    down.add_argument("--dem-folder", default="Train_tile")
    down.add_argument("--hidden-folder", default="Hidden_Mask")
    down.add_argument("--loss-folder", default="Loss_Mask_Pixel")
    down.add_argument("--nodata", type=float, default=-999999.0)
    down.add_argument("--nodata-threshold", type=float, default=-9999.0)
    down.add_argument("--mask-nodata", type=float, default=255.0)
    down.add_argument("--mask-threshold", type=float, default=0.5)
    down.add_argument("--std-scale", type=float, default=1.5)
    down.add_argument("--eps", type=float, default=1e-3)
    down.add_argument("--workers", type=int, default=8)
    down.add_argument("--progress-every", type=int, default=250)
    down.add_argument("--fail-on-error", action="store_true")
    down.set_defaults(func=downstream_main)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.std_scale <= 0:
        parser.error("--std-scale must be > 0")
    if args.eps <= 0:
        parser.error("--eps must be > 0")
    try:
        return int(args.func(args))
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
