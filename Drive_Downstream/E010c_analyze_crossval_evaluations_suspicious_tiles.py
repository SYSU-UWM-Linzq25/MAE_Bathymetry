#!/usr/bin/env python3
# NUMBER-ALIGNED NAME: E010c_analyze_crossval_evaluations_suspicious_tiles.py
# ORIGINAL BACKUP NAME: E040_analyze_crossval_evaluations_suspicious_tiles.py
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
"""Summarize cross-validation E030 evaluations and extract suspicious tiles.

Inputs:
  cross_validation/evaluation/holdout_*/eval_E030_predictionOnly_coreBox/
    summary.json
    per_tile_metrics.csv
    worst_by_<metric>.csv

Outputs:
  cross_validation/evaluation/_summary/
    crossval_holdout_summary.csv
    crossval_holdout_summary_ranked.csv
    suspicious_tiles_topN_by_fold.csv
    suspicious_tiles_global_top.csv
    suspicious_tiles_for_A012.txt
    suspicious_tiles_by_fold/*.txt
    missing_or_incomplete_evaluations.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TILE_RE = re.compile(r"^Select_tile_Basin_(?P<res>\d+)m_(?P<river>.+)_ID(?P<tile_id>\d+)\.tif$", re.I)


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_tile_name(file_name: str) -> Tuple[str, Optional[int]]:
    m = TILE_RE.match(Path(file_name).name)
    if not m:
        return "", None
    return m.group("river"), int(m.group("tile_id"))


def nested_get(d: Dict[str, Any], path: Sequence[str]) -> Optional[float]:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return safe_float(cur)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def load_training_curve(run_dir: Path) -> Dict[str, Any]:
    """Best-effort parse of log.txt/history.csv, if present."""
    result: Dict[str, Any] = {
        "train_curve_found": 0,
        "curve_n_epochs": "",
        "curve_initial_val_rmse_m_mask": "",
        "curve_best_val_rmse_m_mask": "",
        "curve_last_val_rmse_m_mask": "",
        "curve_val_rmse_improve_pct": "",
        "curve_best_epoch": "",
    }
    candidates = [run_dir / "history.csv", run_dir / "log.txt"]
    rows: List[Dict[str, Any]] = []

    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".csv":
            try:
                rows = read_csv(path)
                break
            except Exception:
                pass
        else:
            parsed = []
            for line in path.read_text(errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    parsed.append(obj)
            if parsed:
                rows = parsed
                break

    if not rows:
        return result

    # Find a reasonable val RMSE key.
    possible_keys = [
        "val_rmse_m_mask",
        "test_rmse_m_mask",
        "rmse_m_mask",
        "val_rmse",
        "test_rmse",
    ]
    key = None
    for k in possible_keys:
        if any(safe_float(r.get(k)) is not None for r in rows):
            key = k
            break
    if key is None:
        return result

    curve = []
    for idx, r in enumerate(rows):
        v = safe_float(r.get(key))
        if v is None:
            continue
        ep = safe_int(r.get("epoch"))
        curve.append((ep if ep is not None else idx, v))

    if not curve:
        return result

    first_epoch, first_val = curve[0]
    best_epoch, best_val = min(curve, key=lambda x: x[1])
    last_epoch, last_val = curve[-1]
    improve_pct = (first_val - best_val) / max(abs(first_val), 1e-12) * 100.0

    result.update({
        "train_curve_found": 1,
        "curve_n_epochs": len(curve),
        "curve_initial_val_rmse_m_mask": first_val,
        "curve_best_val_rmse_m_mask": best_val,
        "curve_last_val_rmse_m_mask": last_val,
        "curve_val_rmse_improve_pct": improve_pct,
        "curve_best_epoch": best_epoch,
    })
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval_root",
        default=(
            "/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/"
            "Downstream_Task_Bathy/cross_validation/evaluation"
        ),
    )
    ap.add_argument("--metric", default="rmse_m_core_exact_pixel")
    ap.add_argument("--top_n_per_fold", type=int, default=50)
    ap.add_argument("--global_top_n", type=int, default=300)
    ap.add_argument(
        "--skip_folds",
        default="",
        help="Space-separated fold names to skip. Default is empty, so all folds are included.",
    )
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    out = eval_root / "_summary"
    by_fold_dir = out / "suspicious_tiles_by_fold"
    out.mkdir(parents=True, exist_ok=True)
    by_fold_dir.mkdir(parents=True, exist_ok=True)

    skip = set(args.skip_folds.split()) if args.skip_folds else set()
    metric = args.metric

    fold_rows: List[Dict[str, Any]] = []
    all_tile_rows: List[Dict[str, Any]] = []
    suspicious_by_fold: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    fold_dirs = sorted(eval_root.glob("holdout_*"))
    for fold_dir in fold_dirs:
        fold = fold_dir.name.replace("holdout_", "", 1)
        if fold in skip:
            continue
        eval_dir = fold_dir / "eval_E030_predictionOnly_coreBox"
        summary_path = eval_dir / "summary.json"
        metrics_path = eval_dir / "per_tile_metrics.csv"
        run_dir_txt = eval_dir / "run_dir.txt"

        if not summary_path.is_file() or not metrics_path.is_file():
            missing.append({
                "fold": fold,
                "fold_dir": str(fold_dir),
                "summary_exists": int(summary_path.is_file()),
                "per_tile_metrics_exists": int(metrics_path.is_file()),
            })
            continue

        summary = json.loads(summary_path.read_text())
        rows = read_csv(metrics_path)

        metric_values = [safe_float(r.get(metric)) for r in rows]
        metric_values = [v for v in metric_values if v is not None]
        if not metric_values:
            missing.append({
                "fold": fold,
                "fold_dir": str(fold_dir),
                "summary_exists": 1,
                "per_tile_metrics_exists": 1,
                "problem": f"No finite {metric}",
            })
            continue

        run_dir = Path(run_dir_txt.read_text().strip()) if run_dir_txt.is_file() else Path("")
        curve = load_training_curve(run_dir) if str(run_dir) else {}

        fold_row: Dict[str, Any] = {
            "fold": fold,
            "n_tiles": len(rows),
            "eval_dir": str(eval_dir),
            "run_dir": str(run_dir),
            "metric": metric,
            "tile_mean": sum(metric_values) / len(metric_values),
            "tile_median": percentile(metric_values, 50),
            "tile_p75": percentile(metric_values, 75),
            "tile_p90": percentile(metric_values, 90),
            "tile_p95": percentile(metric_values, 95),
            "tile_max": max(metric_values),
            "global_core_exact_rmse": nested_get(summary, ["global_pixel_weighted_rmse_m", "core_exact_final_mask_pixel"]),
            "global_outer_exact_rmse": nested_get(summary, ["global_pixel_weighted_rmse_m", "outer_exact_final_mask_pixel"]),
            "global_full_exact_rmse": nested_get(summary, ["global_pixel_weighted_rmse_m", "full_exact_final_mask_pixel"]),
            "global_core_patch_rmse": nested_get(summary, ["global_pixel_weighted_rmse_m", "core_patch"]),
        }
        fold_row.update(curve)
        fold_rows.append(fold_row)

        # Enrich all tile rows.
        for r in rows:
            file_name = r.get("file") or Path(r.get("path", "")).name
            river, tile_id = parse_tile_name(file_name)
            rr: Dict[str, Any] = {
                "fold": fold,
                "river": river,
                "tile_id": tile_id if tile_id is not None else "",
                "file": file_name,
                "path": r.get("path", ""),
                "mask_path": r.get("mask_path", ""),
            }
            for k, v in r.items():
                if k not in rr:
                    fv = safe_float(v)
                    rr[k] = fv if fv is not None else v
            all_tile_rows.append(rr)

        finite_rows = [r for r in all_tile_rows if r.get("fold") == fold and safe_float(r.get(metric)) is not None]
        finite_rows.sort(key=lambda r: float(r[metric]), reverse=True)
        top_rows = finite_rows[: args.top_n_per_fold]
        suspicious_by_fold.extend(top_rows)

        txt_path = by_fold_dir / f"holdout_{fold}_top{args.top_n_per_fold}_for_A012.txt"
        with txt_path.open("w") as f:
            for r in top_rows:
                f.write(f"{r['file']}\n")

    # Fold-level suspicion ranking.
    if fold_rows:
        global_vals = [safe_float(r.get("global_core_exact_rmse")) for r in fold_rows]
        global_vals = [v for v in global_vals if v is not None]
        p75_global = percentile(global_vals, 75) if global_vals else None
        median_global = percentile(global_vals, 50) if global_vals else None

        p90_vals = [safe_float(r.get("tile_p90")) for r in fold_rows]
        p90_vals = [v for v in p90_vals if v is not None]
        p75_p90 = percentile(p90_vals, 75) if p90_vals else None

        for r in fold_rows:
            flags = []
            g = safe_float(r.get("global_core_exact_rmse"))
            p90 = safe_float(r.get("tile_p90"))
            if p75_global is not None and g is not None and g >= p75_global:
                flags.append("HIGH_GLOBAL_CORE_RMSE_TOP25")
            if median_global is not None and g is not None and g >= 2.0 * median_global:
                flags.append("GLOBAL_CORE_RMSE_GT_2X_MEDIAN")
            if p75_p90 is not None and p90 is not None and p90 >= p75_p90:
                flags.append("HIGH_TILE_P90_TOP25")
            imp = safe_float(r.get("curve_val_rmse_improve_pct"))
            if imp is not None and imp < 10:
                flags.append("LOW_VAL_CURVE_IMPROVEMENT_LT10PCT")
            r["suspicion_flags"] = ";".join(flags)
            r["n_suspicion_flags"] = len(flags)

    ranked_folds = sorted(
        fold_rows,
        key=lambda r: (
            -int(r.get("n_suspicion_flags", 0)),
            -(safe_float(r.get("global_core_exact_rmse")) or -1),
            -(safe_float(r.get("tile_p90")) or -1),
        ),
    )

    global_tiles = [r for r in all_tile_rows if safe_float(r.get(metric)) is not None]
    global_tiles.sort(key=lambda r: float(r[metric]), reverse=True)
    global_top = global_tiles[: args.global_top_n]

    # Unique suspicious file list for A012.
    seen = set()
    unique_files = []
    for rows in (suspicious_by_fold, global_top):
        for r in rows:
            fn = r.get("file")
            if fn and fn not in seen:
                seen.add(fn)
                unique_files.append(fn)

    write_csv(out / "crossval_holdout_summary.csv", fold_rows)
    write_csv(out / "crossval_holdout_summary_ranked.csv", ranked_folds)
    write_csv(out / f"suspicious_tiles_top{args.top_n_per_fold}_by_fold.csv", suspicious_by_fold)
    write_csv(out / f"suspicious_tiles_global_top{args.global_top_n}.csv", global_top)
    write_csv(out / "missing_or_incomplete_evaluations.csv", missing)

    with (out / "suspicious_tiles_for_A012.txt").open("w") as f:
        for fn in unique_files:
            f.write(f"{fn}\n")

    # Human-readable quick report.
    report = out / "README_suspicious_analysis.txt"
    with report.open("w") as f:
        f.write("Cross-validation evaluation summary\n")
        f.write("===================================\n\n")
        f.write(f"Metric used for tile ranking: {metric}\n")
        f.write(f"Evaluated folds: {len(fold_rows)}\n")
        f.write(f"Missing/incomplete folds: {len(missing)}\n\n")
        f.write("Most suspicious folds first:\n")
        for r in ranked_folds:
            f.write(
                f"  {r['fold']}: flags={r.get('suspicion_flags','') or 'none'}, "
                f"global_core={r.get('global_core_exact_rmse')}, "
                f"tile_p90={r.get('tile_p90')}, tile_max={r.get('tile_max')}\n"
            )
        f.write("\nKey outputs:\n")
        f.write("  crossval_holdout_summary_ranked.csv\n")
        f.write(f"  suspicious_tiles_top{args.top_n_per_fold}_by_fold.csv\n")
        f.write(f"  suspicious_tiles_global_top{args.global_top_n}.csv\n")
        f.write("  suspicious_tiles_for_A012.txt\n")

    print("=== Cross-validation suspicious analysis done ===")
    print(f"eval_root = {eval_root}")
    print(f"summary   = {out}")
    print(f"folds     = {len(fold_rows)}")
    print(f"tiles     = {len(all_tile_rows)}")
    print(f"A012 list = {out / 'suspicious_tiles_for_A012.txt'}")
    print("Top suspicious folds:")
    for r in ranked_folds[:10]:
        print(f"  {r['fold']}: flags={r.get('suspicion_flags','') or 'none'}, global_core={r.get('global_core_exact_rmse')}, p90={r.get('tile_p90')}")


if __name__ == "__main__":
    main()
