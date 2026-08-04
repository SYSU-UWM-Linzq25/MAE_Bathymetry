#!/usr/bin/env python3
"""Export the three selected relaxed-mask reaches into one compact bundle.

This script reuses the existing H052 figure code and H052 geospatial helper.
It exports only the arrays needed to rebuild the final AGU figure offline:

- processed GT mosaic
- formal F044 overlap-averaged prediction mosaic
- final prediction/evaluation mask
- sampling-center coordinates
- resolution and local/full-river metrics
- selected reach metadata and source tile records

No model inference is run.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def import_module(path: Path, name: str) -> ModuleType:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
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


def parse_args() -> argparse.Namespace:
    root = Path("/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography")
    project = root / "Downstream_Task_Bathy_relax_HiddenMask"
    results = project / "results"
    script = project / "script"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--figure_script",
        type=Path,
        default=script / "H052_make_AGU_relaxed_mask_representative_figure.py",
    )
    parser.add_argument(
        "--helper_script",
        type=Path,
        default=script / "H052_AGU_geospatial_utils.py",
    )
    parser.add_argument(
        "--selected_csv",
        type=Path,
        default=(
            results
            / "H052_AGU_RelaxedMask_RepresentativeFigure"
            / "H052_selected_representative_reaches.csv"
        ),
    )
    parser.add_argument(
        "--prediction_root",
        type=Path,
        default=(
            results
            / "FullRiver_Predictions_F044_MeterOnly_D001cAnyVisiblePatch"
        ),
    )
    parser.add_argument(
        "--error_root",
        type=Path,
        default=(
            results
            / "FullRiver_GT_Error_F046_MeterOnly_D001cAnyVisiblePatch"
        ),
    )
    parser.add_argument(
        "--tile_base",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "Tiles_for_MAE_FullRiver_E001c_AnyVisiblePatch"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=results / "H054_AGU_SelectedReach_DataBundle",
    )
    parser.add_argument(
        "--zip_name",
        default="H054_AGU_SelectedReach_DataBundle.zip",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path in (
        args.figure_script,
        args.helper_script,
        args.selected_csv,
        args.prediction_root,
        args.error_root,
        args.tile_base,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.output_dir.exists():
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        elif any(args.output_dir.iterdir()):
            raise RuntimeError(
                f"Output exists: {args.output_dir}. Use --overwrite."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure = import_module(args.figure_script, "h052_export_source")
    helper = import_module(args.helper_script, "h052_export_helper")
    tile_root = helper.resolve_processed_tile_root(args.tile_base)

    selections = figure.select_reaches_manually(args.selected_csv)
    all_summary: Dict[str, Any] = {
        "prediction_root": str(args.prediction_root),
        "error_root": str(args.error_root),
        "tile_root": str(tile_root),
        "selected_csv": str(args.selected_csv),
        "rivers": {},
    }

    source_records: List[Dict[str, Any]] = []

    for case in figure.CASES:
        preset = case["preset"]
        selection = selections[preset]

        river_dir, manifest_rows = figure.load_prediction_manifest(
            helper,
            args.prediction_root,
            case,
        )
        selected_rows = figure.rows_for_selection(
            manifest_rows,
            selection,
        )
        reach = figure.assemble_reach(
            helper,
            selected_rows,
            tile_root,
        )
        full = figure.read_fullriver_metrics(
            args.error_root,
            case,
        )

        npz_path = args.output_dir / f"{preset}_representative_reach.npz"
        np.savez_compressed(
            npz_path,
            gt=reach.gt.astype(np.float32),
            prediction=reach.prediction.astype(np.float32),
            final_mask=reach.final_mask.astype(np.uint8),
            centers=np.asarray(reach.centers, dtype=np.float32),
            resolution_m=np.asarray(reach.resolution_m, dtype=np.float64),
            local_mae_m=np.asarray(reach.local_mae_m, dtype=np.float64),
            local_rmse_m=np.asarray(reach.local_rmse_m, dtype=np.float64),
            local_bias_m=np.asarray(reach.local_bias_m, dtype=np.float64),
            n_final_pixels=np.asarray(reach.n_final_pixels, dtype=np.int64),
            preset=np.asarray(preset),
            river=np.asarray(case["river"]),
            river_label=np.asarray(case["label"]),
            segment_id=np.asarray(selection.segment_id),
            line_id=np.asarray(selection.line_id),
            first_point_id=np.asarray(selection.first_point_id, dtype=np.int64),
            last_point_id=np.asarray(selection.last_point_id, dtype=np.int64),
            fullriver_mae_m=np.asarray(full["mae_m"], dtype=np.float64),
            fullriver_rmse_m=np.asarray(full["rmse_m"], dtype=np.float64),
            fullriver_bias_m=np.asarray(full["bias_m"], dtype=np.float64),
            fullriver_n_pixels=np.asarray(full["n_pixels"], dtype=np.int64),
        )

        row_csv = args.output_dir / f"{preset}_source_manifest_rows.csv"
        write_rows(row_csv, selected_rows)

        for row in selected_rows:
            source_records.append(
                {
                    "preset": preset,
                    "river": case["river"],
                    "segment_id": selection.segment_id,
                    **row,
                }
            )

        all_summary["rivers"][preset] = {
            "river": case["river"],
            "river_label": case["label"],
            "segment_id": selection.segment_id,
            "line_id": selection.line_id,
            "first_point_id": selection.first_point_id,
            "last_point_id": selection.last_point_id,
            "n_sampling_points": selection.n_sampling_points,
            "npz": npz_path.name,
            "source_manifest_rows": row_csv.name,
            "prediction_river_dir": str(river_dir),
            "local_mae_m": reach.local_mae_m,
            "local_rmse_m": reach.local_rmse_m,
            "local_bias_m": reach.local_bias_m,
            "n_final_pixels": reach.n_final_pixels,
            "fullriver_mae_m": full["mae_m"],
            "fullriver_rmse_m": full["rmse_m"],
            "fullriver_bias_m": full["bias_m"],
            "fullriver_n_pixels": full["n_pixels"],
            "fullriver_summary_path": full["summary_path"],
        }

    shutil.copy2(
        args.selected_csv,
        args.output_dir / "H052_selected_representative_reaches.csv",
    )
    write_rows(
        args.output_dir / "H054_all_selected_source_rows.csv",
        source_records,
    )
    (args.output_dir / "H054_bundle_summary.json").write_text(
        json.dumps(all_summary, indent=2)
    )

    zip_path = args.output_dir.parent / args.zip_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(args.output_dir.parent))

    print("[DONE]", args.output_dir)
    print("[ZIP]", zip_path)


if __name__ == "__main__":
    main()
