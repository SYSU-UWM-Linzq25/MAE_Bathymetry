#!/usr/bin/env python3
"""H049: quantify known-information percentage versus full-river reconstruction error.

One row is produced for every sampling point/tile and every configuration:

- strict mask + normalized objective
- strict mask + meter objective
- relaxed mask + normalized objective
- relaxed mask + meter objective

Known information is reported in two ways:

1. patch-known percentage: visible valid 16x16 patches / all valid 16x16 patches
2. pixel-known percentage: visible valid pixels / all valid pixels

The primary point-level error is measured on exact four-way common final pixels
for that sampling tile, so all four configurations are compared on the same
local pixels. Own-footprint metrics are also retained for diagnostics.

Predictions are read from the existing full-river averaged prediction tiles in
the F010/F060/F049/F044 manifests. The model is not rerun here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import H046_visualize_local_reaches_6panel as h046


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Analyze point-level known-information percentage versus "
            "full-river reconstruction error."
        ),
    )
    strict_results = Path(
        "/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/"
        "Downstream_Task_Bathy/Results"
    )
    relax_results = Path(
        "/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/"
        "Downstream_Task_Bathy_relax_HiddenMask/results"
    )
    parser.add_argument(
        "--strict_normalized_pred_root",
        type=Path,
        default=strict_results
        / "FullRiver_Predictions_F010_TileAvgVRT_D001NoDataSafe",
    )
    parser.add_argument(
        "--strict_meter_pred_root",
        type=Path,
        default=strict_results
        / "FullRiver_Predictions_F060_TileAvgVRT_D003MeterMAE_BaselineEval_D001NoDataSafe",
    )
    parser.add_argument(
        "--relaxed_normalized_pred_root",
        type=Path,
        default=relax_results
        / "FullRiver_Predictions_F049_NormalizedObjective_D001cAnyVisiblePatch",
    )
    parser.add_argument(
        "--relaxed_meter_pred_root",
        type=Path,
        default=relax_results
        / "FullRiver_Predictions_F044_MeterOnly_D001cAnyVisiblePatch",
    )
    parser.add_argument(
        "--relax_tile_base",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/"
            "Tiles_for_MAE_FullRiver_E001c_AnyVisiblePatch"
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument(
        "--min_common_pixels_per_point",
        type=int,
        default=100,
        help="Minimum four-way common pixels for correlation/scatter analyses.",
    )
    parser.add_argument("--bin_width_percent", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def block_all(mask: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Tile shape {mask.shape} is not divisible by patch_size={patch_size}"
        )
    return mask.reshape(
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    ).all(axis=(1, 3))


def block_any(mask: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Tile shape {mask.shape} is not divisible by patch_size={patch_size}"
        )
    return mask.reshape(
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    ).any(axis=(1, 3))


def stats(error: np.ndarray) -> Dict[str, float]:
    values = np.asarray(error, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "p90": float("nan"),
        }
    absolute = np.abs(values)
    return {
        "n": int(values.size),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(values).mean())),
        "bias": float(values.mean()),
        "p90": float(np.percentile(absolute, 90)),
    }


def rank_values(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(method="average").to_numpy(dtype=float)


def correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return float("nan"), float("nan")
    xv = x[valid].astype(float)
    yv = y[valid].astype(float)
    pearson = float(np.corrcoef(xv, yv)[0, 1])
    spearman = float(np.corrcoef(rank_values(xv), rank_values(yv))[0, 1])
    return pearson, spearman


def load_case(
    args: argparse.Namespace,
    case: Mapping[str, str],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, str]]],
    List[Dict[str, str]],
]:
    directories = h046.branch_dirs(args, case)
    maps: Dict[str, Dict[str, Dict[str, str]]] = {}
    audits: List[Dict[str, str]] = []
    for config, directory in directories.items():
        manifest, summary = h046.locate_manifest(directory)
        if config.startswith("relaxed_"):
            h046.validate_relaxed_summary(summary, config)
        rows, audit = h046.read_manifest_with_rebased_paths(
            manifest,
            directory,
            config,
        )
        maps[config] = {row["key"]: row for row in rows}
        audits.extend(audit)
        print(f"[{case['preset']}] {config}: {len(rows)} rows")
    return maps, audits


def point_rows_for_case(
    args: argparse.Namespace,
    case: Mapping[str, str],
    maps: Mapping[str, Mapping[str, Dict[str, str]]],
) -> List[Dict[str, Any]]:
    common_keys = set.intersection(
        *(set(maps[config]) for config in h046.CONFIG_ORDER)
    )
    reference_rows = [maps["relaxed_meter"][key] for key in common_keys]
    reference_rows.sort(key=lambda row: int(float(row["tile_id"])))

    line_mapping, line_source = h046.load_line_mapping(
        args.relax_tile_base,
        case["river"],
    )

    records: List[Dict[str, Any]] = []
    for index, reference in enumerate(reference_rows, start=1):
        key = reference["key"]
        tile_id = int(float(reference["tile_id"]))

        gt, _ = h046.read_tif(reference["tile_path"])
        gt = gt.astype(np.float32, copy=False)
        valid_gt = h046.valid_gt(gt)

        branch_arrays: Dict[str, Dict[str, np.ndarray]] = {}
        common4 = valid_gt.copy()
        for config in h046.CONFIG_ORDER:
            row = maps[config][key]
            pred, _ = h046.read_tif(row["avg_pred_tile_path"])
            hidden, _ = h046.read_tif(row["hidden_path"])
            core, _ = h046.read_tif(row["core_loss_path"])

            pred = pred.astype(np.float32, copy=False)
            hidden_mask = (
                np.isfinite(hidden)
                & (hidden.astype(np.float32, copy=False) > 0.5)
            )
            core_mask = (
                np.isfinite(core)
                & (core.astype(np.float32, copy=False) > 0.5)
            )
            final_mask = core_mask & valid_gt & h046.valid_pred(pred)
            branch_arrays[config] = {
                "pred": pred,
                "hidden": hidden_mask,
                "core": core_mask,
                "final": final_mask,
            }
            common4 &= final_mask

        for config in h046.CONFIG_ORDER:
            arrays = branch_arrays[config]
            config_row = maps[config][key]
            hidden_mask = arrays["hidden"]

            valid_patch = block_all(valid_gt, args.patch_size)
            hidden_patch = block_any(hidden_mask, args.patch_size)
            known_patch = valid_patch & (~hidden_patch)

            n_valid_patches = int(valid_patch.sum())
            n_known_patches = int(known_patch.sum())
            known_patch_fraction = (
                n_known_patches / n_valid_patches
                if n_valid_patches > 0
                else float("nan")
            )

            n_valid_pixels = int(valid_gt.sum())
            known_pixels = valid_gt & (~hidden_mask)
            n_known_pixels = int(known_pixels.sum())
            known_pixel_fraction = (
                n_known_pixels / n_valid_pixels
                if n_valid_pixels > 0
                else float("nan")
            )

            pred = arrays["pred"]
            own = arrays["final"]
            own_stats = stats(pred[own] - gt[own])
            common_stats = stats(pred[common4] - gt[common4])

            records.append(
                {
                    "preset": case["preset"],
                    "river": case["river"],
                    "river_label": case["label"],
                    "line_id": line_mapping.get(tile_id, "SEQUENTIAL"),
                    "line_mapping_source": line_source,
                    "key": key,
                    "tile_id": tile_id,
                    "configuration": config,
                    "configuration_label": h046.CONFIG_LABELS[config],
                    "mask_regime": (
                        "Strict" if config.startswith("strict_") else "Relaxed"
                    ),
                    "objective": (
                        "Normalized objective"
                        if config.endswith("normalized")
                        else "Meter objective"
                    ),
                    "valid_patch_count": n_valid_patches,
                    "known_patch_count": n_known_patches,
                    "known_patch_fraction": known_patch_fraction,
                    "known_patch_percent": known_patch_fraction * 100.0,
                    "hidden_patch_fraction": 1.0 - known_patch_fraction
                    if np.isfinite(known_patch_fraction)
                    else float("nan"),
                    "valid_pixel_count": n_valid_pixels,
                    "known_pixel_count": n_known_pixels,
                    "known_pixel_fraction": known_pixel_fraction,
                    "known_pixel_percent": known_pixel_fraction * 100.0,
                    "own_final_pixel_count": own_stats["n"],
                    "own_mae_m": own_stats["mae"],
                    "own_rmse_m": own_stats["rmse"],
                    "own_bias_m": own_stats["bias"],
                    "own_p90_abs_error_m": own_stats["p90"],
                    "fourway_common_pixel_count": common_stats["n"],
                    "common4_mae_m": common_stats["mae"],
                    "common4_rmse_m": common_stats["rmse"],
                    "common4_bias_m": common_stats["bias"],
                    "common4_p90_abs_error_m": common_stats["p90"],
                    "prediction_source": config_row["avg_pred_tile_path"],
                    "hidden_mask_source": config_row["hidden_path"],
                    "core_loss_mask_source": config_row["core_loss_path"],
                }
            )

        if index == 1 or index == len(reference_rows) or index % 250 == 0:
            print(
                f"  [{case['preset']}] {index}/{len(reference_rows)} sampling points"
            )
    return records


def binned_summary(
    frame: pd.DataFrame,
    x_column: str,
    bin_width: float,
) -> pd.DataFrame:
    edges = np.arange(0.0, 100.0 + bin_width, bin_width)
    if edges[-1] < 100.0:
        edges = np.r_[edges, 100.0]
    labels = [f"{edges[i]:.0f}-{edges[i + 1]:.0f}%" for i in range(len(edges) - 1)]
    work = frame.copy()
    work["known_information_bin"] = pd.cut(
        work[x_column],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    grouped = (
        work.groupby(
            ["configuration", "configuration_label", "river_label", "known_information_bin"],
            observed=True,
            dropna=True,
        )
        .agg(
            n_points=("key", "count"),
            mean_known_percent=(x_column, "mean"),
            median_common4_mae_m=("common4_mae_m", "median"),
            mean_common4_mae_m=("common4_mae_m", "mean"),
            median_common4_rmse_m=("common4_rmse_m", "median"),
            mean_common_pixels=("fourway_common_pixel_count", "mean"),
        )
        .reset_index()
    )
    grouped["known_information_definition"] = x_column
    return grouped


def make_scatter(
    frame: pd.DataFrame,
    x_column: str,
    x_label: str,
    output: Path,
    dpi: int,
    minimum_pixels: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
    river_labels = [case["label"] for case in h046.CASES]
    for ax, config in zip(axes.flat, h046.CONFIG_ORDER):
        subset = frame[
            (frame["configuration"] == config)
            & (frame["fourway_common_pixel_count"] >= minimum_pixels)
            & np.isfinite(frame[x_column])
            & np.isfinite(frame["common4_mae_m"])
        ].copy()

        for river_label in river_labels:
            river = subset[subset["river_label"] == river_label]
            if river.empty:
                continue
            ax.scatter(
                river[x_column],
                river["common4_mae_m"],
                s=12,
                alpha=0.45,
                label=river_label,
            )

        if not subset.empty:
            bin_width = 10.0
            centers = np.arange(5.0, 100.0, bin_width)
            medians = []
            usable_centers = []
            for center in centers:
                lower = center - bin_width / 2.0
                upper = center + bin_width / 2.0
                values = subset[
                    (subset[x_column] > lower)
                    & (subset[x_column] <= upper)
                ]["common4_mae_m"]
                if len(values) >= 5:
                    usable_centers.append(center)
                    medians.append(float(values.median()))
            if usable_centers:
                ax.plot(
                    usable_centers,
                    medians,
                    linewidth=2.0,
                    marker="o",
                    label="10%-bin median",
                )

        ax.set_title(h046.CONFIG_LABELS[config])
        ax.set_xlabel(x_label)
        ax.set_ylabel("Point-local MAE on four-way common pixels (m)")
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    fig.suptitle(
        "Known information versus local full-river reconstruction error\n"
        "Each point is one sampling tile; error uses exact four-way common final pixels"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    path_audits: List[Dict[str, str]] = []
    for case in h046.CASES:
        print("=" * 76)
        print(case["label"])
        maps, audit = load_case(args, case)
        path_audits.extend(audit)
        all_rows.extend(point_rows_for_case(args, case, maps))

    frame = pd.DataFrame(all_rows)
    if frame.empty:
        raise RuntimeError("No point-level records produced")

    point_csv = output / "H049_sampling_point_known_information_error.csv"
    frame.to_csv(point_csv, index=False)
    h046.write_csv(output / "H049_manifest_path_rebase_audit.csv", path_audits)

    eligible = frame[
        frame["fourway_common_pixel_count"] >= args.min_common_pixels_per_point
    ].copy()

    correlation_rows: List[Dict[str, Any]] = []
    for config in h046.CONFIG_ORDER:
        for river_label in ["All rivers", *[case["label"] for case in h046.CASES]]:
            subset = eligible[eligible["configuration"] == config]
            if river_label != "All rivers":
                subset = subset[subset["river_label"] == river_label]
            for x_column in ("known_patch_percent", "known_pixel_percent"):
                pearson, spearman = correlation(
                    subset[x_column].to_numpy(dtype=float),
                    subset["common4_mae_m"].to_numpy(dtype=float),
                )
                correlation_rows.append(
                    {
                        "configuration": config,
                        "configuration_label": h046.CONFIG_LABELS[config],
                        "river_label": river_label,
                        "known_information_definition": x_column,
                        "n_points": int(len(subset)),
                        "pearson_r": pearson,
                        "spearman_rho": spearman,
                        "mean_known_percent": float(subset[x_column].mean())
                        if len(subset)
                        else float("nan"),
                        "mean_common4_mae_m": float(subset["common4_mae_m"].mean())
                        if len(subset)
                        else float("nan"),
                    }
                )
    pd.DataFrame(correlation_rows).to_csv(
        output / "H049_known_information_error_correlations.csv",
        index=False,
    )

    binned = pd.concat(
        [
            binned_summary(eligible, "known_patch_percent", args.bin_width_percent),
            binned_summary(eligible, "known_pixel_percent", args.bin_width_percent),
        ],
        ignore_index=True,
    )
    binned.to_csv(
        output / "H049_known_information_error_binned_summary.csv",
        index=False,
    )

    make_scatter(
        frame,
        "known_patch_percent",
        "Known valid patches (%)",
        output / "H049_known_patch_percent_vs_common4_mae.png",
        args.dpi,
        args.min_common_pixels_per_point,
    )
    make_scatter(
        frame,
        "known_pixel_percent",
        "Known valid pixels (%)",
        output / "H049_known_pixel_percent_vs_common4_mae.png",
        args.dpi,
        args.min_common_pixels_per_point,
    )

    summary = {
        "analysis_unit": "one sampling point / one 336x336 tile",
        "prediction_source": (
            "existing overlap-averaged full-river prediction tiles from "
            "F010/F060/F049/F044; model inference is not rerun"
        ),
        "primary_known_information": (
            "visible valid 16x16 patches divided by all valid 16x16 patches"
        ),
        "secondary_known_information": (
            "visible valid pixels divided by all valid pixels"
        ),
        "primary_error_footprint": (
            "exact per-tile intersection of valid GT and final prediction pixels "
            "from all four configurations"
        ),
        "important_interpretation": (
            "Point-level rows use full-river averaged predictions. Nearby sampling "
            "points have overlapping support and are not statistically independent."
        ),
        "minimum_common_pixels_per_point": args.min_common_pixels_per_point,
        "n_rows": int(len(frame)),
        "n_eligible_rows": int(len(eligible)),
        "outputs": {
            "point_table": point_csv.name,
            "correlations": "H049_known_information_error_correlations.csv",
            "binned_summary": "H049_known_information_error_binned_summary.csv",
            "patch_scatter": "H049_known_patch_percent_vs_common4_mae.png",
            "pixel_scatter": "H049_known_pixel_percent_vs_common4_mae.png",
        },
    }
    (output / "H049_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("[DONE]", output)


if __name__ == "__main__":
    main()
