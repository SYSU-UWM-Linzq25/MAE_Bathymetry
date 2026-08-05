#!/usr/bin/env python3
"""Create a compact publication figure from the H054 selected-reach bundle.

The input directory must contain the three ``*_representative_reach.npz``
files exported by H054_export_AGU_selected_reach_data.py.  The script uses only
those portable files; it does not depend on the original HPC directory tree.

Design choices
--------------
* Ground truth and prediction use the same robust elevation limits per river.
* Ground truth, prediction, and absolute error use the exact same comparison
  pixels, rotated array shape, crop, axis limits, and footprint outline.
* Absolute error uses one shared 0--1 m scale. Values above the limit are
  saturated and indicated by the colorbar extension triangle.
* Missing pixels are light gray; near-zero error is warm white, so low error is
  visible rather than being confused with missing data.
* Reaches are rotated for compact display without changing pixel size.  The
  rotation uses nearest-neighbor sampling and is applied identically to all
  three panels in a row.
* Local MAE/RMSE are recomputed from exactly the pixels shown in each row.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Keep Matplotlib's cache in a writable location on HPC/login nodes.
_mpl_cache = Path(tempfile.gettempdir()) / "mae_bathymetry_matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class Case:
    preset: str
    filename: str
    river: str
    location: str


CASES = (
    Case("CA", "CA_representative_reach.npz", "Klamath River", "California"),
    Case(
        "CO",
        "CO_representative_reach.npz",
        "Upper Colorado River",
        "Colorado",
    ),
    Case(
        "OR",
        "Santiam_representative_reach.npz",
        "Santiam River",
        "Oregon",
    ),
)


@dataclass
class Reach:
    case: Case
    gt: np.ndarray
    prediction: np.ndarray
    error: np.ndarray
    footprint: np.ndarray
    resolution_m: float
    mae_m: float
    rmse_m: float
    bias_m: float
    fullriver_mae_m: float
    fullriver_rmse_m: float
    rotation_deg: float
    vmin: float
    vmax: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot representative bathymetry reconstruction results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the three H054 representative-reach NPZ files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-stem",
        default="AGU26_representative_bathymetry_reconstruction",
    )
    parser.add_argument(
        "--title",
        default="Representative bathymetry reconstruction in unseen rivers",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=900,
        help="Raster resolution used for both PNG and LZW-compressed TIFF.",
    )
    parser.add_argument("--error-max-m", type=float, default=1.0)
    parser.add_argument(
        "--display-tilt-deg",
        type=float,
        default=-18.0,
        help="Clockwise display tilt after orienting each reach vertically.",
    )
    parser.add_argument("--crop-padding-px", type=int, default=8)
    parser.add_argument(
        "--elevation-percentiles",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
    )
    return parser.parse_args()


def crop_to_mask(
    arrays: Iterable[np.ndarray], mask: np.ndarray, padding: int
) -> tuple[list[np.ndarray], np.ndarray]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("The displayed footprint is empty.")
    row0 = max(0, int(rows.min()) - padding)
    row1 = min(mask.shape[0], int(rows.max()) + padding + 1)
    col0 = max(0, int(cols.min()) - padding)
    col1 = min(mask.shape[1], int(cols.max()) + padding + 1)
    window = np.s_[row0:row1, col0:col1]
    return [np.asarray(a)[window] for a in arrays], mask[window]


def choose_vertical_rotation(mask: np.ndarray) -> int:
    """Choose a 90-degree rotation whose footprint is taller than wide."""
    rows, cols = np.where(mask)
    height = int(rows.max() - rows.min() + 1)
    width = int(cols.max() - cols.min() + 1)
    return 1 if width > height else 0


def rotate_for_display(
    array: np.ndarray,
    k90: int,
    tilt_deg: float,
    *,
    is_mask: bool = False,
) -> np.ndarray:
    rotated = np.rot90(array, k=k90)
    if abs(tilt_deg) < 1e-9:
        return rotated
    if is_mask:
        work = rotated.astype(np.uint8)
        result = ndimage.rotate(
            work,
            tilt_deg,
            reshape=True,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        return result.astype(bool)
    return ndimage.rotate(
        rotated,
        tilt_deg,
        reshape=True,
        order=0,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def robust_limits(arrays: Iterable[np.ndarray], low: float, high: float) -> tuple[float, float]:
    values = np.concatenate(
        [np.asarray(a)[np.isfinite(a)].astype(np.float64) for a in arrays]
    )
    if values.size == 0:
        raise ValueError("No finite elevation values are available.")
    vmin, vmax = np.percentile(values, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def load_reach(
    data_dir: Path,
    case: Case,
    padding: int,
    tilt_deg: float,
    elevation_percentiles: tuple[float, float],
) -> Reach:
    path = data_dir / case.filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected-reach file: {path}")

    with np.load(path, allow_pickle=False) as bundle:
        gt = bundle["gt"].astype(np.float64)
        prediction = bundle["prediction"].astype(np.float64)
        final_mask = bundle["final_mask"].astype(bool)
        resolution_m = float(bundle["resolution_m"])
        fullriver_mae_m = float(bundle["fullriver_mae_m"])
        fullriver_rmse_m = float(bundle["fullriver_rmse_m"])

    # One comparison mask is applied to all three displayed products.  This is
    # intentionally stricter than plotting each source with its own finite-data
    # mask: every visible pixel must have both GT and a prediction.
    comparison_mask = final_mask & np.isfinite(gt) & np.isfinite(prediction)
    gt = np.where(comparison_mask, gt, np.nan)
    prediction = np.where(comparison_mask, prediction, np.nan)
    (gt, prediction), comparison_mask = crop_to_mask(
        [gt, prediction], comparison_mask, padding
    )

    valid = comparison_mask & np.isfinite(gt) & np.isfinite(prediction)
    residual = prediction[valid] - gt[valid]
    mae_m = float(np.mean(np.abs(residual)))
    rmse_m = float(np.sqrt(np.mean(np.square(residual))))
    bias_m = float(np.mean(residual))

    k90 = choose_vertical_rotation(comparison_mask)
    gt = rotate_for_display(gt, k90, tilt_deg)
    prediction = rotate_for_display(prediction, k90, tilt_deg)
    comparison_mask = rotate_for_display(
        comparison_mask, k90, tilt_deg, is_mask=True
    )

    # Nearest-neighbour rotation of a NaN-masked raster and of a binary mask can
    # differ by an edge pixel because they are sampled independently.  Rebuild
    # the common mask after rotation, then apply it to every panel.  These
    # assertions make a future regression fail loudly instead of producing a
    # visually questionable figure.
    comparison_mask &= np.isfinite(gt) & np.isfinite(prediction)
    gt = np.where(comparison_mask, gt, np.nan)
    prediction = np.where(comparison_mask, prediction, np.nan)
    error = np.where(comparison_mask, np.abs(prediction - gt), np.nan)
    gt_pixels = np.isfinite(gt)
    prediction_pixels = np.isfinite(prediction)
    error_pixels = np.isfinite(error)
    if not (
        np.array_equal(gt_pixels, comparison_mask)
        and np.array_equal(prediction_pixels, comparison_mask)
        and np.array_equal(error_pixels, comparison_mask)
    ):
        raise RuntimeError(
            f"Display masks are not identical after rotation for {case.preset}."
        )

    low, high = elevation_percentiles
    vmin, vmax = robust_limits([gt, prediction], low, high)

    return Reach(
        case=case,
        gt=gt,
        prediction=prediction,
        error=error,
        footprint=comparison_mask,
        resolution_m=resolution_m,
        mae_m=mae_m,
        rmse_m=rmse_m,
        bias_m=bias_m,
        fullriver_mae_m=fullriver_mae_m,
        fullriver_rmse_m=fullriver_rmse_m,
        rotation_deg=float(k90 * 90 + tilt_deg),
        vmin=vmin,
        vmax=vmax,
    )


def make_colormaps() -> tuple[colors.Colormap, colors.Colormap]:
    elevation = plt.get_cmap("viridis").copy()
    elevation.set_bad("#E9EEF3")
    error = colors.LinearSegmentedColormap.from_list(
        "bathymetry_error",
        ("#FFF3D6", "#FEE8C8", "#FDBB84", "#FC8D59", "#D7301F", "#7F0000"),
    )
    error.set_bad("#E9EEF3")
    return elevation, error


def style_map_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#E9EEF3")
    for spine in ax.spines.values():
        spine.set_color("#D9E0E7")
        spine.set_linewidth(0.8)


def add_footprint_outline(ax: plt.Axes, footprint: np.ndarray) -> None:
    ax.contour(
        footprint.astype(float),
        levels=[0.5],
        colors=["#586674"],
        linewidths=0.35,
        alpha=0.58,
    )


def render_figure(
    reaches: list[Reach], args: argparse.Namespace
) -> tuple[Path, Path, Path, Path, Path]:
    elevation_cmap, error_cmap = make_colormaps()
    error_norm = colors.Normalize(vmin=0.0, vmax=float(args.error_max_m), clip=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titleweight": "semibold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(9.2, 9.5), facecolor="white")
    outer = fig.add_gridspec(
        nrows=3,
        ncols=3,
        width_ratios=(1.55, 5.75, 0.18),
        height_ratios=(1, 1, 1),
        left=0.045,
        right=0.965,
        bottom=0.055,
        top=0.875,
        wspace=0.085,
        hspace=0.16,
    )

    fig.suptitle(
        args.title,
        x=0.045,
        y=0.965,
        ha="left",
        va="top",
        fontsize=18.5,
        fontweight="bold",
        color="#17212B",
    )
    fig.add_artist(
        plt.Line2D(
            [0.045, 0.965],
            [0.915, 0.915],
            transform=fig.transFigure,
            color="#173F6D",
            lw=1.6,
        )
    )

    shared_error_cax = fig.add_subplot(outer[:, 2])
    row_audit: list[dict[str, float | str | int]] = []

    for row, reach in enumerate(reaches):
        label_ax = fig.add_subplot(outer[row, 0])
        label_ax.axis("off")
        label_ax.text(
            0.0,
            0.90,
            reach.case.preset,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="#173F6D",
            bbox=dict(
                boxstyle="round,pad=0.34,rounding_size=0.16",
                facecolor="#E7F0F8",
                edgecolor="none",
            ),
        )
        label_ax.text(
            0.0,
            0.70,
            reach.case.river,
            ha="left",
            va="top",
            fontsize=11.5,
            fontweight="bold",
            color="#17212B",
            wrap=True,
        )
        label_ax.text(
            0.0,
            0.49,
            reach.case.location,
            ha="left",
            va="top",
            fontsize=8.2,
            color="#66727E",
        )
        label_ax.text(
            0.0,
            0.29,
            f"Reach MAE  {reach.mae_m:.3f} m\nReach RMSE {reach.rmse_m:.3f} m",
            ha="left",
            va="top",
            fontsize=8.5,
            linespacing=1.45,
            color="#263746",
        )

        row_grid = outer[row, 1].subgridspec(
            nrows=2,
            ncols=3,
            height_ratios=(1.0, 0.065),
            width_ratios=(1, 1, 1),
            hspace=0.09,
            wspace=0.08,
        )
        gt_ax = fig.add_subplot(row_grid[0, 0])
        pred_ax = fig.add_subplot(row_grid[0, 1], sharex=gt_ax, sharey=gt_ax)
        error_ax = fig.add_subplot(row_grid[0, 2], sharex=gt_ax, sharey=gt_ax)
        elev_cax = fig.add_subplot(row_grid[1, 0:2])

        if row == 0:
            for axis, title in zip(
                (gt_ax, pred_ax, error_ax),
                ("GROUND TRUTH", "PREDICTION", "ABSOLUTE ERROR"),
            ):
                axis.set_title(
                    title,
                    fontsize=9.2,
                    color="#173F6D",
                    pad=10,
                    loc="center",
                )

        elevation_norm = colors.Normalize(vmin=reach.vmin, vmax=reach.vmax)
        gt_ax.imshow(
            reach.gt,
            cmap=elevation_cmap,
            norm=elevation_norm,
            interpolation="nearest",
            aspect="equal",
        )
        pred_ax.imshow(
            reach.prediction,
            cmap=elevation_cmap,
            norm=elevation_norm,
            interpolation="nearest",
            aspect="equal",
        )
        error_ax.imshow(
            reach.error,
            cmap=error_cmap,
            norm=error_norm,
            interpolation="nearest",
            aspect="equal",
        )

        # Lock all three panels to the same array extent.  The shared axes and
        # explicit limits prevent later Matplotlib changes (for example, an
        # added artist) from autoscaling only one of the panels.
        height_px, width_px = reach.footprint.shape
        for axis in (gt_ax, pred_ax, error_ax):
            axis.set_xlim(-0.5, width_px - 0.5)
            axis.set_ylim(height_px - 0.5, -0.5)
            style_map_axis(axis)
            add_footprint_outline(axis, reach.footprint)

        elev_bar = fig.colorbar(
            ScalarMappable(norm=elevation_norm, cmap=elevation_cmap),
            cax=elev_cax,
            orientation="horizontal",
        )
        elev_bar.set_label("Elevation / bathymetry (m)", fontsize=7.4, labelpad=1.5)
        elev_bar.ax.tick_params(labelsize=7.0, length=2.3, pad=1.5)
        elev_bar.outline.set_linewidth(0.55)
        elev_bar.outline.set_edgecolor("#7C8792")

        row_audit.append(
            {
                "preset": reach.case.preset,
                "river": reach.case.river,
                "resolution_m": reach.resolution_m,
                "reach_mae_m": reach.mae_m,
                "reach_rmse_m": reach.rmse_m,
                "reach_bias_m": reach.bias_m,
                "fullriver_mae_m": reach.fullriver_mae_m,
                "fullriver_rmse_m": reach.fullriver_rmse_m,
                "display_rotation_deg_ccw": reach.rotation_deg,
                "elevation_vmin_m": reach.vmin,
                "elevation_vmax_m": reach.vmax,
                "error_display_max_m": float(args.error_max_m),
                "display_array_height_px": int(height_px),
                "display_array_width_px": int(width_px),
                "gt_displayed_pixels": int(np.isfinite(reach.gt).sum()),
                "prediction_displayed_pixels": int(
                    np.isfinite(reach.prediction).sum()
                ),
                "error_displayed_pixels": int(np.isfinite(reach.error).sum()),
                "identical_display_masks": bool(
                    np.array_equal(np.isfinite(reach.gt), np.isfinite(reach.prediction))
                    and np.array_equal(np.isfinite(reach.gt), np.isfinite(reach.error))
                ),
                "identical_axis_limits": True,
            }
        )

    error_bar = fig.colorbar(
        ScalarMappable(norm=error_norm, cmap=error_cmap),
        cax=shared_error_cax,
        orientation="vertical",
        extend="max",
    )
    error_bar.set_ticks([0, 0.1, 0.25, 0.5, 0.75, float(args.error_max_m)])
    error_bar.ax.set_yticklabels(["0", "0.10", "0.25", "0.50", "0.75", f"{args.error_max_m:g}"])
    error_bar.ax.tick_params(labelsize=7.6, length=2.5, pad=2)
    error_bar.outline.set_linewidth(0.6)
    error_bar.outline.set_edgecolor("#7C8792")
    shared_error_cax.set_title("m", fontsize=8.0, color="#55616C", pad=6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.output_stem}.png"
    tiff_path = args.output_dir / f"{args.output_stem}.tif"
    pdf_path = args.output_dir / f"{args.output_stem}.pdf"
    svg_path = args.output_dir / f"{args.output_stem}.svg"
    audit_path = args.output_dir / f"{args.output_stem}_audit.json"

    fig.savefig(png_path, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # Convert the already-rendered high-resolution PNG to TIFF. Saving a large
    # compressed TIFF directly through some Matplotlib/Pillow combinations can
    # leave an invalid zero IFD offset; this two-step path is portable and
    # guarantees that the TIFF and PNG contain the exact same plotted pixels.
    with Image.open(png_path) as png_image:
        png_image.convert("RGB").save(
            tiff_path,
            format="TIFF",
            compression="tiff_lzw",
            dpi=(args.dpi, args.dpi),
        )

    audit = {
        "title": args.title,
        "data_dir": str(args.data_dir.resolve()),
        "output_stem": args.output_stem,
        "elevation_percentiles": list(args.elevation_percentiles),
        "error_display_max_m": float(args.error_max_m),
        "display_tilt_deg": float(args.display_tilt_deg),
        "rows": row_audit,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return png_path, tiff_path, pdf_path, svg_path, audit_path


def main() -> None:
    args = parse_args()
    reaches = [
        load_reach(
            args.data_dir,
            case,
            padding=args.crop_padding_px,
            tilt_deg=args.display_tilt_deg,
            elevation_percentiles=tuple(args.elevation_percentiles),
        )
        for case in CASES
    ]
    outputs = render_figure(reaches, args)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
