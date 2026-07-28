#!/usr/bin/env python3
# NUMBER-ALIGNED NEW FAMILY COPY: E025_evaluate_NormThenMeter_core_pixel_metrics_overlayvis_20260727.py
# TEMPLATE SOURCE: E030_evaluate_MeterOnly_core_pixel_metrics_overlayvis_20260713.py
# Evaluation mathematics are intentionally identical to the MeterOnly evaluator for a fair model comparison.
# NUMBER-ALIGNED NAME: E030_evaluate_MeterOnly_core_pixel_metrics_overlayvis_20260713.py
# ORIGINAL BACKUP NAME: E025_v4_meterMAE_evaluate_core_pixel_metrics_overlayvis_20260713.py
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
"""E025: evaluate NormThenMeter dual-mask checkpoints with training-consistent core pixel metrics.

This script is designed for the MAE v2 data structure:
  Train_tile + Hidden_Mask + Loss_Mask_Pixel

It answers the current diagnosis question:
  * why normalized loss can be small while meter-RMSE is large;
  * whether high meter-RMSE is driven by high tile_std_safe;
  * which rivers/tiles dominate the train-vs-val discrepancy;
  * what the true core pixel metric is when Loss_Mask_Pixel is intersected
    with the core patch region.

Output files:
  per_tile_metrics.csv
  per_river_summary.csv
  summary.json
  worst_by_rmse_m_core_loss_pixel.csv
  worst_by_tile_std_safe.csv
  worst_by_norm_to_meter_scale.csv
  worst_by_rmse_norm_core_loss_pixel.csv
  visuals_worst_by_<rank_metric>/
  visuals_median_by_<rank_metric>/
  visuals_best_by_<rank_metric>/
    quicklook_core_and_full_loss_pixels.png
    overlay_input_bathy_hidden_mask.png
    overlay_input_bathy_final_loss_mask.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


TILE_RE = re.compile(
    r"^Select_tile_(?:Basin_)?(?P<res>\d+)m_(?P<river>.+)_ID(?P<id>\d+)(?:_[^.]+)?\.tif$",
    re.IGNORECASE,
)


def add_code_path(code_dir: str) -> None:
    p = str(Path(code_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def meta_item(meta: Dict[str, Any], key: str, i: int, default=None):
    if key not in meta:
        return default
    v = meta[key]
    if torch.is_tensor(v):
        item = v[i]
        return item.item() if item.numel() == 1 else item.detach().cpu().numpy()
    if isinstance(v, (list, tuple)):
        return v[i]
    return v


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def parse_river(path: str) -> str:
    name = Path(path).name
    m = TILE_RE.match(name)
    return m.group("river") if m else ""


def expand_patch_mask(model, patch_mask: torch.Tensor, in_chans: int = 1) -> torch.Tensor:
    p = int(model.patch_embed.patch_size[0])
    expanded = patch_mask.float().unsqueeze(-1).repeat(1, 1, p * p * in_chans)
    return model.unpatchify(expanded)[:, :1]


def masked_stats(err: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    values = np.asarray(err, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0, "sse": 0.0, "mse": float("nan"),
            "rmse": float("nan"), "mae": float("nan"), "bias": float("nan"),
            "max_abs": float("nan"), "p95_abs": float("nan"),
        }
    abs_values = np.abs(values)
    sse = float(np.square(values).sum(dtype=np.float64))
    return {
        "count": int(values.size),
        "sse": sse,
        "mse": float(sse / values.size),
        "rmse": float(np.sqrt(sse / values.size)),
        "mae": float(abs_values.mean()),
        "bias": float(values.mean()),
        "max_abs": float(abs_values.max()),
        "p95_abs": float(np.percentile(abs_values, 95)),
    }


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


def summary_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    a = np.asarray([v for v in values if safe_float(v) is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0, "mean": None, "median": None, "p75": None, "p90": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def pixel_weighted_rmse(rows: Sequence[Dict[str, Any]], sse_key: str, count_key: str) -> Optional[float]:
    sse = 0.0
    count = 0
    for r in rows:
        ss = safe_float(r.get(sse_key))
        cc = safe_float(r.get(count_key))
        if ss is not None and cc is not None and cc > 0:
            sse += ss
            count += int(cc)
    return float(math.sqrt(sse / count)) if count > 0 else None


def pixel_weighted_mse(rows: Sequence[Dict[str, Any]], sse_key: str, count_key: str) -> Optional[float]:
    """Pixel-weighted MSE reconstructed from per-tile sums of squared errors."""
    sse = 0.0
    count = 0
    for r in rows:
        ss = safe_float(r.get(sse_key))
        cc = safe_float(r.get(count_key))
        if ss is not None and cc is not None and cc > 0:
            sse += ss
            count += int(cc)
    return float(sse / count) if count > 0 else None


def pixel_weighted_mean(
    rows: Sequence[Dict[str, Any]],
    value_key: str,
    count_key: str,
) -> Optional[float]:
    """Pixel-weighted mean for a per-tile MAE, bias, or another tile mean."""
    numerator = 0.0
    count = 0
    for r in rows:
        value = safe_float(r.get(value_key))
        cc = safe_float(r.get(count_key))
        if value is not None and cc is not None and cc > 0:
            n = int(cc)
            numerator += value * n
            count += n
    return float(numerator / count) if count > 0 else None


def select_middle(rows: Sequence[Dict[str, Any]], n: int, key: str) -> List[Dict[str, Any]]:
    finite = [r for r in rows if safe_float(r.get(key)) is not None]
    finite.sort(key=lambda r: float(r[key]))
    if not finite or n <= 0:
        return []
    n = min(n, len(finite))
    center = len(finite) // 2
    start = max(0, center - n // 2)
    end = min(len(finite), start + n)
    return finite[max(0, end - n):end]


def robust_limits(*arrays: np.ndarray, qlo: float = 1.0, qhi: float = 99.0) -> Tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return 0.0, 1.0
    joined = np.concatenate(vals)
    vmin = float(np.percentile(joined, qlo))
    vmax = float(np.percentile(joined, qhi))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(joined))
        vmax = float(np.nanmax(joined))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def core_bounds(height: int, width: int, patch_size: int, radius: int) -> Tuple[int, int, int, int]:
    n_rows = height // patch_size
    n_cols = width // patch_size
    center_row = n_rows // 2
    center_col = n_cols // 2
    row0 = max(0, center_row - radius)
    row1 = min(n_rows, center_row + radius + 1)
    col0 = max(0, center_col - radius)
    col1 = min(n_cols, center_col + radius + 1)
    return col0 * patch_size, row0 * patch_size, col1 * patch_size, row1 * patch_size


def save_quicklook(
    out_png: Path,
    gt_m: np.ndarray,
    pred_m: np.ndarray,
    err_m: np.ndarray,
    core_loss_mask: np.ndarray,
    full_loss_mask: np.ndarray,
    visible_input_mask: np.ndarray,
    valid_pixel_mask: np.ndarray,
    core_box: Tuple[int, int, int, int],
    title: str,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_png.parent.mkdir(parents=True, exist_ok=True)

    gt_core = np.where(core_loss_mask, gt_m, np.nan)
    pred_core = np.where(core_loss_mask, pred_m, np.nan)
    err_core = np.where(core_loss_mask, err_m, np.nan)

    gt_full = np.where(full_loss_mask, gt_m, np.nan)
    pred_full = np.where(full_loss_mask, pred_m, np.nan)
    err_full = np.where(full_loss_mask, err_m, np.nan)

    elev_vmin, elev_vmax = robust_limits(gt_core, pred_core, gt_full, pred_full)
    err_values = np.concatenate([
        np.abs(err_core[np.isfinite(err_core)]),
        np.abs(err_full[np.isfinite(err_full)]),
    ]) if (np.isfinite(err_core).any() or np.isfinite(err_full).any()) else np.asarray([1.0])
    err_vmax = float(np.percentile(err_values, 98)) if err_values.size else 1.0
    err_vmax = max(err_vmax, 1e-6)

    panels = [
        (gt_core, "GT — core loss pixels", elev_vmin, elev_vmax),
        (pred_core, "Prediction — core loss pixels", elev_vmin, elev_vmax),
        (err_core, "Error — core loss pixels", -err_vmax, err_vmax),
        (visible_input_mask.astype(float), "Visible input mask", 0, 1),
        (gt_full, "GT — full loss pixels", elev_vmin, elev_vmax),
        (pred_full, "Prediction — full loss pixels", elev_vmin, elev_vmax),
        (err_full, "Error — full loss pixels", -err_vmax, err_vmax),
        (valid_pixel_mask.astype(float), "Valid pixel mask", 0, 1),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
    x0, y0, x1, y1 = core_box

    for ax, (arr, name, vmin, vmax) in zip(axes.flat, panels):
        im = ax.imshow(arr, vmin=vmin, vmax=vmax)
        ax.add_patch(
            Rectangle(
                (x0 - 0.5, y0 - 0.5),
                x1 - x0,
                y1 - y0,
                fill=False,
                linewidth=2.0,
                linestyle="--",
            )
        )
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def save_bathy_mask_overlay(
    out_png: Path,
    bathy_m: np.ndarray,
    overlay_mask: np.ndarray,
    core_box: Tuple[int, int, int, int],
    title: str,
    mask_label: str,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy.ma as ma

    out_png.parent.mkdir(parents=True, exist_ok=True)
    bathy = np.asarray(bathy_m, dtype=np.float64)
    mask = np.asarray(overlay_mask, dtype=bool)
    bathy_plot = bathy.copy()
    bathy_plot[~np.isfinite(bathy_plot)] = np.nan
    vmin, vmax = robust_limits(bathy_plot)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 7.2))
    im = ax.imshow(bathy_plot, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Elevation / bathymetry (m)")
    overlay = ma.masked_where(~mask, np.ones_like(mask, dtype=float))
    ax.imshow(overlay, alpha=0.38, vmin=0, vmax=1)
    x0, y0, x1, y1 = core_box
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, linewidth=2.0, linestyle="--"))
    ax.set_title(f"{title}\nOverlay: {mask_label} | overlay pixels={int(mask.sum())}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


@torch.no_grad()
def render_selected_visuals(
    model,
    dataset,
    rows: Sequence[Dict[str, Any]],
    args,
    device: torch.device,
    group_name: str,
) -> None:
    out_dir = Path(args.output_dir) / group_name
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_size = int(model.patch_embed.patch_size[0])

    for rank, row in enumerate(rows, start=1):
        index = int(float(row["index"]))
        sample = dataset[index]
        samples, meta, path, hidden, valid, loss_pixel = sample

        xb = samples.unsqueeze(0).to(device)
        hidden_b = hidden.unsqueeze(0).to(device)
        valid_b = valid.unsqueeze(0).to(device)
        loss_b = loss_pixel.unsqueeze(0).to(device)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            _, pred, core_loss_patch_mask, prediction_patch_mask = model(
                xb,
                mask_ratio=args.mask_ratio,
                lcc_mask=hidden_b,
                valid_mask=valid_b,
                loss_pixel_mask=loss_b,
                loss_on_lcc_only=False,
                lcc_mask_mode="exact",
                lcc_patch_threshold=args.lcc_patch_threshold,
                loss_region_mode=args.loss_region_mode,
                core_patch_radius=args.core_patch_radius,
                return_aux_masks=True,
            )

        pred_img = model.unpatchify(pred)[0, 0].detach().float().cpu().numpy()
        target_norm = samples[0].detach().float().cpu().numpy()
        hidden_np = hidden[0].detach().float().cpu().numpy() > 0.5
        valid_np = valid[0].detach().float().cpu().numpy() > 0.5
        loss_np = loss_pixel[0].detach().float().cpu().numpy() > 0.5

        core_loss_patch_px = expand_patch_mask(model, core_loss_patch_mask, args.in_chans)[0, 0].detach().cpu().numpy() > 0.5
        prediction_patch_px = expand_patch_mask(model, prediction_patch_mask, args.in_chans)[0, 0].detach().cpu().numpy() > 0.5
        valid_patch = model._valid_patch_from_mask(valid_b).float()
        valid_patch_px = expand_patch_mask(model, valid_patch, args.in_chans)[0, 0].detach().cpu().numpy() > 0.5

        mean_m = float(meta_item(meta, "tile_mean_m", 0, 0.0))
        std_safe = float(meta_item(meta, "tile_std_safe", 0, 1.0))

        gt_m = target_norm * std_safe + mean_m if args.tile_norm else target_norm
        pred_m = pred_img * std_safe + mean_m if args.tile_norm else pred_img
        err_m = pred_m - gt_m

        core_loss_pix = loss_np & core_loss_patch_px & valid_patch_px
        full_loss_pix = loss_np & prediction_patch_px & valid_patch_px
        visible_input_pix = (~hidden_np) & valid_np

        core_stats = masked_stats(err_m, core_loss_pix)
        full_stats = masked_stats(err_m, full_loss_pix)

        metric_value = safe_float(row.get(args.rank_metric))
        metric_text = "nan" if metric_value is None else f"{metric_value:.4f}"
        river = parse_river(str(path))
        tile_name = Path(str(path)).stem

        sample_dir = out_dir / f"rank{rank:03d}_idx{index:06d}_{args.rank_metric}_{metric_text}_{river}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        metrics = dict(row)
        metrics.update({
            "rendered_path": str(path),
            "rendered_core_rmse_m": core_stats["rmse"],
            "rendered_full_loss_rmse_m": full_stats["rmse"],
            "rendered_core_n_pixels": core_stats["count"],
            "rendered_full_n_pixels": full_stats["count"],
            "tile_mean_m_render": mean_m,
            "tile_std_safe_render": std_safe,
        })
        (sample_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        title = (
            f"{tile_name} | {group_name} rank {rank} | "
            f"{args.rank_metric}={metric_text} m | "
            f"core RMSE={core_stats['rmse']:.3f} m | "
            f"full loss RMSE={full_stats['rmse']:.3f} m | "
            f"std_safe={std_safe:.3f} m"
        )
        core_box = core_bounds(gt_m.shape[0], gt_m.shape[1], patch_size, args.core_patch_radius)

        save_quicklook(
            sample_dir / "quicklook_core_and_full_loss_pixels.png",
            gt_m=gt_m,
            pred_m=pred_m,
            err_m=err_m,
            core_loss_mask=core_loss_pix,
            full_loss_mask=full_loss_pix,
            visible_input_mask=visible_input_pix,
            valid_pixel_mask=valid_np,
            core_box=core_box,
            title=title,
            dpi=args.vis_dpi,
        )

        save_bathy_mask_overlay(
            sample_dir / "overlay_input_bathy_hidden_mask.png",
            bathy_m=gt_m,
            overlay_mask=hidden_np,
            core_box=core_box,
            title=f"{tile_name} | input bathy + Hidden_Mask",
            mask_label="Hidden_Mask (model cannot see)",
            dpi=args.vis_dpi,
        )
        save_bathy_mask_overlay(
            sample_dir / "overlay_input_bathy_final_loss_mask.png",
            bathy_m=gt_m,
            overlay_mask=full_loss_pix,
            core_box=core_box,
            title=f"{tile_name} | input bathy + final Loss_Mask_Pixel",
            mask_label="Final loss pixels, bathy-valid, prediction-supported",
            dpi=args.vis_dpi,
        )

        print(
            f"[VIS] {group_name} rank={rank} idx={index} "
            f"{args.rank_metric}={metric_text} -> {sample_dir}"
        )


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--hidden_list", required=True)
    ap.add_argument("--loss_list", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split_name", default="val")
    ap.add_argument("--model", default="mae_vit_large_patch16")
    ap.add_argument("--input_size", type=int, default=336)
    ap.add_argument("--in_chans", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--nodata", type=float, default=-999999.0)
    ap.add_argument("--nodata_threshold", type=float, default=-9999.0)
    ap.add_argument("--tile_norm", action="store_true")
    ap.add_argument("--tile_norm_visible_only", action="store_true")
    ap.add_argument("--tile_norm_eps", type=float, default=1e-3)
    ap.add_argument("--tile_norm_std_scale", type=float, default=1.5)

    ap.add_argument("--bottleneck_norm", default="inst1d", choices=["none", "inst1d"])
    ap.add_argument("--loss_mode", default="mse", choices=["mse"])
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--lcc_patch_threshold", type=float, default=0.5)
    ap.add_argument("--loss_region_mode", choices=["all", "core"], default="core")
    ap.add_argument("--core_patch_radius", type=int, default=3)

    ap.add_argument("--min_valid_visible_patch_ratio", type=float, default=0.70)
    ap.add_argument("--min_loss_pixel_count", type=int, default=1)
    ap.add_argument("--min_core_loss_pixel_count", type=int, default=0)
    ap.add_argument("--min_core_loss_pixel_ratio", type=float, default=0.0)

    # Visualization controls. Enabled by default.
    ap.add_argument("--rank_metric", default="rmse_m_core_loss_pixel")
    ap.add_argument("--worst_vis", type=int, default=20)
    ap.add_argument("--median_vis", type=int, default=10)
    ap.add_argument("--best_vis", type=int, default=10)
    ap.add_argument("--vis_dpi", type=int, default=180)
    ap.add_argument("--no_visuals", action="store_true")
    args = ap.parse_args()

    add_code_path(args.code_dir)
    import models_mae
    from util.dem_dataset import DEMDualMaskDataset

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    dataset = DEMDualMaskDataset(
        dem_dir=args.data_root,
        dem_list_path=args.list,
        hidden_list_path=args.hidden_list,
        loss_list_path=args.loss_list,
        input_size=args.input_size,
        nodata=args.nodata,
        nodata_threshold=args.nodata_threshold,
        random_flip=False,
        return_path=True,
        tile_norm=args.tile_norm,
        tile_norm_eps=args.tile_norm_eps,
        tile_norm_std_scale=args.tile_norm_std_scale,
        tile_norm_visible_only=args.tile_norm_visible_only,
        min_valid_visible_patch_ratio=args.min_valid_visible_patch_ratio,
        min_loss_pixel_count=args.min_loss_pixel_count,
        min_core_loss_pixel_count=args.min_core_loss_pixel_count,
        min_core_loss_pixel_ratio=args.min_core_loss_pixel_ratio,
        core_patch_radius=args.core_patch_radius,
        patch_size=16,
        hidden_patch_threshold=args.lcc_patch_threshold,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = models_mae.__dict__[args.model](
        norm_pix_loss=False,
        img_size=args.input_size,
        in_chans=args.in_chans,
        bottleneck_norm=args.bottleneck_norm,
        loss_mode=args.loss_mode,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    msg = model.load_state_dict(state, strict=False)
    print(f"[CKPT] loaded: {args.ckpt}")
    print(f"[CKPT] missing_keys={msg.missing_keys}")
    print(f"[CKPT] unexpected_keys={msg.unexpected_keys}")

    model.to(device)
    model.eval()

    rows: List[Dict[str, Any]] = []
    global_index = 0

    for batch_i, batch in enumerate(loader):
        samples, meta, paths, hidden, valid, loss_pixel = batch
        samples = samples.to(device, non_blocking=True)
        hidden = hidden.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        loss_pixel = loss_pixel.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            loss, pred, core_loss_patch_mask, prediction_patch_mask = model(
                samples,
                mask_ratio=args.mask_ratio,
                lcc_mask=hidden,
                valid_mask=valid,
                loss_pixel_mask=loss_pixel,
                loss_on_lcc_only=False,
                lcc_mask_mode="exact",
                lcc_patch_threshold=args.lcc_patch_threshold,
                loss_region_mode=args.loss_region_mode,
                core_patch_radius=args.core_patch_radius,
                return_aux_masks=True,
            )

        pred_img = model.unpatchify(pred).detach().float().cpu().numpy()
        samples_np = samples.detach().float().cpu().numpy()
        valid_np_b = valid.detach().float().cpu().numpy() > 0.5
        loss_np_b = loss_pixel.detach().float().cpu().numpy() > 0.5

        core_loss_patch_px = expand_patch_mask(model, core_loss_patch_mask, args.in_chans).detach().cpu().numpy() > 0.5
        prediction_patch_px = expand_patch_mask(model, prediction_patch_mask, args.in_chans).detach().cpu().numpy() > 0.5
        valid_patch = model._valid_patch_from_mask(valid).float()
        valid_patch_px = expand_patch_mask(model, valid_patch, args.in_chans).detach().cpu().numpy() > 0.5

        bs = samples.shape[0]
        for i in range(bs):
            path = paths[i] if isinstance(paths, (list, tuple)) else str(paths)
            mean_m = float(meta_item(meta, "tile_mean_m", i, 0.0))
            std_m = float(meta_item(meta, "tile_std_m", i, 1.0))
            std_safe = float(meta_item(meta, "tile_std_safe", i, 1.0))

            target_norm = samples_np[i, 0]
            pred_norm = pred_img[i, 0]
            err_norm = pred_norm - target_norm
            err_m = err_norm * std_safe

            valid_pix = valid_np_b[i, 0]
            loss_pix = loss_np_b[i, 0]
            core_loss_pix = loss_pix & core_loss_patch_px[i, 0] & valid_patch_px[i, 0]
            full_loss_pix = loss_pix & prediction_patch_px[i, 0] & valid_patch_px[i, 0]
            prediction_pix = prediction_patch_px[i, 0] & valid_patch_px[i, 0]

            core_norm = masked_stats(err_norm, core_loss_pix)
            core_m = masked_stats(err_m, core_loss_pix)
            full_norm = masked_stats(err_norm, full_loss_pix)
            full_m = masked_stats(err_m, full_loss_pix)
            pred_m = masked_stats(err_m, prediction_pix)

            rmse_norm_core = core_norm["rmse"]
            rmse_m_core = core_m["rmse"]
            norm_to_meter_scale = (
                float(rmse_m_core / rmse_norm_core)
                if math.isfinite(rmse_norm_core) and rmse_norm_core > 0
                else float("nan")
            )

            row = {
                "index": global_index,
                "batch_index": batch_i,
                "split": args.split_name,
                "path": str(path),
                "file": Path(str(path)).name,
                "river": parse_river(str(path)),
                "hidden_path": meta_item(meta, "hidden_path", i, ""),
                "loss_path": meta_item(meta, "loss_path", i, ""),
                "tile_mean_m": mean_m,
                "tile_std_m": std_m,
                "tile_std_safe": std_safe,
                "norm_to_meter_scale_core": norm_to_meter_scale,
                "valid_pixel_ratio": float(meta_item(meta, "valid_pixel_ratio", i, float("nan"))),
                "hidden_pixel_ratio": float(meta_item(meta, "hidden_pixel_ratio", i, float("nan"))),
                "loss_mask_pixel_ratio": float(meta_item(meta, "loss_mask_pixel_ratio", i, float("nan"))),
                "loss_mask_pixel_count_dataset": int(meta_item(meta, "loss_mask_pixel_count", i, 0)),
                "core_loss_pixel_count_dataset": int(meta_item(meta, "core_loss_pixel_count", i, 0)),
                "visible_valid_patch_ratio": float(meta_item(meta, "visible_valid_patch_ratio", i, float("nan"))),
                "prediction_patch_ratio": float(meta_item(meta, "prediction_patch_ratio", i, float("nan"))),
                "ignored_patch_ratio": float(meta_item(meta, "ignored_patch_ratio", i, float("nan"))),
                "n_core_loss_pixels": core_m["count"],
                "loss_norm_core_loss_pixel": core_norm["mse"],
                "rmse_norm_core_loss_pixel": core_norm["rmse"],
                "rmse_m_core_loss_pixel": core_m["rmse"],
                "mae_m_core_loss_pixel": core_m["mae"],
                "bias_m_core_loss_pixel": core_m["bias"],
                "p95_abs_m_core_loss_pixel": core_m["p95_abs"],
                "max_abs_m_core_loss_pixel": core_m["max_abs"],
                "sse_m_core_loss_pixel": core_m["sse"],
                "sse_norm_core_loss_pixel": core_norm["sse"],
                "n_full_loss_pixels": full_m["count"],
                "loss_norm_full_loss_pixel": full_norm["mse"],
                "rmse_norm_full_loss_pixel": full_norm["rmse"],
                "rmse_m_full_loss_pixel": full_m["rmse"],
                "mae_m_full_loss_pixel": full_m["mae"],
                "bias_m_full_loss_pixel": full_m["bias"],
                "sse_m_full_loss_pixel": full_m["sse"],
                "sse_norm_full_loss_pixel": full_norm["sse"],
                "n_prediction_patch_pixels": pred_m["count"],
                "rmse_m_prediction_patch_pixel": pred_m["rmse"],
                "mae_m_prediction_patch_pixel": pred_m["mae"],
                "bias_m_prediction_patch_pixel": pred_m["bias"],
                "sse_m_prediction_patch_pixel": pred_m["sse"],
            }
            rows.append(row)
            global_index += 1

        if batch_i % 50 == 0:
            print(f"[EVAL] batch={batch_i}/{len(loader)} samples={global_index} loss_norm_batch={float(loss.item()):.6g}")

    write_csv(out / "per_tile_metrics.csv", rows)

    per_river: List[Dict[str, Any]] = []
    for river in sorted(set(r["river"] for r in rows)):
        rr = [r for r in rows if r["river"] == river]
        per_river.append({
            "split": args.split_name,
            "river": river,
            "n_tiles": len(rr),
            "global_mae_m_core_loss_pixel": pixel_weighted_mean(rr, "mae_m_core_loss_pixel", "n_core_loss_pixels"),
            "global_rmse_m_core_loss_pixel": pixel_weighted_rmse(rr, "sse_m_core_loss_pixel", "n_core_loss_pixels"),
            "global_bias_m_core_loss_pixel": pixel_weighted_mean(rr, "bias_m_core_loss_pixel", "n_core_loss_pixels"),
            "global_normalized_mse_core_loss_pixel": pixel_weighted_mse(rr, "sse_norm_core_loss_pixel", "n_core_loss_pixels"),
            "global_rmse_norm_core_loss_pixel": pixel_weighted_rmse(rr, "sse_norm_core_loss_pixel", "n_core_loss_pixels"),
            "global_mae_m_full_loss_pixel": pixel_weighted_mean(rr, "mae_m_full_loss_pixel", "n_full_loss_pixels"),
            "global_rmse_m_full_loss_pixel": pixel_weighted_rmse(rr, "sse_m_full_loss_pixel", "n_full_loss_pixels"),
            "global_bias_m_full_loss_pixel": pixel_weighted_mean(rr, "bias_m_full_loss_pixel", "n_full_loss_pixels"),
            "tile_mean_rmse_m_core_loss_pixel": float(np.nanmean([r["rmse_m_core_loss_pixel"] for r in rr])),
            "tile_median_rmse_m_core_loss_pixel": float(np.nanmedian([r["rmse_m_core_loss_pixel"] for r in rr])),
            "tile_p90_rmse_m_core_loss_pixel": float(np.nanpercentile([r["rmse_m_core_loss_pixel"] for r in rr], 90)),
            "tile_mean_rmse_norm_core_loss_pixel": float(np.nanmean([r["rmse_norm_core_loss_pixel"] for r in rr])),
            "tile_mean_std_safe": float(np.nanmean([r["tile_std_safe"] for r in rr])),
            "tile_median_std_safe": float(np.nanmedian([r["tile_std_safe"] for r in rr])),
            "tile_p90_std_safe": float(np.nanpercentile([r["tile_std_safe"] for r in rr], 90)),
            "total_core_loss_pixels": int(sum(r["n_core_loss_pixels"] for r in rr)),
        })
    write_csv(out / "per_river_summary.csv", per_river)

    def sorted_rows(metric: str, reverse: bool = True):
        finite = [r for r in rows if safe_float(r.get(metric)) is not None]
        return sorted(finite, key=lambda r: float(r[metric]), reverse=reverse)

    write_csv(out / "worst_by_rmse_m_core_loss_pixel.csv", sorted_rows("rmse_m_core_loss_pixel")[:200])
    write_csv(out / "worst_by_tile_std_safe.csv", sorted_rows("tile_std_safe")[:200])
    write_csv(out / "worst_by_norm_to_meter_scale.csv", sorted_rows("norm_to_meter_scale_core")[:200])
    write_csv(out / "worst_by_rmse_norm_core_loss_pixel.csv", sorted_rows("rmse_norm_core_loss_pixel")[:200])

    rank_metric = args.rank_metric
    finite_rank_rows = [r for r in rows if safe_float(r.get(rank_metric)) is not None]
    worst_rows = sorted(finite_rank_rows, key=lambda r: float(r[rank_metric]), reverse=True)[:args.worst_vis]
    median_rows = select_middle(finite_rank_rows, args.median_vis, rank_metric)
    best_rows = sorted(finite_rank_rows, key=lambda r: float(r[rank_metric]))[:args.best_vis]

    summary = {
        "split_name": args.split_name,
        "checkpoint": args.ckpt,
        "n_tiles": len(rows),
        "global_pixel_weighted": {
            "mae_m_core_loss_pixel": pixel_weighted_mean(rows, "mae_m_core_loss_pixel", "n_core_loss_pixels"),
            "rmse_m_core_loss_pixel": pixel_weighted_rmse(rows, "sse_m_core_loss_pixel", "n_core_loss_pixels"),
            "bias_m_core_loss_pixel": pixel_weighted_mean(rows, "bias_m_core_loss_pixel", "n_core_loss_pixels"),
            "normalized_mse_core_loss_pixel": pixel_weighted_mse(rows, "sse_norm_core_loss_pixel", "n_core_loss_pixels"),
            "rmse_norm_core_loss_pixel": pixel_weighted_rmse(rows, "sse_norm_core_loss_pixel", "n_core_loss_pixels"),
            "mae_m_full_loss_pixel": pixel_weighted_mean(rows, "mae_m_full_loss_pixel", "n_full_loss_pixels"),
            "rmse_m_full_loss_pixel": pixel_weighted_rmse(rows, "sse_m_full_loss_pixel", "n_full_loss_pixels"),
            "bias_m_full_loss_pixel": pixel_weighted_mean(rows, "bias_m_full_loss_pixel", "n_full_loss_pixels"),
            "normalized_mse_full_loss_pixel": pixel_weighted_mse(rows, "sse_norm_full_loss_pixel", "n_full_loss_pixels"),
            "rmse_norm_full_loss_pixel": pixel_weighted_rmse(rows, "sse_norm_full_loss_pixel", "n_full_loss_pixels"),
            "mae_m_prediction_patch_pixel": pixel_weighted_mean(rows, "mae_m_prediction_patch_pixel", "n_prediction_patch_pixels"),
            "rmse_m_prediction_patch_pixel": pixel_weighted_rmse(rows, "sse_m_prediction_patch_pixel", "n_prediction_patch_pixels"),
            "bias_m_prediction_patch_pixel": pixel_weighted_mean(rows, "bias_m_prediction_patch_pixel", "n_prediction_patch_pixels"),
        },
        "per_tile_summary": {
            "rmse_m_core_loss_pixel": summary_values([r["rmse_m_core_loss_pixel"] for r in rows]),
            "rmse_norm_core_loss_pixel": summary_values([r["rmse_norm_core_loss_pixel"] for r in rows]),
            "tile_std_safe": summary_values([r["tile_std_safe"] for r in rows]),
            "norm_to_meter_scale_core": summary_values([r["norm_to_meter_scale_core"] for r in rows]),
            "n_core_loss_pixels": summary_values([r["n_core_loss_pixels"] for r in rows]),
        },
        "visual_selection": {
            "enabled": not args.no_visuals,
            "rank_metric": rank_metric,
            "worst_vis": len(worst_rows),
            "median_vis": len(median_rows),
            "best_vis": len(best_rows),
            "worst_indices": [int(r["index"]) for r in worst_rows],
            "median_indices": [int(r["index"]) for r in median_rows],
            "best_indices": [int(r["index"]) for r in best_rows],
        },
        "notes": [
            "mae_m_core_loss_pixel and rmse_m_core_loss_pixel use Loss_Mask_Pixel AND core/prediction patch mask AND valid patch mask.",
            "normalized_mse_core_loss_pixel and rmse_norm_core_loss_pixel are computed over exactly the same pixels as the meter metrics.",
            "rmse_m_core_loss_pixel ≈ rmse_norm_core_loss_pixel * tile_std_safe per tile.",
            "Large train meter RMSE with small normalized loss usually indicates large tile_std_safe in some train rivers/tiles.",
            "Visuals show core loss pixels, full loss pixels, visible input mask, and valid pixel mask.",
            "Each visualized sample also includes overlay_input_bathy_hidden_mask.png and overlay_input_bathy_final_loss_mask.png for mask QA.",
        ],
    }
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    if not args.no_visuals:
        render_selected_visuals(
            model, dataset, worst_rows, args, device,
            group_name=f"visuals_worst_by_{rank_metric}",
        )
        render_selected_visuals(
            model, dataset, median_rows, args, device,
            group_name=f"visuals_median_by_{rank_metric}",
        )
        render_selected_visuals(
            model, dataset, best_rows, args, device,
            group_name=f"visuals_best_by_{rank_metric}",
        )

    print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
