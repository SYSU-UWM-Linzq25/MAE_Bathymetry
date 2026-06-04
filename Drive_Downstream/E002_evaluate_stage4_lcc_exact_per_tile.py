#!/usr/bin/env python3
"""Per-tile evaluation for Stage4 bathymetry + final-mask/LCC exact-mask MAE runs.

This script is designed for the downstream bathymetry task where each bathy tile has
a paired LCC mask. Unlike the older random-mask evaluation scripts, this evaluates
with the exact LCC-derived patch mask and writes per-sample metrics so hard/coastal
or out-of-domain tiles can be diagnosed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _add_code_path(code_dir: str) -> None:
    code_dir = str(Path(code_dir).resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)


def _get_meta(meta: Dict[str, Any], key: str, i: int, default=None):
    if key not in meta:
        return default
    v = meta[key]
    if torch.is_tensor(v):
        item = v[i]
        if item.numel() == 1:
            return item.item()
        return item.detach().cpu().numpy()
    if isinstance(v, (list, tuple)):
        return v[i]
    return v


def _write_geotiff_like(ref_path: str, out_path: Path, arr2d, dtype: str = "float32", nodata=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr2d = np.asarray(arr2d)
    try:
        import rasterio
        with rasterio.open(ref_path) as src:
            profile = src.profile.copy()
        h, w = arr2d.shape
        if profile.get("height") != h or profile.get("width") != w:
            import tifffile
            tifffile.imwrite(str(out_path), arr2d.astype(np.float32))
            return
        profile.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata,
                       compress=profile.get("compress", "LZW"))
        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(arr2d.astype(dtype), 1)
    except Exception:
        import tifffile
        tifffile.imwrite(str(out_path), arr2d.astype(np.float32))


def _safe_float(x):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _summary(vals: np.ndarray) -> Dict[str, Optional[float]]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {k: None for k in ["mean", "std", "median", "p75", "p90", "p95", "min", "max"]}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _save_quicklook_png(out_png: Path, gt, recon, err, lcc, patch_mask, title: str):
    try:
        import matplotlib.pyplot as plt
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        ims = []
        ims.append(axes[0].imshow(gt)); axes[0].set_title("GT (m)")
        ims.append(axes[1].imshow(recon)); axes[1].set_title("Recon (m)")
        vmax = float(np.nanpercentile(np.abs(err), 98)) if np.isfinite(err).any() else 1.0
        vmax = max(vmax, 1e-6)
        ims.append(axes[2].imshow(err, vmin=-vmax, vmax=vmax)); axes[2].set_title("Error (m)")
        ims.append(axes[3].imshow(lcc)); axes[3].set_title("LCC pixel mask")
        ims.append(axes[4].imshow(patch_mask)); axes[4].set_title("MAE patch mask")
        for ax, im in zip(axes, ims):
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] failed to save quicklook {out_png}: {e}")


@torch.no_grad()
def save_visuals(model, dataset, rows: List[Dict[str, Any]], args, device: torch.device, split_name: str,
                 vis_dir_name: str, max_vis: int, rank_label: str = "rank"):
    """Save GeoTIFF/PNG quicklooks for selected rows.

    This is used for worst tiles, best tiles, and median/representative tiles.
    """
    if max_vis <= 0 or not rows:
        return
    out_dir = Path(args.output_dir) / vis_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for rank, row in enumerate(rows[:max_vis], start=1):
        idx = int(row["index"])
        sample = dataset[idx]
        if len(sample) == 4:
            x, meta, ref_path, lcc = sample
        elif len(sample) == 3:
            x, meta, lcc = sample
            ref_path = meta.get("path", dataset.files[idx])
        else:
            raise RuntimeError("Unexpected dataset sample format; expected LCC paired dataset.")

        x_b = x.unsqueeze(0).to(device)
        lcc_b = lcc.unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            _, pred, mask = model(
                x_b,
                mask_ratio=args.mask_ratio,
                lcc_mask=lcc_b,
                loss_on_lcc_only=args.loss_on_lcc_only,
                lcc_priority=args.lcc_priority,
                lcc_mask_mode=args.lcc_mask_mode,
                lcc_patch_threshold=args.lcc_patch_threshold,
            )
        pred_img = model.unpatchify(pred)[0, 0].float().cpu()
        p = model.patch_embed.patch_size[0]
        mask_img = mask.unsqueeze(-1).repeat(1, 1, p * p * args.in_chans)
        mask_img = model.unpatchify(mask_img)[0, 0].float().cpu()
        x0 = x[0].float().cpu()
        recon = x0 * (1 - mask_img) + pred_img * mask_img

        tile_mean = float(meta.get("tile_mean_m", 0.0))
        tile_std = float(meta.get("tile_std_safe", 1.0))
        if args.tile_norm:
            gt_m = x0 * tile_std + tile_mean
            pred_m = pred_img * tile_std + tile_mean
            recon_m = recon * tile_std + tile_mean
        else:
            gt_m = x0
            pred_m = pred_img
            recon_m = recon
        err_m = recon_m - gt_m

        rmse_val = _safe_float(row.get('rmse_m_mask'))
        rmse_txt = "nan" if rmse_val is None else f"{rmse_val:.3f}"
        base = f"{rank_label}{rank:03d}_idx{idx:06d}_rmse{rmse_txt}"
        out_sample = out_dir / base
        out_sample.mkdir(parents=True, exist_ok=True)
        gt_np = gt_m.numpy().astype(np.float32)
        pred_np = pred_m.numpy().astype(np.float32)
        recon_np = recon_m.numpy().astype(np.float32)
        err_np = err_m.numpy().astype(np.float32)
        lcc_np = lcc[0].numpy().astype(np.uint8)
        mask_np = (mask_img.numpy() > 0.5).astype(np.uint8)
        _write_geotiff_like(ref_path, out_sample / "gt_m.tif", gt_np, "float32")
        _write_geotiff_like(ref_path, out_sample / "pred_m.tif", pred_np, "float32")
        _write_geotiff_like(ref_path, out_sample / "recon_m.tif", recon_np, "float32")
        _write_geotiff_like(ref_path, out_sample / "err_m.tif", err_np, "float32")
        _write_geotiff_like(ref_path, out_sample / "lcc_input_mask.tif", lcc_np, "uint8", nodata=0)
        _write_geotiff_like(ref_path, out_sample / "mae_patch_mask.tif", mask_np, "uint8", nodata=0)
        with open(out_sample / "metrics.json", "w") as f:
            json.dump(row, f, indent=2)
        _save_quicklook_png(
            out_sample / "quicklook.png",
            gt_np,
            recon_np,
            err_np,
            lcc_np,
            mask_np,
            title=f"{rank_label}={rank} idx={idx} rmse={rmse_txt} m lcc_patch={row.get('lcc_patch_ratio_meta')}",
        )


def _finite_rows(rows: List[Dict[str, Any]], key: str = "rmse_m_mask") -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        v = _safe_float(r.get(key))
        if v is not None:
            out.append(r)
    return out


def _middle_rows(rows_asc: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Return rows centered around the median RMSE from an ascending list."""
    if n <= 0 or not rows_asc:
        return []
    n = min(n, len(rows_asc))
    mid = len(rows_asc) // 2
    start = max(0, mid - n // 2)
    end = min(len(rows_asc), start + n)
    start = max(0, end - n)
    return rows_asc[start:end]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--list", required=True, help="DEM/bathy list txt")
    ap.add_argument("--lcc_mask_path", required=True)
    ap.add_argument("--lcc_list", required=True, help="LCC mask list txt paired line-by-line with --list")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split_name", default="val")
    ap.add_argument("--model", default="mae_vit_large_patch16")
    ap.add_argument("--input_size", type=int, default=336)
    ap.add_argument("--in_chans", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--nodata", type=float, default=-9999.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--tile_norm", action="store_true")
    ap.add_argument("--tile_norm_eps", type=float, default=1e-3)
    ap.add_argument("--tile_norm_std_scale", type=float, default=1.0,
                    help="Scale factor multiplied to tile std for tile-wise normalization. Must match training.")
    ap.add_argument("--tile_norm_visible_only", action="store_true")
    ap.add_argument("--bottleneck_norm", default="inst1d", choices=["none", "inst1d"])
    ap.add_argument("--loss_mode", default="mse", choices=["mse"])
    ap.add_argument("--mask_ratio", type=float, default=0.75, help="Ignored by exact mode but kept for model API compatibility")
    ap.add_argument("--lcc_mask_mode", default="exact", choices=["exact", "priority", "none"])
    ap.add_argument("--loss_on_lcc_only", action="store_true")
    ap.add_argument("--lcc_priority", type=float, default=10.0)
    ap.add_argument("--lcc_patch_threshold", type=float, default=0.5)
    ap.add_argument("--min_lcc_patch_ratio", type=float, default=0.0001)
    ap.add_argument("--max_lcc_patch_ratio", type=float, default=0.80)
    ap.add_argument("--topk", type=int, default=200, help="Number of worst-error rows to write to top_errors CSV")
    ap.add_argument("--topk_vis", type=int, default=50, help="Number of worst-error tiles to save GeoTIFF/PNG visuals for")
    ap.add_argument("--bestk", type=int, default=200, help="Number of best-error rows to write to best CSV")
    ap.add_argument("--bestk_vis", type=int, default=50, help="Number of best-error tiles to save GeoTIFF/PNG visuals for")
    ap.add_argument("--median_vis", type=int, default=30, help="Number of median/typical-error tiles to visualize")
    ap.add_argument("--good_min_lcc_patch_ratio", type=float, default=0.02, help="Minimum patch-level LCC ratio for non-trivial good samples")
    ap.add_argument("--good_max_lcc_patch_ratio", type=float, default=0.60, help="Maximum patch-level LCC ratio for non-trivial good samples")
    ap.add_argument("--good_min_masked_gt_std", type=float, default=0.5, help="Minimum masked-region GT std in meters for non-trivial good samples")
    args = ap.parse_args()

    _add_code_path(args.code_dir)
    import models_mae
    from util.dem_dataset import DEMLCCPairDataset

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] ckpt={args.ckpt}")
    print(f"[INFO] list={args.list}")
    print(f"[INFO] lcc_list={args.lcc_list}")
    print(f"[INFO] tile_norm={args.tile_norm}")
    print(f"[INFO] tile_norm_visible_only={args.tile_norm_visible_only}")
    print(f"[INFO] tile_norm_std_scale={args.tile_norm_std_scale}")
    print(f"[INFO] tile_norm_eps={args.tile_norm_eps}")

    dataset = DEMLCCPairDataset(
        dem_dir=args.data_root,
        lcc_dir=args.lcc_mask_path,
        dem_list_path=args.list,
        lcc_list_path=args.lcc_list,
        input_size=args.input_size,
        nodata=args.nodata,
        random_flip=False,
        return_path=True,
        tile_norm=args.tile_norm,
        tile_norm_eps=args.tile_norm_eps,
        tile_norm_std_scale=args.tile_norm_std_scale,
        tile_norm_visible_only=args.tile_norm_visible_only,
        min_lcc_patch_ratio=args.min_lcc_patch_ratio,
        max_lcc_patch_ratio=args.max_lcc_patch_ratio,
        patch_size=16,
        lcc_patch_threshold=args.lcc_patch_threshold,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
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
    print(f"[CKPT] loaded {args.ckpt}")
    print(f"[CKPT] missing_keys={msg.missing_keys}")
    print(f"[CKPT] unexpected_keys={msg.unexpected_keys}")
    model.to(device)
    model.eval()

    rows: List[Dict[str, Any]] = []
    global_index = 0
    for ib, batch in enumerate(loader):
        samples, meta, paths, lcc = batch
        samples = samples.to(device, non_blocking=True)
        lcc = lcc.to(device, non_blocking=True)
        # Evaluation only: disable gradient tracking so tensors can be safely
        # moved to CPU/NumPy and memory stays low.
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                _, pred, mask = model(
                    samples,
                    mask_ratio=args.mask_ratio,
                    lcc_mask=lcc,
                    loss_on_lcc_only=args.loss_on_lcc_only,
                    lcc_priority=args.lcc_priority,
                    lcc_mask_mode=args.lcc_mask_mode,
                    lcc_patch_threshold=args.lcc_patch_threshold,
                )
            target = model.patchify(samples).float().cpu()  # [B,L,P]
            pred_cpu = pred.float().cpu()
            mask_cpu = mask.float().cpu()
            samples_cpu = samples.float().cpu()
            lcc_cpu = lcc.float().cpu()
        B = samples_cpu.shape[0]
        for i in range(B):
            path = paths[i] if isinstance(paths, (list, tuple)) else str(paths)
            mask_i = mask_cpu[i] > 0.5
            diff = pred_cpu[i] - target[i]
            tile_std = float(_get_meta(meta, "tile_std_safe", i, 1.0)) if args.tile_norm else 1.0
            tile_mean = float(_get_meta(meta, "tile_mean_m", i, 0.0)) if args.tile_norm else 0.0

            if mask_i.sum().item() > 0:
                diff_mask_m = diff[mask_i].reshape(-1).detach().cpu().numpy() * tile_std
                rmse_mask = float(np.sqrt(np.mean(diff_mask_m ** 2)))
                mae_mask = float(np.mean(np.abs(diff_mask_m)))
                bias_mask = float(np.mean(diff_mask_m))
                masked_gt_m = (target[i][mask_i].reshape(-1).detach().cpu().numpy() * tile_std + tile_mean)
                masked_gt_std = float(np.std(masked_gt_m)) if masked_gt_m.size else None
                masked_gt_range = float(np.max(masked_gt_m) - np.min(masked_gt_m)) if masked_gt_m.size else None
            else:
                rmse_mask = mae_mask = bias_mask = float("nan")
                masked_gt_std = masked_gt_range = None

            keep_i = ~mask_i
            if keep_i.sum().item() > 0 and mask_i.sum().item() > 0:
                vis_bias_norm = torch.median(diff[keep_i].reshape(-1)).item()
                diff_mask_corr_m = (diff[mask_i].reshape(-1).detach().cpu().numpy() - vis_bias_norm) * tile_std
                rmse_mask_viscorr = float(np.sqrt(np.mean(diff_mask_corr_m ** 2)))
                bias_vis_med_m = float(vis_bias_norm * tile_std)
            else:
                rmse_mask_viscorr = float("nan")
                bias_vis_med_m = float("nan")

            diff_all = torch.zeros_like(diff)
            diff_all[mask_i] = diff[mask_i]
            diff_all_m = diff_all.reshape(-1).detach().cpu().numpy() * tile_std
            rmse_all = float(np.sqrt(np.mean(diff_all_m ** 2)))

            x_m = samples_cpu[i, 0].detach().cpu().numpy() * tile_std + tile_mean
            gt_range = float(np.nanmax(x_m) - np.nanmin(x_m))
            gt_std = float(np.nanstd(x_m))

            lcc_np = lcc_cpu[i, 0].detach().cpu().numpy() > 0.5
            if lcc_np.any():
                ys, xs = np.where(lcc_np)
                bbox_area_ratio = float(((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) / lcc_np.size)
            else:
                bbox_area_ratio = 0.0
            border = np.concatenate([lcc_np[0, :], lcc_np[-1, :], lcc_np[:, 0], lcc_np[:, -1]])
            border_frac = float(border.mean())
            touches_border = int(border.any())

            row = {
                "index": global_index,
                "path": path,
                "mask_path": _get_meta(meta, "lcc_path", i, ""),
                "file": Path(path).name,
                "rmse_m_mask": rmse_mask,
                "mae_m_mask": mae_mask,
                "bias_m_mask": bias_mask,
                "rmse_m_all": rmse_all,
                "rmse_m_mask_viscorr": rmse_mask_viscorr,
                "bias_m_vis_med": bias_vis_med_m,
                "actual_mask_ratio": float(mask_cpu[i].mean().item()),
                "lcc_pixel_ratio": float(_get_meta(meta, "lcc_pixel_ratio", i, float(lcc_np.mean()))),
                "lcc_patch_ratio_meta": float(_get_meta(meta, "lcc_patch_ratio", i, float(mask_cpu[i].mean().item()))),
                "tile_mean_m": tile_mean,
                "tile_std_m": float(_get_meta(meta, "tile_std_m", i, gt_std)),
                "tile_std_safe": tile_std,
                "gt_std_m": gt_std,
                "gt_range_m": gt_range,
                "masked_gt_std_m": masked_gt_std,
                "masked_gt_range_m": masked_gt_range,
                "lcc_bbox_area_ratio": bbox_area_ratio,
                "lcc_border_frac": border_frac,
                "lcc_touches_border": touches_border,
            }
            rows.append(row)
            global_index += 1
        if ib % 20 == 0:
            print(f"[EVAL] batch {ib+1}/{len(loader)} samples={global_index}")

    finite_rows = _finite_rows(rows, "rmse_m_mask")
    # Worst/high-error rows: useful for diagnosing coastal/open-water/out-of-domain or data-quality issues.
    rows_worst = sorted(finite_rows, key=lambda r: float(r["rmse_m_mask"]), reverse=True)
    # Best/low-error rows: useful for understanding where the model works well.
    rows_best = sorted(finite_rows, key=lambda r: float(r["rmse_m_mask"]))
    # Good but non-trivial rows: avoid declaring success only on nearly-flat or tiny-mask tiles.
    rows_best_nontrivial = []
    for r in rows_best:
        lcc_pr = _safe_float(r.get("lcc_patch_ratio_meta"))
        masked_std = _safe_float(r.get("masked_gt_std_m"))
        if lcc_pr is None or masked_std is None:
            continue
        if args.good_min_lcc_patch_ratio <= lcc_pr <= args.good_max_lcc_patch_ratio and masked_std >= args.good_min_masked_gt_std:
            rows_best_nontrivial.append(r)
    rows_median = _middle_rows(rows_best, args.median_vis)

    metrics_csv = out_dir / f"per_tile_metrics_{args.split_name}.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_worst)

    top_csv = out_dir / f"top{args.topk:03d}_errors_{args.split_name}.csv"
    with open(top_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_worst[:args.topk])

    best_csv = out_dir / f"best{args.bestk:03d}_tiles_{args.split_name}.csv"
    with open(best_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_best[:args.bestk])

    best_nontrivial_csv = out_dir / f"best{args.bestk:03d}_nontrivial_tiles_{args.split_name}.csv"
    with open(best_nontrivial_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_best_nontrivial[:args.bestk])

    median_csv = out_dir / f"median{len(rows_median):03d}_tiles_{args.split_name}.csv"
    with open(median_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_median)

    summary = {
        "n": len(rows),
        "n_finite_rmse": len(finite_rows),
        "split_name": args.split_name,
        "tile_norm": bool(args.tile_norm),
        "tile_norm_visible_only": bool(args.tile_norm_visible_only),
        "tile_norm_std_scale": float(args.tile_norm_std_scale),
        "tile_norm_eps": float(args.tile_norm_eps),
        "metrics_csv": str(metrics_csv),
        "top_csv": str(top_csv),
        "best_csv": str(best_csv),
        "best_nontrivial_csv": str(best_nontrivial_csv),
        "median_csv": str(median_csv),
        "good_nontrivial_rule": {
            "good_min_lcc_patch_ratio": args.good_min_lcc_patch_ratio,
            "good_max_lcc_patch_ratio": args.good_max_lcc_patch_ratio,
            "good_min_masked_gt_std": args.good_min_masked_gt_std,
            "n_good_nontrivial": len(rows_best_nontrivial),
        },
    }
    for key in ["rmse_m_mask", "mae_m_mask", "bias_m_mask", "rmse_m_all", "rmse_m_mask_viscorr", "actual_mask_ratio", "lcc_pixel_ratio", "lcc_patch_ratio_meta", "tile_std_m", "gt_range_m", "masked_gt_std_m", "masked_gt_range_m", "lcc_border_frac", "lcc_bbox_area_ratio"]:
        vals = np.array([_safe_float(r.get(key)) for r in finite_rows], dtype=object)
        vals = np.array([v for v in vals if v is not None], dtype=float)
        summary[key] = _summary(vals)
    with open(out_dir / f"summary_{args.split_name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    save_visuals(model, dataset, rows_worst, args, device, args.split_name,
                 vis_dir_name=f"worst{args.topk_vis:03d}_visuals_{args.split_name}",
                 max_vis=args.topk_vis, rank_label="worst")
    save_visuals(model, dataset, rows_best, args, device, args.split_name,
                 vis_dir_name=f"best{args.bestk_vis:03d}_visuals_{args.split_name}",
                 max_vis=args.bestk_vis, rank_label="best")
    save_visuals(model, dataset, rows_best_nontrivial, args, device, args.split_name,
                 vis_dir_name=f"best_nontrivial{args.bestk_vis:03d}_visuals_{args.split_name}",
                 max_vis=args.bestk_vis, rank_label="best_nontrivial")
    save_visuals(model, dataset, rows_median, args, device, args.split_name,
                 vis_dir_name=f"median{args.median_vis:03d}_visuals_{args.split_name}",
                 max_vis=args.median_vis, rank_label="median")
    print(f"[DONE] wrote {metrics_csv}")
    print(f"[DONE] wrote {top_csv}")


if __name__ == "__main__":
    main()
