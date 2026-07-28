#!/usr/bin/env python3
# NUMBER-ALIGNED NAME: E003_evaluate_stage4_nodata_core_per_tile.py
# ORIGINAL BACKUP NAME: E003_evaluate_stage4_nodata_core_per_tile.py
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
"""NoData-safe per-tile evaluation for the Stage4 core-loss bathymetry MAE.

This evaluator matches the current D004 training logic:
  * exact final-mask/LCC patch masking;
  * any patch containing any NoData pixel is ignored;
  * optional centered core region for loss/evaluation;
  * full-tile prediction mask remains separate from the core loss mask;
  * tile-wise normalization and de-normalization use dataset metadata.

It reports both:
  1) patch-core RMSE, matching the training/validation metric; and
  2) exact-pixel RMSE, restricted to the original final-mask pixels, which is
     usually more useful for diagnosing river-channel prediction quality.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _add_code_path(code_dir: str) -> None:
    p = str(Path(code_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _safe_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _meta_item(meta: Dict[str, Any], key: str, index: int, default=None):
    if key not in meta:
        return default
    value = meta[key]
    if torch.is_tensor(value):
        item = value[index]
        return item.item() if item.numel() == 1 else item.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _expand_patch_mask(model, patch_mask: torch.Tensor, in_chans: int = 1) -> torch.Tensor:
    """Convert [B,L] patch mask to [B,1,H,W] pixel mask."""
    p = int(model.patch_embed.patch_size[0])
    expanded = patch_mask.unsqueeze(-1).repeat(1, 1, p * p * in_chans)
    return model.unpatchify(expanded)[:, :1]


def _masked_stats(err: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    values = np.asarray(err, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "sse": 0.0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
            "max_abs": float("nan"),
            "p95_abs": float("nan"),
        }
    abs_values = np.abs(values)
    sse = float(np.square(values).sum(dtype=np.float64))
    return {
        "count": int(values.size),
        "sse": sse,
        "rmse": float(np.sqrt(sse / values.size)),
        "mae": float(abs_values.mean()),
        "bias": float(values.mean()),
        "max_abs": float(abs_values.max()),
        "p95_abs": float(np.percentile(abs_values, 95)),
    }


def _summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    a = np.asarray([x for x in values if _safe_float(x) is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "count": 0, "mean": None, "std": None, "median": None,
            "p75": None, "p90": None, "p95": None, "min": None, "max": None,
        }
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_geotiff_like(
    reference: str,
    output: Path,
    array: np.ndarray,
    dtype: str,
    nodata: Optional[float],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    try:
        import rasterio
        with rasterio.open(reference) as src:
            profile = src.profile.copy()
        if profile["height"] != arr.shape[0] or profile["width"] != arr.shape[1]:
            raise ValueError("Output dimensions do not match source raster.")
        profile.update(
            driver="GTiff",
            count=1,
            dtype=dtype,
            nodata=nodata,
            compress="LZW",
        )
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(arr.astype(dtype), 1)
    except Exception:
        import tifffile
        tifffile.imwrite(str(output), arr.astype(dtype))


def _quicklook(
    output: Path,
    gt: np.ndarray,
    reconstruction: np.ndarray,
    exact_error: np.ndarray,
    final_mask: np.ndarray,
    loss_patch_mask: np.ndarray,
    prediction_patch_mask: np.ndarray,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt

        valid_error = exact_error[np.isfinite(exact_error)]
        vmax = float(np.percentile(np.abs(valid_error), 98)) if valid_error.size else 1.0
        vmax = max(vmax, 1e-6)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        panels = [
            (gt, "GT elevation (m)", None),
            (reconstruction, "Patch reconstruction (m)", None),
            (exact_error, "Exact final-mask error (m)", (-vmax, vmax)),
            (final_mask, "Final pixel mask", None),
            (loss_patch_mask, "Core loss patch mask", None),
            (prediction_patch_mask, "Full prediction patch mask", None),
        ]
        for ax, (arr, name, limits) in zip(axes.ravel(), panels):
            if limits is None:
                im = ax.imshow(arr)
            else:
                im = ax.imshow(arr, vmin=limits[0], vmax=limits[1])
            ax.set_title(name)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] quicklook failed for {output}: {exc}")


def _select_middle(rows: Sequence[Dict[str, Any]], n: int, key: str) -> List[Dict[str, Any]]:
    finite = [r for r in rows if _safe_float(r.get(key)) is not None]
    finite.sort(key=lambda r: float(r[key]))
    if not finite or n <= 0:
        return []
    n = min(n, len(finite))
    center = len(finite) // 2
    start = max(0, center - n // 2)
    end = min(len(finite), start + n)
    return finite[max(0, end - n):end]


@torch.no_grad()
def _save_selected_visuals(
    model,
    dataset,
    rows: Sequence[Dict[str, Any]],
    args,
    device: torch.device,
    folder: str,
    rank_key: str,
) -> None:
    out_root = Path(args.output_dir) / folder
    out_root.mkdir(parents=True, exist_ok=True)
    nodata_out = float(args.output_nodata)

    for rank, row in enumerate(rows, start=1):
        index = int(row["index"])
        x, meta, path, lcc, valid = dataset[index]
        xb = x.unsqueeze(0).to(device)
        lccb = lcc.unsqueeze(0).to(device)
        validb = valid.unsqueeze(0).to(device)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            _, pred, loss_mask, prediction_mask = model(
                xb,
                mask_ratio=args.mask_ratio,
                lcc_mask=lccb,
                valid_mask=validb,
                loss_on_lcc_only=True,
                lcc_priority=args.lcc_priority,
                lcc_mask_mode="exact",
                lcc_patch_threshold=args.lcc_patch_threshold,
                loss_region_mode=args.loss_region_mode,
                core_patch_radius=args.core_patch_radius,
                return_aux_masks=True,
            )

        pred_img = model.unpatchify(pred)[0, 0].detach().float().cpu().numpy()
        loss_img = (_expand_patch_mask(model, loss_mask.float(), args.in_chans)[0, 0]
                    .detach().float().cpu().numpy() > 0.5)
        prediction_img = (
            _expand_patch_mask(model, prediction_mask.float(), args.in_chans)[0, 0]
            .float().cpu().numpy() > 0.5
        )

        x_np = x[0].float().numpy()
        lcc_np = lcc[0].numpy() > 0.5
        valid_np = valid[0].numpy() > 0.5
        mean_m = float(meta["tile_mean_m"])
        std_safe = float(meta["tile_std_safe"])

        gt_m = x_np * std_safe + mean_m if args.tile_norm else x_np
        pred_m = pred_img * std_safe + mean_m if args.tile_norm else pred_img

        reconstruction = gt_m.copy()
        reconstruction[prediction_img & valid_np] = pred_m[prediction_img & valid_np]

        exact_core = lcc_np & valid_np & loss_img
        exact_full = lcc_np & valid_np & prediction_img
        err = pred_m - gt_m

        gt_out = np.where(valid_np, gt_m, nodata_out).astype(np.float32)
        pred_out = np.where(prediction_img & valid_np, pred_m, nodata_out).astype(np.float32)
        recon_out = np.where(valid_np, reconstruction, nodata_out).astype(np.float32)
        err_patch_out = np.where(prediction_img & valid_np, err, nodata_out).astype(np.float32)
        err_core_exact = np.where(exact_core, err, np.nan).astype(np.float32)
        err_full_exact = np.where(exact_full, err, np.nan).astype(np.float32)

        metric = _safe_float(row.get(rank_key))
        metric_text = "nan" if metric is None else f"{metric:.3f}"
        sample_dir = out_root / f"rank{rank:03d}_idx{index:06d}_{rank_key}{metric_text}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        _write_geotiff_like(path, sample_dir / "gt_m.tif", gt_out, "float32", nodata_out)
        _write_geotiff_like(path, sample_dir / "pred_patch_m.tif", pred_out, "float32", nodata_out)
        _write_geotiff_like(path, sample_dir / "recon_patch_m.tif", recon_out, "float32", nodata_out)
        _write_geotiff_like(path, sample_dir / "err_prediction_patch_m.tif",
                            np.where(np.isfinite(err_patch_out), err_patch_out, nodata_out),
                            "float32", nodata_out)
        _write_geotiff_like(path, sample_dir / "err_core_exact_pixel_m.tif",
                            np.where(np.isfinite(err_core_exact), err_core_exact, nodata_out),
                            "float32", nodata_out)
        _write_geotiff_like(path, sample_dir / "err_full_exact_pixel_m.tif",
                            np.where(np.isfinite(err_full_exact), err_full_exact, nodata_out),
                            "float32", nodata_out)
        _write_geotiff_like(path, sample_dir / "final_mask_pixel.tif",
                            lcc_np.astype(np.uint8), "uint8", 0)
        _write_geotiff_like(path, sample_dir / "valid_pixel_mask.tif",
                            valid_np.astype(np.uint8), "uint8", 0)
        _write_geotiff_like(path, sample_dir / "loss_patch_mask.tif",
                            loss_img.astype(np.uint8), "uint8", 0)
        _write_geotiff_like(path, sample_dir / "prediction_patch_mask.tif",
                            prediction_img.astype(np.uint8), "uint8", 0)

        with (sample_dir / "metrics.json").open("w") as f:
            json.dump(row, f, indent=2)

        _quicklook(
            sample_dir / "quicklook.png",
            np.where(valid_np, gt_m, np.nan),
            np.where(valid_np, reconstruction, np.nan),
            err_core_exact,
            lcc_np,
            loss_img,
            prediction_img,
            title=f"{Path(path).name} | {rank_key}={metric_text} m",
        )


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--lcc_mask_path", required=True)
    ap.add_argument("--lcc_list", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split_name", default="val")
    ap.add_argument("--model", default="mae_vit_large_patch16")
    ap.add_argument("--input_size", type=int, default=336)
    ap.add_argument("--in_chans", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--nodata", type=float, default=-999999.0)
    ap.add_argument("--nodata_threshold", type=float, default=-9999.0)
    ap.add_argument("--output_nodata", type=float, default=-999999.0)

    ap.add_argument("--tile_norm", action="store_true")
    ap.add_argument("--tile_norm_visible_only", action="store_true")
    ap.add_argument("--tile_norm_eps", type=float, default=1e-3)
    ap.add_argument("--tile_norm_std_scale", type=float, default=1.5)

    ap.add_argument("--bottleneck_norm", default="inst1d", choices=["none", "inst1d"])
    ap.add_argument("--loss_mode", default="mse", choices=["mse"])
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--lcc_priority", type=float, default=10.0)
    ap.add_argument("--lcc_patch_threshold", type=float, default=0.5)

    ap.add_argument("--min_prediction_patch_ratio", type=float, default=0.0001)
    ap.add_argument("--max_prediction_patch_ratio", type=float, default=0.80)
    ap.add_argument("--min_valid_visible_patch_ratio", type=float, default=0.70)
    ap.add_argument("--loss_region_mode", choices=["all", "core"], default="core")
    ap.add_argument("--core_patch_radius", type=int, default=3)
    ap.add_argument("--min_core_valid_patch_ratio", type=float, default=0.85)
    ap.add_argument("--min_core_prediction_patch_ratio", type=float, default=0.02)
    ap.add_argument("--max_core_prediction_patch_ratio", type=float, default=0.90)

    ap.add_argument("--worst_vis", type=int, default=20)
    ap.add_argument("--median_vis", type=int, default=10)
    ap.add_argument("--best_vis", type=int, default=10)
    ap.add_argument(
        "--rank_metric",
        choices=["rmse_m_core_exact_pixel", "rmse_m_core_patch",
                 "rmse_m_full_exact_pixel", "rmse_m_full_prediction_patch"],
        default="rmse_m_core_exact_pixel",
    )
    args = ap.parse_args()

    _add_code_path(args.code_dir)
    import models_mae
    from util.dem_dataset import DEMLCCPairDataset

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    dataset = DEMLCCPairDataset(
        dem_dir=args.data_root,
        lcc_dir=args.lcc_mask_path,
        dem_list_path=args.list,
        lcc_list_path=args.lcc_list,
        input_size=args.input_size,
        nodata=args.nodata,
        nodata_threshold=args.nodata_threshold,
        random_flip=False,
        return_path=True,
        tile_norm=args.tile_norm,
        tile_norm_eps=args.tile_norm_eps,
        tile_norm_std_scale=args.tile_norm_std_scale,
        tile_norm_visible_only=args.tile_norm_visible_only,
        min_lcc_patch_ratio=args.min_prediction_patch_ratio,
        max_lcc_patch_ratio=args.max_prediction_patch_ratio,
        min_valid_visible_patch_ratio=args.min_valid_visible_patch_ratio,
        loss_region_mode=args.loss_region_mode,
        core_patch_radius=args.core_patch_radius,
        min_core_valid_patch_ratio=args.min_core_valid_patch_ratio,
        min_core_prediction_patch_ratio=args.min_core_prediction_patch_ratio,
        max_core_prediction_patch_ratio=args.max_core_prediction_patch_ratio,
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
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    message = model.load_state_dict(state, strict=False)
    print(f"[CKPT] loaded: {args.ckpt}")
    print(f"[CKPT] missing_keys={message.missing_keys}")
    print(f"[CKPT] unexpected_keys={message.unexpected_keys}")
    model.to(device)
    model.eval()

    rows: List[Dict[str, Any]] = []
    totals = {
        "core_patch_sse": 0.0, "core_patch_count": 0,
        "full_patch_sse": 0.0, "full_patch_count": 0,
        "core_exact_sse": 0.0, "core_exact_count": 0,
        "full_exact_sse": 0.0, "full_exact_count": 0,
    }

    global_index = 0
    for batch_index, batch in enumerate(loader):
        samples, meta, paths, lcc, valid = batch
        samples = samples.to(device, non_blocking=True)
        lcc = lcc.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            _, pred, loss_mask, prediction_mask = model(
                samples,
                mask_ratio=args.mask_ratio,
                lcc_mask=lcc,
                valid_mask=valid,
                loss_on_lcc_only=True,
                lcc_priority=args.lcc_priority,
                lcc_mask_mode="exact",
                lcc_patch_threshold=args.lcc_patch_threshold,
                loss_region_mode=args.loss_region_mode,
                core_patch_radius=args.core_patch_radius,
                return_aux_masks=True,
            )

        pred_img = model.unpatchify(pred).float()
        target_patch = model.patchify(samples).float()
        pred_patch = pred.float()
        patch_diff_norm = pred_patch - target_patch

        loss_img = _expand_patch_mask(model, loss_mask.float(), args.in_chans) > 0.5
        prediction_img = (
            _expand_patch_mask(model, prediction_mask.float(), args.in_chans) > 0.5
        )

        batch_size = samples.shape[0]
        for i in range(batch_size):
            mean_m = float(_meta_item(meta, "tile_mean_m", i, 0.0))
            std_safe = float(_meta_item(meta, "tile_std_safe", i, 1.0))
            path = paths[i] if isinstance(paths, (list, tuple)) else str(paths)

            x_norm = samples[i, 0].detach().float().cpu().numpy()
            pred_norm = pred_img[i, 0].detach().float().cpu().numpy()
            valid_np = valid[i, 0].detach().bool().cpu().numpy()
            lcc_np = lcc[i, 0].detach().bool().cpu().numpy()
            loss_img_np = loss_img[i, 0].detach().cpu().numpy()
            prediction_img_np = prediction_img[i, 0].detach().cpu().numpy()

            if args.tile_norm:
                gt_m = x_norm * std_safe + mean_m
                pred_m = pred_norm * std_safe + mean_m
            else:
                gt_m = x_norm
                pred_m = pred_norm
            err_m = pred_m - gt_m

            core_patch_pixel_mask = loss_img_np & valid_np
            full_patch_pixel_mask = prediction_img_np & valid_np
            core_exact_pixel_mask = core_patch_pixel_mask & lcc_np
            full_exact_pixel_mask = full_patch_pixel_mask & lcc_np

            core_patch = _masked_stats(err_m, core_patch_pixel_mask)
            full_patch = _masked_stats(err_m, full_patch_pixel_mask)
            core_exact = _masked_stats(err_m, core_exact_pixel_mask)
            full_exact = _masked_stats(err_m, full_exact_pixel_mask)

            totals["core_patch_sse"] += core_patch["sse"]
            totals["core_patch_count"] += core_patch["count"]
            totals["full_patch_sse"] += full_patch["sse"]
            totals["full_patch_count"] += full_patch["count"]
            totals["core_exact_sse"] += core_exact["sse"]
            totals["core_exact_count"] += core_exact["count"]
            totals["full_exact_sse"] += full_exact["sse"]
            totals["full_exact_count"] += full_exact["count"]

            row = {
                "index": global_index,
                "batch_index": batch_index,
                "path": str(path),
                "mask_path": _meta_item(meta, "lcc_path", i, ""),
                "file": Path(str(path)).name,
                "rmse_m_core_patch": core_patch["rmse"],
                "mae_m_core_patch": core_patch["mae"],
                "bias_m_core_patch": core_patch["bias"],
                "max_abs_m_core_patch": core_patch["max_abs"],
                "rmse_m_full_prediction_patch": full_patch["rmse"],
                "rmse_m_core_exact_pixel": core_exact["rmse"],
                "mae_m_core_exact_pixel": core_exact["mae"],
                "bias_m_core_exact_pixel": core_exact["bias"],
                "max_abs_m_core_exact_pixel": core_exact["max_abs"],
                "p95_abs_m_core_exact_pixel": core_exact["p95_abs"],
                "rmse_m_full_exact_pixel": full_exact["rmse"],
                "n_core_patch_pixels": core_patch["count"],
                "n_core_exact_pixels": core_exact["count"],
                "n_full_prediction_patch_pixels": full_patch["count"],
                "n_full_exact_pixels": full_exact["count"],
                "valid_pixel_ratio": float(_meta_item(meta, "valid_pixel_ratio", i, 0.0)),
                "prediction_patch_ratio": float(_meta_item(meta, "prediction_patch_ratio", i, 0.0)),
                "visible_valid_patch_ratio": float(_meta_item(meta, "visible_valid_patch_ratio", i, 0.0)),
                "ignored_patch_ratio": float(_meta_item(meta, "ignored_patch_ratio", i, 0.0)),
                "core_valid_patch_ratio": float(_meta_item(meta, "core_valid_patch_ratio", i, 0.0)),
                "core_prediction_patch_ratio": float(
                    _meta_item(meta, "core_prediction_patch_ratio", i, 0.0)
                ),
                "tile_mean_m": mean_m,
                "tile_std_safe": std_safe,
            }
            rows.append(row)
            global_index += 1

        if batch_index % 50 == 0:
            print(f"[EVAL] batch={batch_index}/{len(loader)} samples={global_index}")

    _write_csv(output / "per_tile_metrics.csv", rows)

    rank_key = args.rank_metric
    finite = [r for r in rows if _safe_float(r.get(rank_key)) is not None]
    worst = sorted(finite, key=lambda r: float(r[rank_key]), reverse=True)
    best = sorted(finite, key=lambda r: float(r[rank_key]))
    middle = _select_middle(rows, args.median_vis, rank_key)

    _write_csv(output / f"worst_by_{rank_key}.csv", worst)
    _write_csv(output / f"best_by_{rank_key}.csv", best)
    _write_csv(output / f"median_by_{rank_key}.csv", middle)

    def global_rmse(sse_key: str, count_key: str) -> Optional[float]:
        count = int(totals[count_key])
        return float(np.sqrt(totals[sse_key] / count)) if count > 0 else None

    summary = {
        "split_name": args.split_name,
        "checkpoint": args.ckpt,
        "n_tiles": len(rows),
        "rank_metric": rank_key,
        "global_pixel_weighted_rmse_m": {
            "core_patch": global_rmse("core_patch_sse", "core_patch_count"),
            "full_prediction_patch": global_rmse("full_patch_sse", "full_patch_count"),
            "core_exact_final_mask_pixel": global_rmse(
                "core_exact_sse", "core_exact_count"
            ),
            "full_exact_final_mask_pixel": global_rmse(
                "full_exact_sse", "full_exact_count"
            ),
        },
        "global_counts": totals,
        "per_tile_summary": {
            "rmse_m_core_patch": _summary([r["rmse_m_core_patch"] for r in rows]),
            "rmse_m_core_exact_pixel": _summary(
                [r["rmse_m_core_exact_pixel"] for r in rows]
            ),
            "rmse_m_full_exact_pixel": _summary(
                [r["rmse_m_full_exact_pixel"] for r in rows]
            ),
            "rmse_m_full_prediction_patch": _summary(
                [r["rmse_m_full_prediction_patch"] for r in rows]
            ),
        },
    }
    with (output / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    _save_selected_visuals(
        model, dataset, worst[:args.worst_vis], args, device,
        folder=f"visuals_worst_by_{rank_key}", rank_key=rank_key,
    )
    _save_selected_visuals(
        model, dataset, middle[:args.median_vis], args, device,
        folder=f"visuals_median_by_{rank_key}", rank_key=rank_key,
    )
    _save_selected_visuals(
        model, dataset, best[:args.best_vis], args, device,
        folder=f"visuals_best_by_{rank_key}", rank_key=rank_key,
    )

    print(json.dumps(summary, indent=2))
    print(f"[DONE] {output}")


if __name__ == "__main__":
    main()
