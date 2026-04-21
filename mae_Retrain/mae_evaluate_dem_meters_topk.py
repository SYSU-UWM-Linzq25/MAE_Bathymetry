#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a MAE model on DEM GeoTIFF tiles (1-channel) tile-wise normalization during evaluation, consistent with training.

This is the DEM/GeoTIFF replacement of:
  - mae_evaluate.py (JPG + ImageNet norm)
  - mae_evaluate_meters_topk.py (JPG + ImageNet norm + per-tile meters)

What it does:
  * reads DEM GeoTIFF tiles via util.dem_dataset.DEMTileDataset (same reader as training)
  * norm_json kept only for record/fallback, not as primary input normalization (TRAIN stats)
  * runs the MAE forward with the requested mask_ratio
  * reports RMSE in meters:
      - rmse_m_mask: on masked patches (same region as loss)
      - rmse_m_all : on pasted full tile (keep patches copied from target)
  * writes metrics.csv + summary.json + topk worst CSV

Example:
  python mae_evaluate_dem_meters_topk.py \
    --ckpt runs/Small_meanstd_336/checkpoint-best.pth \
    --list splits/smoke_small_1000/global/holdout_KY.txt \
    --norm_json runs/Small_meanstd_336/norm_stats_train.json \
    --output_dir runs/Small_meanstd_336/eval_KY \
    --mask_ratio 0.75 \
    --topk 200

Notes:
  - This script uses whatever masking strategy is implemented inside models_mae.
    In your current models_mae.py, forward_encoder() calls middle_masking() unconditionally.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import models_mae
from util.dem_dataset import DEMTileDataset, load_json


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _meta_to_tile_std_tensor(meta, device, dtype=torch.float32):
    if meta is None:
        raise ValueError("meta must be provided for tile-wise evaluation")
    if isinstance(meta, dict):
        vals = meta["tile_std_safe"]
        if torch.is_tensor(vals):
            return vals.to(device=device, dtype=dtype)
        return torch.as_tensor(vals, device=device, dtype=dtype)
    raise TypeError(f"Unsupported meta type: {type(meta)}")

@torch.no_grad()
def rmse_meters_per_sample(model, samples, pred, mask, meta):
    """
    Per-sample metrics using tile-wise normalization metadata.

    Returns
    -------
    mse_norm_mask : torch.Tensor, shape [N]
        Masked-only MSE in normalized/model space
    rmse_m_mask   : torch.Tensor, shape [N]
        Masked-only RMSE in meters
    rmse_m_all    : torch.Tensor, shape [N]
        Pasted full-tile RMSE in meters
    """
    if meta is None:
        raise ValueError("meta must be provided for tile-wise evaluation")

    target = model.patchify(samples)  # [N, L, P]

    pred_f = pred.float()
    target_f = target.float()

    keep = (mask == 0)  # [N, L]
    pred_paste = pred_f.clone()
    pred_paste[keep] = target_f[keep]

    mse_patch = ((pred_paste - target_f) ** 2).mean(dim=-1)  # [N, L]
    mask_f = mask.float()
    denom = mask_f.sum(dim=1).clamp(min=1.0)

    mse_norm_mask = (mse_patch * mask_f).sum(dim=1) / denom
    rmse_norm_mask = torch.sqrt(mse_norm_mask)
    rmse_norm_all = torch.sqrt(mse_patch.mean(dim=1))

    tile_std = _meta_to_tile_std_tensor(meta, device=samples.device, dtype=rmse_norm_mask.dtype).view(-1)
    rmse_m_mask = rmse_norm_mask * tile_std
    rmse_m_all = rmse_norm_all * tile_std

    return mse_norm_mask, rmse_m_mask, rmse_m_all

def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def build_model(arch, img_size, in_chans, norm_pix_loss, bottleneck_norm="none", loss_mode="mse"):
    return models_mae.__dict__[arch](
        img_size=img_size,
        in_chans=in_chans,
        norm_pix_loss=norm_pix_loss,
        bottleneck_norm=bottleneck_norm,
        loss_mode=loss_mode,
    )


def load_ckpt(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected

def quantile_map_from_visible(pred_patch, target_patch, mask, n_q=101):
    """
    pred_patch, target_patch: [L, P]
    mask: [L], 0=keep/visible, 1=masked
    returns corrected pred_patch in the same space
    """
    keep = (mask == 0)
    if keep.sum() == 0:
        return pred_patch

    pred_vis = pred_patch[keep].reshape(-1).detach().cpu().numpy()
    gt_vis   = target_patch[keep].reshape(-1).detach().cpu().numpy()

    if pred_vis.size < 16:
        return pred_patch

    q = np.linspace(0.0, 1.0, n_q)
    pred_q = np.quantile(pred_vis, q)
    gt_q   = np.quantile(gt_vis, q)

    # make pred_q strictly increasing for np.interp stability
    pred_q = np.maximum.accumulate(pred_q)

    pred_all = pred_patch.detach().cpu().numpy().reshape(-1)
    pred_corr = np.interp(pred_all, pred_q, gt_q, left=gt_q[0], right=gt_q[-1])

    pred_corr = torch.from_numpy(pred_corr).to(pred_patch.device, dtype=pred_patch.dtype)
    return pred_corr.view_as(pred_patch)

def _denorm_eval(x: torch.Tensor, meta: dict, norm: dict, use_tile_norm: bool) -> torch.Tensor:
    """
    x: [H,W] or [1,H,W] in model space
    """
    if use_tile_norm and isinstance(meta, dict):
        tile_mean = meta["tile_mean_m"]
        tile_std = meta["tile_std_safe"]
        if torch.is_tensor(tile_mean):
            tile_mean = tile_mean.item()
        if torch.is_tensor(tile_std):
            tile_std = tile_std.item()
        return x * float(tile_std) + float(tile_mean)

    method = str(norm.get("method", "meanstd")).lower()
    if method == "meanstd":
        return x * float(norm["std"]) + float(norm["mean"])
    else:
        vmin = float(norm["min"])
        vmax = float(norm["max"])
        return x * (vmax - vmin) + vmin


def _write_geotiff_like(ref_path: str, out_path: Path, arr2d, dtype: str = "float32", nodata=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import rasterio
        with rasterio.open(ref_path) as src:
            profile = src.profile.copy()

        h, w = arr2d.shape
        if profile.get("height", None) != h or profile.get("width", None) != w:
            import tifffile
            tifffile.imwrite(str(out_path), arr2d.astype(np.float32))
            return

        profile.update(
            driver="GTiff",
            count=1,
            dtype=dtype,
            nodata=nodata,
            compress=profile.get("compress", "LZW"),
        )

        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(arr2d, 1)

    except Exception:
        import tifffile
        tifffile.imwrite(str(out_path), arr2d.astype(np.float32))


def get_or_make_vis_indices(dataset, n: int, seed: int, save_path: str = "") -> np.ndarray:
    if save_path:
        save_path = str(save_path)
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        if sp.exists():
            idxs = np.loadtxt(sp, dtype=int)
            return np.atleast_1d(idxs)

    rng = np.random.RandomState(seed)
    n = min(int(n), len(dataset))
    idxs = rng.choice(len(dataset), size=n, replace=False).astype(int)

    if save_path:
        np.savetxt(save_path, idxs, fmt="%d")

    return idxs


def _apply_postproc_patch(pred_patch: torch.Tensor, target_patch: torch.Tensor, mask_patch: torch.Tensor, mode: str):
    """
    pred_patch, target_patch: [L,P]
    mask_patch: [L], 0=visible, 1=masked
    returns corrected pred_patch in model/normalized space
    """
    if mode == "median":
        keep_patch = (mask_patch == 0)
        if keep_patch.sum() > 0:
            vis_bias = (pred_patch[keep_patch] - target_patch[keep_patch]).reshape(-1).median()
        else:
            vis_bias = torch.tensor(0.0, device=pred_patch.device, dtype=pred_patch.dtype)
        return pred_patch - vis_bias

    elif mode == "histogram":
        return quantile_map_from_visible(pred_patch, target_patch, mask_patch)

    else:
        return pred_patch


@torch.no_grad()
def save_fixed_vis_tifs(model, dataset, idxs: np.ndarray, device, args, out_dir: Path, norm: dict):
    """
    Save a fixed set of GeoTIFF visualizations for evaluation.
    Output:
      - gt_m.tif
      - pred_m.tif
      - recon_m.tif
      - err_m.tif
      - mask.tif
      - pred_post_m.tif / recon_post_m.tif / err_post_m.tif (if postproc != none)
    """
    out_vis = out_dir / "vis_tif"
    out_vis.mkdir(parents=True, exist_ok=True)

    model.eval()

    for idx in idxs.tolist():
        sample = dataset[int(idx)]

        if isinstance(sample, (tuple, list)):
            if len(sample) == 3:
                x, meta, ref_path = sample
            elif len(sample) == 2:
                x, meta = sample
                ref_path = meta["path"] if isinstance(meta, dict) and "path" in meta else dataset.files[int(idx)]
            else:
                x = sample[0]
                meta = None
                ref_path = dataset.files[int(idx)]
        else:
            x = sample
            meta = None
            ref_path = dataset.files[int(idx)]

        x = x.unsqueeze(0).to(device, non_blocking=True)  # [1,1,H,W]

        # fixed mask per tile for reproducible eval visualization
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        vis_seed = int(args.seed + 100000 + int(idx))
        torch.manual_seed(vis_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(vis_seed)

        if device.type == "cuda" and args.amp:
            with torch.cuda.amp.autocast():
                _, pred, mask = model(x, mask_ratio=args.mask_ratio)
        else:
            _, pred, mask = model(x, mask_ratio=args.mask_ratio)

        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

        pred_img = model.unpatchify(pred)[0, 0].float().cpu()
        p = model.patch_embed.patch_size[0]
        mask_img = mask.unsqueeze(-1).repeat(1, 1, p * p * args.in_chans)
        mask_img = model.unpatchify(mask_img)[0, 0].float().cpu()   # [H,W], 1=masked
        x0 = x[0, 0].float().cpu()

        recon = x0 * (1 - mask_img) + pred_img * mask_img

        target_patch = model.patchify(x)[0].float().cpu()
        pred_patch = pred[0].float().cpu()
        mask_patch = mask[0].float().cpu()

        pred_patch_post = _apply_postproc_patch(pred_patch, target_patch, mask_patch, args.postproc)
        pred_post_img = model.unpatchify(pred_patch_post.unsqueeze(0).to(device))[0, 0].float().cpu()
        recon_post = x0 * (1 - mask_img) + pred_post_img * mask_img

        gt_m = _denorm_eval(x0, meta, norm, args.tile_norm).cpu()
        pred_m = _denorm_eval(pred_img, meta, norm, args.tile_norm).cpu()
        recon_m = _denorm_eval(recon, meta, norm, args.tile_norm).cpu()

        pred_post_m = _denorm_eval(pred_post_img, meta, norm, args.tile_norm).cpu()
        recon_post_m = _denorm_eval(recon_post, meta, norm, args.tile_norm).cpu()

        err_m = (recon_m - gt_m).cpu()
        err_post_m = (recon_post_m - gt_m).cpu()

        gt_np = gt_m.numpy().astype(np.float32, copy=False)
        pred_np = pred_m.numpy().astype(np.float32, copy=False)
        recon_np = recon_m.numpy().astype(np.float32, copy=False)
        err_np = err_m.numpy().astype(np.float32, copy=False)
        mask_np = (mask_img.numpy() > 0.5).astype(np.uint8)
        mask_mean = float(mask_np.mean())
        print(f"[VIS_MASK] idx={int(idx)} mask_mean={mask_mean:.4f}")
        
        pred_post_np = pred_post_m.numpy().astype(np.float32, copy=False)
        recon_post_np = recon_post_m.numpy().astype(np.float32, copy=False)
        err_post_np = err_post_m.numpy().astype(np.float32, copy=False)

        base = f"idx{int(idx):06d}"
        _write_geotiff_like(ref_path, out_vis / f"{base}_gt_m.tif", gt_np, dtype="float32", nodata=None)
        _write_geotiff_like(ref_path, out_vis / f"{base}_pred_m.tif", pred_np, dtype="float32", nodata=None)
        _write_geotiff_like(ref_path, out_vis / f"{base}_recon_m.tif", recon_np, dtype="float32", nodata=None)
        _write_geotiff_like(ref_path, out_vis / f"{base}_err_m.tif", err_np, dtype="float32", nodata=None)
        _write_geotiff_like(ref_path, out_vis / f"{base}_mask.tif", mask_np, dtype="uint8", nodata=None)

        if args.postproc != "none":
            _write_geotiff_like(ref_path, out_vis / f"{base}_pred_post_m.tif", pred_post_np, dtype="float32", nodata=None)
            _write_geotiff_like(ref_path, out_vis / f"{base}_recon_post_m.tif", recon_post_np, dtype="float32", nodata=None)
            _write_geotiff_like(ref_path, out_vis / f"{base}_err_post_m.tif", err_post_np, dtype="float32", nodata=None)

@torch.no_grad()
def _rmse_meters_postproc_from_pred(model, samples, pred, mask, meta, mode="none"):
    target = model.patchify(samples)   # [N, L, P]
    pred_f = pred.float()
    target_f = target.float()

    keep_patch = (mask == 0)

    pred_corr = pred_f.clone()

    if mode == "median":
        bias_list = []
        for i in range(pred_f.shape[0]):
            ei = pred_f[i] - target_f[i]
            ki = keep_patch[i]
            if ki.sum() == 0:
                bias_list.append(torch.zeros((), device=pred_f.device, dtype=pred_f.dtype))
            else:
                bias_list.append(ei[ki].reshape(-1).median())
        bias = torch.stack(bias_list, dim=0)
        pred_corr = pred_f - bias[:, None, None]
        bias_m = bias * _meta_to_tile_std_tensor(meta, device=samples.device, dtype=pred_f.dtype).view(-1)

    elif mode == "histogram":
        bias_vals = []
        pred_corr_list = []
        for i in range(pred_f.shape[0]):
            pc = quantile_map_from_visible(pred_f[i], target_f[i], mask[i])
            pred_corr_list.append(pc)
            # histogram 不是单值 bias，这里可以放 NaN 占位
            bias_vals.append(torch.tensor(float("nan"), device=pred_f.device, dtype=pred_f.dtype))
        pred_corr = torch.stack(pred_corr_list, dim=0)
        bias_m = torch.stack(bias_vals, dim=0)

    else:
        bias_m = torch.full((pred_f.shape[0],), float("nan"), device=pred_f.device, dtype=pred_f.dtype)

    pred_paste = pred_corr.clone()
    pred_paste[keep_patch] = target_f[keep_patch]

    err = pred_paste - target_f
    tile_std = _meta_to_tile_std_tensor(meta, device=samples.device, dtype=err.dtype).view(-1)
    err_m = err * tile_std[:, None, None]

    mse_patch_m = (err_m ** 2).mean(dim=-1)
    mask_f = mask.float()
    denom = mask_f.sum(dim=1).clamp(min=1.0)

    rmse_m_mask_ps = torch.sqrt((mse_patch_m * mask_f).sum(dim=1) / denom)
    rmse_m_all_ps = torch.sqrt(mse_patch_m.mean(dim=1))

    return rmse_m_mask_ps, rmse_m_all_ps, bias_m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", required=True, help="TXT listing DEM GeoTIFF tiles")
    ap.add_argument("--norm_json", required=True, help="TRAIN norm_stats_train.json")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--arch", default="mae_vit_large_patch16")
    ap.add_argument("--input_size", type=int, default=336)
    ap.add_argument("--in_chans", type=int, default=1)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--nodata", type=float, default=-9999.0)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--norm_pix_loss", action="store_true")
    ap.add_argument("--tile_norm", action="store_true")
    ap.add_argument("--tile_norm_eps", type=float, default=1e-3)
    ap.add_argument("--bottleneck_norm", default="none", choices=["none", "inst1d"])
    ap.add_argument("--loss_mode", default="mse", choices=["mse"])
    ap.add_argument("--postproc", default="none", choices=["none", "median", "histogram"])
    ap.add_argument("--save_vis_tif", action="store_true", help="Save fixed-sample GeoTIFF visualizations during evaluation")
    ap.add_argument("--vis_n", type=int, default=10, help="Number of fixed samples to visualize")
    ap.add_argument("--vis_seed", type=int, default=42, help="Seed for fixed visualization sample selection")
    ap.add_argument("--vis_indices_txt", default="", help="Optional txt file to store/reuse fixed visualization indices across runs")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    norm = load_json(args.norm_json)
    method = str(norm.get("method", "meanstd")).lower()
    if method == "meanstd":
        a, b = float(norm["mean"]), float(norm["std"])
        scale_m = float(norm["std"])
    elif method == "minmax":
        a, b = float(norm["min"]), float(norm["max"])
        scale_m = float(norm["max"]) - float(norm["min"])
    else:
        raise ValueError(f"Unknown norm method in json: {method}")

    ds = DEMTileDataset(
        dir_path=None,
        list_path=args.list,
        input_size=args.input_size,
        nodata=args.nodata,
        random_flip=False,
        return_path=True,
        return_meta=True,
        tile_norm=args.tile_norm,
        tile_norm_eps=args.tile_norm_eps,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device)
    model = build_model(
        args.arch,
        args.input_size,
        args.in_chans,
        args.norm_pix_loss,
        bottleneck_norm=args.bottleneck_norm,
        loss_mode=args.loss_mode,
    )
    model.to(device)
    model.eval()

    missing, unexpected = load_ckpt(model, args.ckpt)
    with (out_dir / "load_report.json").open("w") as f:
        json.dump({"missing": missing, "unexpected": unexpected}, f, indent=2)

    rows = []
    rmse_mask_list: List[float] = []
    rmse_all_list: List[float] = []
    rmse_mask_viscorr_list: List[float] = []
    rmse_all_viscorr_list: List[float] = []
    bias_vis_med_list: List[float] = []

    for x, meta, paths in loader:
        x = x.to(device, non_blocking=True)

        if device.type == "cuda" and args.amp:
            with torch.cuda.amp.autocast():
                loss, pred, mask = model(x, mask_ratio=args.mask_ratio)
        else:
            loss, pred, mask = model(x, mask_ratio=args.mask_ratio)

        mse_norm_mask, rmse_m_mask, rmse_m_all = rmse_meters_per_sample(
            model, x, pred, mask, meta=meta
        )

        if args.postproc == "none":
            rmse_m_mask_post_ps = rmse_m_mask
            rmse_m_all_post_ps = rmse_m_all
            bias_post_ps = torch.full_like(rmse_m_mask, float("nan"))
        elif args.postproc == "median":
            rmse_m_mask_post_ps, rmse_m_all_post_ps, bias_post_ps = _rmse_meters_postproc_from_pred(
                model, x, pred, mask, meta=meta, mode="median"
            )
        elif args.postproc == "histogram":
            rmse_m_mask_post_ps, rmse_m_all_post_ps, bias_post_ps = _rmse_meters_postproc_from_pred(
                model, x, pred, mask, meta=meta, mode="histogram"
            )

        for i in range(x.shape[0]):
            p = paths[i]
            rmm = float(rmse_m_mask[i].item())
            rma = float(rmse_m_all[i].item())
            rmse_mask_list.append(rmm)
            rmse_all_list.append(rma)
            rmm_vc = float(rmse_m_mask_post_ps[i].item())
            rma_vc = float(rmse_m_all_post_ps[i].item())
            bvm = float(bias_post_ps[i].item())

            rmse_mask_viscorr_list.append(rmm_vc)
            rmse_all_viscorr_list.append(rma_vc)
            bias_vis_med_list.append(bvm)

            rows.append({
                "path": p,
                "mse_norm_mask": float(mse_norm_mask[i].item()),
                "rmse_m_mask": rmm,
                "rmse_m_all": rma,
                "rmse_m_mask_viscorr": rmm_vc,
                "rmse_m_all_viscorr": rma_vc,
                "bias_m_vis_med": bvm,
            })

    # metrics.csv
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "mse_norm_mask",
                "rmse_m_mask",
                "rmse_m_all",
                "rmse_m_mask_viscorr",
                "rmse_m_all_viscorr",
                "bias_m_vis_med",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # summary.json
    summary = {
        "ckpt": os.path.abspath(args.ckpt),
        "list": os.path.abspath(args.list),
        "norm_json": os.path.abspath(args.norm_json),
        "mask_ratio": args.mask_ratio,
        "norm_method": method,
        "postproc": args.postproc,
        "rmse_m_mask": summarize(rmse_mask_list),
        "rmse_m_all": summarize(rmse_all_list),
        "rmse_m_mask_viscorr": summarize(rmse_mask_viscorr_list),
        "rmse_m_all_viscorr": summarize(rmse_all_viscorr_list),
        "bias_m_vis_med": summarize(bias_vis_med_list),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # topk worst
    topk = int(args.topk)
    if topk > 0:
        rows_sorted = sorted(rows, key=lambda r: r["rmse_m_mask"], reverse=True)
        rows_top = rows_sorted[:topk]
        top_path = out_dir / f"topk_worst_rmse_mask_{topk}.csv"
        with top_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "path",
                    "rmse_m_mask",
                    "rmse_m_all",
                    "rmse_m_mask_viscorr",
                    "rmse_m_all_viscorr",
                    "bias_m_vis_med",
                    "mse_norm_mask",
                ],
            )
            w.writeheader()
            for r in rows_top:
                w.writerow({
                    "path": r["path"],
                    "rmse_m_mask": r["rmse_m_mask"],
                    "rmse_m_all": r["rmse_m_all"],
                    "rmse_m_mask_viscorr": r["rmse_m_mask_viscorr"],
                    "rmse_m_all_viscorr": r["rmse_m_all_viscorr"],
                    "bias_m_vis_med": r["bias_m_vis_med"],
                    "mse_norm_mask": r["mse_norm_mask"],
                })

    # fixed-sample GeoTIFF visualization
    if args.save_vis_tif and args.vis_n > 0:
        vis_idx_path = args.vis_indices_txt if args.vis_indices_txt else str(out_dir / "vis_indices.txt")
        idxs = get_or_make_vis_indices(ds, args.vis_n, args.vis_seed, vis_idx_path)
        save_fixed_vis_tifs(model, ds, idxs, device, args, out_dir, norm)

    print("[DONE]", csv_path)
    print("[DONE]", out_dir / "summary.json")
    if topk > 0:
        print("[DONE]", out_dir / f"topk_worst_rmse_mask_{topk}.csv")


if __name__ == "__main__":
    main()
