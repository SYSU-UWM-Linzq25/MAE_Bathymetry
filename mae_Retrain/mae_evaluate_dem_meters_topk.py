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

@torch.no_grad()
def _rmse_meters_visible_median_bias_from_pred(model, samples, pred, mask, meta):
    """
    Per-sample visible/unmask median-bias correction metrics.

    Returns
    -------
    rmse_m_mask_viscorr_ps : torch.Tensor, shape [N]
    rmse_m_all_viscorr_ps  : torch.Tensor, shape [N]
    bias_m_vis_med_ps      : torch.Tensor, shape [N]
    """
    if meta is None:
        raise ValueError("meta must be provided for tile-wise evaluation")

    target = model.patchify(samples)   # [N, L, P]
    pred_f = pred.float()
    target_f = target.float()

    keep_patch = (mask == 0)           # [N, L]
    e = pred_f - target_f              # [N, L, P]

    # 1) visible median bias per sample in normalized/model space
    bias_list = []
    for i in range(e.shape[0]):
        ei = e[i]                      # [L, P]
        ki = keep_patch[i]             # [L]
        if ki.sum() == 0:
            bias_list.append(torch.zeros((), device=e.device, dtype=e.dtype))
        else:
            vals = ei[ki].reshape(-1)
            bias_list.append(vals.median())

    bias = torch.stack(bias_list, dim=0)   # [N]

    # 2) apply correction in normalized/model space
    pred_corr = pred_f - bias[:, None, None]
    pred_paste = pred_corr.clone()
    pred_paste[keep_patch] = target_f[keep_patch]

    err = pred_paste - target_f            # [N, L, P]

    # 3) convert to meters per tile
    tile_std = _meta_to_tile_std_tensor(meta, device=samples.device, dtype=err.dtype).view(-1)  # [N]
    err_m = err * tile_std[:, None, None]
    bias_m = bias * tile_std               # [N]

    mse_patch_m = (err_m ** 2).mean(dim=-1)    # [N, L]
    mask_f = mask.float()
    denom = mask_f.sum(dim=1).clamp(min=1.0)

    rmse_m_mask_viscorr_ps = torch.sqrt((mse_patch_m * mask_f).sum(dim=1) / denom)   # [N]
    rmse_m_all_viscorr_ps = torch.sqrt(mse_patch_m.mean(dim=1))                       # [N]
    bias_m_vis_med_ps = bias_m                                                        # [N]

    return rmse_m_mask_viscorr_ps, rmse_m_all_viscorr_ps, bias_m_vis_med_ps
    
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

        rmse_m_mask_viscorr_ps, rmse_m_all_viscorr_ps, bias_m_vis_med_ps = _rmse_meters_visible_median_bias_from_pred(
            model, x, pred, mask, meta=meta
        )

        for i in range(x.shape[0]):
            p = paths[i]
            rmm = float(rmse_m_mask[i].item())
            rma = float(rmse_m_all[i].item())
            rmse_mask_list.append(rmm)
            rmse_all_list.append(rma)
            rmm_vc = float(rmse_m_mask_viscorr_ps[i].item())
            rma_vc = float(rmse_m_all_viscorr_ps[i].item())
            bvm = float(bias_m_vis_med_ps[i].item())

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
                    "mse_norm_mask": r["mse_norm_mask"],
                })

    print("[DONE]", csv_path)
    print("[DONE]", out_dir / "summary.json")
    if topk > 0:
        print("[DONE]", out_dir / f"topk_worst_rmse_mask_{topk}.csv")


if __name__ == "__main__":
    main()
