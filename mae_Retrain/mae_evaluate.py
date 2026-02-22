#!/usr/bin/env python3
import os
import glob
import csv
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

import models_mae  # 需要在 MAE repo 里运行，确保能 import


# =============== Keep consistent with mae_visualize.py ===============
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def imagenet_norm(img01_hwc):
    """img in [0,1], HWC3 -> HWC3 normalized"""
    return (img01_hwc - IMAGENET_MEAN) / IMAGENET_STD


def imagenet_denorm(img_norm_hwc):
    """inverse: HWC3 normalized -> HWC3 in [0,1] (clipped)"""
    out = img_norm_hwc * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(out, 0.0, 1.0)


def load_one_jpg(path, img_size=336):
    """Strictly follow mae_visualize.py: read RGB, resize, /255, imagenet norm, CHW tensor"""
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC [0,1]
    arr = imagenet_norm(arr)  # HWC normalized
    x = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
    return x


def prepare_model(chkpt_path, arch="mae_vit_large_patch16", img_size=336, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    model = models_mae.__dict__[arch](img_size=img_size)
    ckpt = torch.load(chkpt_path, map_location="cpu")

    # ckpt structure: usually {"model": state_dict, ...}
    if "model" in ckpt:
        msg = model.load_state_dict(ckpt["model"], strict=False)
    else:
        msg = model.load_state_dict(ckpt, strict=False)

    print("[ckpt load]", msg)

    model.to(device)
    model.eval()
    return model, device


@torch.no_grad()
def eval_batch(model, device, x, mask_ratio=0.35):
    """
    x: N,3,H,W (imagenet normalized)
    return:
      x01_c0, y01_c0, paste01_c0, mask_pix_bool (N,H,W)
    All in [0,1] space for metrics (channel0).
    """
    # forward MAE
    loss, y, mask = model(x.float(), mask_ratio=mask_ratio)  # y: N,L,p*p*3 ; mask: N,L
    y = model.unpatchify(y)  # N,3,H,W  (still imagenet normalized)

    # pixel mask (same as mae_visualize.py)
    p = model.patch_embed.patch_size[0]
    mask_pix = mask.unsqueeze(-1).repeat(1, 1, p * p * 3)
    mask_pix = model.unpatchify(mask_pix)  # N,3,H,W
    mask_pix = mask_pix[:, 0, :, :] > 0.5  # N,H,W bool

    # paste
    paste = x * (~mask_pix).unsqueeze(1) + y * mask_pix.unsqueeze(1)  # N,3,H,W (normalized)

    # denorm to [0,1]
    def denorm_to01(t):
        # t: N,3,H,W normalized
        tn = t.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # NHWC
        tn = imagenet_denorm(tn)  # NHWC in [0,1]
        return tn  # NHWC

    x01 = denorm_to01(x)
    y01 = denorm_to01(y)
    p01 = denorm_to01(paste)

    # use channel0 (all 3 channels identical in your DEM JPG, so any channel works)
    x01_c0 = x01[..., 0]  # N,H,W
    y01_c0 = y01[..., 0]
    p01_c0 = p01[..., 0]

    return x01_c0, y01_c0, p01_c0, mask_pix.cpu().numpy()


def metrics_for_one(x01, y01, p01, mask_pix):
    """
    x01,y01,p01: H,W in [0,1]
    mask_pix: H,W bool where True = masked region
    """
    err_pred = (y01 - x01)
    err_paste = (p01 - x01)

    # masked-only (main)
    m = mask_pix
    e_m = err_pred[m]
    rmse_m = float(np.sqrt(np.mean(e_m**2))) if e_m.size else np.nan
    mae_m  = float(np.mean(np.abs(e_m)))     if e_m.size else np.nan
    bias_m = float(np.mean(e_m))             if e_m.size else np.nan

    # full-image
    e_f = err_paste.reshape(-1)
    rmse_f = float(np.sqrt(np.mean(e_f**2)))
    mae_f  = float(np.mean(np.abs(e_f)))

    return rmse_m, mae_m, bias_m, rmse_f, mae_f


def summarize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "p95": np.nan}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def main():
    ap = argparse.ArgumentParser("MAE evaluation on KY sampled DEM JPG patches (336)")
    ap.add_argument("--img_dir",
                    default="/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Model_Evaluation/KY_eval_336",
                    help="Directory containing sampled jpg patches")
    ap.add_argument("--ckpt", required=True, help="MAE checkpoint path (.pth)")
    ap.add_argument("--arch", default="mae_vit_large_patch16")
    ap.add_argument("--img_size", type=int, default=336)
    ap.add_argument("--mask_ratio", type=float, default=0.35)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1, help="Random seed for mask sampling")
    ap.add_argument("--out_csv",
                    default="/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Model_Evaluation/metrics/ky_mae_metrics.csv")
    ap.add_argument("--max_files", type=int, default=0, help="0=all; otherwise limit number of jpgs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    model, device = prepare_model(args.ckpt, arch=args.arch, img_size=args.img_size, device=args.device)

    files = sorted(glob.glob(os.path.join(args.img_dir, "**", "*.jpg"), recursive=True))
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]
    if len(files) == 0:
        raise RuntimeError(f"No jpg found in {args.img_dir}")

    print(f"[Input] {len(files)} jpg patches")
    print(f"[Eval ] arch={args.arch} img={args.img_size} mask_ratio={args.mask_ratio} batch={args.batch_size} seed={args.seed}")
    print(f"[CSV  ] {args.out_csv}")

    # CSV header
    rows = []
    rmse_m_all, mae_m_all, bias_m_all, rmse_f_all, mae_f_all = [], [], [], [], []

    for i in tqdm(range(0, len(files), args.batch_size)):
        batch_files = files[i:i + args.batch_size]
        x = torch.stack([load_one_jpg(f, img_size=args.img_size) for f in batch_files], dim=0).to(device)

        x01_c0, y01_c0, p01_c0, mask_pix = eval_batch(model, device, x, mask_ratio=args.mask_ratio)

        for k, f in enumerate(batch_files):
            rmse_m, mae_m, bias_m, rmse_f, mae_f = metrics_for_one(
                x01_c0[k], y01_c0[k], p01_c0[k], mask_pix[k]
            )
            rows.append([os.path.basename(f), rmse_m, mae_m, bias_m, rmse_f, mae_f])

            rmse_m_all.append(rmse_m)
            mae_m_all.append(mae_m)
            bias_m_all.append(bias_m)
            rmse_f_all.append(rmse_f)
            mae_f_all.append(mae_f)

    # write csv
    with open(args.out_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "rmse_masked_0to1", "mae_masked_0to1", "bias_masked_0to1", "rmse_full_0to1", "mae_full_0to1"])
        w.writerows(rows)

    # summary
    s_rmse_m = summarize(rmse_m_all)
    s_mae_m  = summarize(mae_m_all)
    s_bias_m = summarize(bias_m_all)
    s_rmse_f = summarize(rmse_f_all)
    s_mae_f  = summarize(mae_f_all)

    print("\n=== SUMMARY (in 0~1 intensity space; multiply by 255 for 0~255) ===")
    print("[Masked] RMSE:", s_rmse_m, "  MAE:", s_mae_m, "  BIAS:", s_bias_m)
    print("[Full  ] RMSE:", s_rmse_f, "  MAE:", s_mae_f)
    print("\nDone.")


if __name__ == "__main__":
    main()

