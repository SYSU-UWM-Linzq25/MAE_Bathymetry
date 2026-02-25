#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import glob
import time
import heapq
import random
import argparse

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ImageNet normalize (same as training/eval)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def parse_patch_name(jpg_path: str):
    """
    支持命名:
      <tif_stem>_patch_<i>_<j>.jpg
      <tif_stem>patch_<i>_<j>.jpg
    返回: tif_stem, patch_i, patch_j, base_name
    """
    base = os.path.basename(jpg_path)
    stem = os.path.splitext(base)[0]
    m = re.search(r"patch_(\d+)_(\d+)$", stem)
    if not m:
        raise ValueError(f"Bad tile name (no patch_i_j): {base}")
    pi = int(m.group(1))
    pj = int(m.group(2))
    tif_stem = stem[:m.start()].rstrip("_-")  # 去掉末尾多余 "_" "-"
    return tif_stem, pi, pj, base


def load_minmax_csv(path: str):
    """
    Expect header: tif_stem,min,max,range
    range 可缺省（用 max-min）
    return dict: stem -> (vmin, vmax, vrange)
    """
    out = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            stem = row["tif_stem"]
            vmin = float(row["min"])
            vmax = float(row["max"])
            if "range" in row and row["range"] not in (None, "", "nan", "NaN"):
                vr = float(row["range"])
            else:
                vr = vmax - vmin
            out[stem] = (vmin, vmax, vr)
    return out


def denorm_0to1(x: torch.Tensor) -> torch.Tensor:
    """undo ImageNet normalize -> clamp to [0,1]"""
    return (x * IMAGENET_STD.to(x.device) + IMAGENET_MEAN.to(x.device)).clamp(0, 1)


def mask_to_pixels(model, mask):
    """
    mask: [N, L] (0=visible, 1=masked)
    -> mask_img: [N,3,H,W] with 0/1 blocks
    """
    p = model.patch_embed.patch_size[0]
    mask_img = mask.unsqueeze(-1).repeat(1, 1, p * p * 3)
    mask_img = model.unpatchify(mask_img)
    return mask_img


def save_tiff_float32(out_path: str, arr_2d: np.ndarray):
    """
    输出无地理信息 float32 tif（用于人工检查），依赖 tifffile（纯 python，推荐）
    """
    import tifffile
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tifffile.imwrite(out_path, np.asarray(arr_2d, dtype=np.float32))


class JpgDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.tfm = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = Image.open(p).convert("RGB")
        x = self.tfm(im)
        return x, p


@torch.no_grad()
def forward_batch(model, imgs, mask_ratio: float):
    loss, pred, mask = model(imgs, mask_ratio=mask_ratio)
    y = model.unpatchify(pred)  # [N,3,H,W]
    return y, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ckpt", dest="checkpoint", help="alias of --checkpoint")

    ap.add_argument("--model", default="mae_vit_large_patch16")
    ap.add_argument("--arch", dest="model", help="alias of --model")

    ap.add_argument("--img_size", type=int, default=336)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--minmax_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--export_topk_tiffs", action="store_true")
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N tiles (0=all)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed, deterministic=args.deterministic)

    # collect jpgs
    jpgs = sorted(glob.glob(os.path.join(args.img_dir, "**", "*.jpg"), recursive=True))
    if args.limit and args.limit > 0:
        jpgs = jpgs[:args.limit]
    if not jpgs:
        raise SystemExit(f"No jpg found under: {args.img_dir}")

    # load minmax
    minmax = load_minmax_csv(args.minmax_csv)

    # import models_mae from current working directory or script directory
    try:
        import models_mae
    except Exception:
        import sys
        sys.path.insert(0, os.getcwd())
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import models_mae

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # build model
    try:
        model = models_mae.__dict__[args.model](img_size=args.img_size, norm_pix_loss=False)
    except TypeError:
        model = models_mae.__dict__[args.model](norm_pix_loss=False)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # dataset/loader
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = JpgDataset(jpgs, tfm)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    metrics_csv = os.path.join(args.out_dir, "metrics_all.csv")
    worst_csv   = os.path.join(args.out_dir, f"worst{args.topk}.csv")
    summary_js  = os.path.join(args.out_dir, "summary.json")
    missing_txt = os.path.join(args.out_dir, "missing_minmax.txt")

    header = [
        "jpg_path","tif_stem","patch_i","patch_j",
        "min","max","range",
        "rmse_masked_0to1","mae_masked_0to1","bias_masked_0to1",
        "rmse_full_0to1","mae_full_0to1",
        "rmse_masked_m","mae_masked_m","bias_masked_m",
        "rmse_full_m","mae_full_m"
    ]

    # topk heap by rmse_masked_m
    heap = []      # (rmse_m, counter, row)
    counter = 0

    # for exporting (avoid re-run)
    topk_cache = {}  # jpg_path -> (x_m, p_m, d_m, stem0)

    rmse_m_list = []
    missing = []

    t0 = time.time()

    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for step, (imgs, paths) in enumerate(loader, 1):
            imgs = imgs.to(device, non_blocking=True)

            y, mask = forward_batch(model, imgs, args.mask_ratio)

            x01 = denorm_0to1(imgs)
            y01 = denorm_0to1(y)

            mask_img = mask_to_pixels(model, mask)               # [N,3,H,W] 0/1 block
            mask_pix = (mask_img[:,0,:,:] > 0.5).float()         # [N,H,W] 1=masked

            # masked-only metrics on channel 0
            diff_y = (y01[:,0,:,:] - x01[:,0,:,:])               # [N,H,W]
            denom = mask_pix.sum(dim=(1,2)).clamp_min(1.0)
            mse_mask = ((diff_y*diff_y) * mask_pix).sum(dim=(1,2)) / denom
            rmse_mask_01 = torch.sqrt(mse_mask)
            mae_mask_01  = (diff_y.abs() * mask_pix).sum(dim=(1,2)) / denom
            bias_mask_01 = (diff_y * mask_pix).sum(dim=(1,2)) / denom

            # full pasted image metrics
            p01 = x01 * (1.0 - mask_img) + y01 * mask_img
            diff_p = (p01[:,0,:,:] - x01[:,0,:,:])
            rmse_full_01 = torch.sqrt((diff_p*diff_p).mean(dim=(1,2)))
            mae_full_01  = diff_p.abs().mean(dim=(1,2))

            rmse_mask_01_cpu = rmse_mask_01.detach().cpu().numpy()
            mae_mask_01_cpu  = mae_mask_01.detach().cpu().numpy()
            bias_mask_01_cpu = bias_mask_01.detach().cpu().numpy()
            rmse_full_01_cpu = rmse_full_01.detach().cpu().numpy()
            mae_full_01_cpu  = mae_full_01.detach().cpu().numpy()

            for k, jpg_path in enumerate(paths):
                jpg_path = str(jpg_path)
                try:
                    tif_stem, pi, pj, base = parse_patch_name(jpg_path)
                except Exception:
                    tif_stem, pi, pj, base = "", -1, -1, os.path.basename(jpg_path)

                if tif_stem not in minmax:
                    missing.append(jpg_path)
                    continue

                vmin, vmax, vr = minmax[tif_stem]

                rmse_m = float(rmse_mask_01_cpu[k] * vr)
                mae_m  = float(mae_mask_01_cpu[k]  * vr)
                bias_m = float(bias_mask_01_cpu[k] * vr)
                rmse_full_m = float(rmse_full_01_cpu[k] * vr)
                mae_full_m  = float(mae_full_01_cpu[k]  * vr)

                row = [
                    jpg_path, tif_stem, pi, pj,
                    f"{vmin:.6f}", f"{vmax:.6f}", f"{vr:.6f}",
                    f"{rmse_mask_01_cpu[k]:.6f}", f"{mae_mask_01_cpu[k]:.6f}", f"{bias_mask_01_cpu[k]:.6f}",
                    f"{rmse_full_01_cpu[k]:.6f}", f"{mae_full_01_cpu[k]:.6f}",
                    f"{rmse_m:.6f}", f"{mae_m:.6f}", f"{bias_m:.6f}",
                    f"{rmse_full_m:.6f}", f"{mae_full_m:.6f}",
                ]
                w.writerow(row)
                rmse_m_list.append(rmse_m)

                # update heap (keep topk worst by rmse_m)
                counter += 1
                item = (rmse_m, counter, row)

                def cache_export():
                    if not args.export_topk_tiffs:
                        return
                    stem0 = os.path.splitext(base)[0]
                    x_m = (x01[k,0].detach().cpu().numpy() * vr + vmin).astype(np.float32)
                    p_m = (p01[k,0].detach().cpu().numpy() * vr + vmin).astype(np.float32)
                    d_m = (p_m - x_m).astype(np.float32)
                    topk_cache[jpg_path] = (x_m, p_m, d_m, stem0)

                if len(heap) < args.topk:
                    heapq.heappush(heap, item)
                    cache_export()
                else:
                    if rmse_m > heap[0][0]:
                        old = heapq.heapreplace(heap, item)
                        old_path = old[2][0]
                        topk_cache.pop(old_path, None)
                        cache_export()

            if args.print_every and (step % args.print_every == 0):
                done = min(step * args.batch_size, len(jpgs))
                dt = time.time() - t0
                spd = done / max(dt, 1e-6)
                print(f"[batch {step}] {done}/{len(jpgs)} tiles | {spd:.2f} tiles/s")

    # missing list
    if missing:
        with open(missing_txt, "w") as f:
            for p in missing:
                f.write(p + "\n")

    # write worst csv
    heap_sorted = sorted(heap, key=lambda x: x[0], reverse=True)
    with open(worst_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for _, _, row in heap_sorted:
            w.writerow(row)

    # summary
    arr = np.array(rmse_m_list, dtype=float)
    summary = {
        "img_dir": args.img_dir,
        "checkpoint": args.checkpoint,
        "model": args.model,
        "img_size": args.img_size,
        "mask_ratio": args.mask_ratio,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "n_tiles_total": int(len(jpgs)),
        "n_tiles_with_minmax": int(arr.size),
        "n_missing_minmax": int(len(missing)),
        "rmse_masked_m_mean": float(arr.mean()) if arr.size else None,
        "rmse_masked_m_median": float(np.median(arr)) if arr.size else None,
        "rmse_masked_m_p95": float(np.percentile(arr, 95)) if arr.size else None,
        "rmse_masked_m_max": float(arr.max()) if arr.size else None,
        "metrics_csv": metrics_csv,
        "worst_csv": worst_csv,
        "missing_minmax_txt": missing_txt if missing else None,
    }
    with open(summary_js, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] metrics: {metrics_csv}")
    print(f"[OK] worst{args.topk}: {worst_csv}")
    print(f"[OK] summary: {summary_js}")
    if missing:
        print(f"[WARN] missing minmax: {len(missing)} -> {missing_txt}")

    # export worst topk tiffs (meters) without re-run
    if args.export_topk_tiffs:
        out_root = os.path.join(args.out_dir, f"worst{args.topk}_tiles_meters")
        os.makedirs(out_root, exist_ok=True)

        try:
            import tifffile  # noqa
        except Exception:
            raise SystemExit("Need tifffile for --export_topk_tiffs. Install: pip install tifffile")

        for _, _, row in heap_sorted:
            jpg_path = row[0]
            if jpg_path not in topk_cache:
                continue
            x_m, p_m, d_m, stem0 = topk_cache[jpg_path]
            save_tiff_float32(os.path.join(out_root, "input_m", f"{stem0}_input_m.tif"), x_m)
            save_tiff_float32(os.path.join(out_root, "pred_m",  f"{stem0}_pred_m.tif"),  p_m)
            save_tiff_float32(os.path.join(out_root, "diff_m",  f"{stem0}_diff_m.tif"),  d_m)

        print(f"[OK] exported worst tiles to: {out_root}")


if __name__ == "__main__":
    main()
