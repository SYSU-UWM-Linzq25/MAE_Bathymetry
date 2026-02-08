#!/usr/bin/env python3
"""
MAE visualize for DEM + LCC_Mask with *LCC-first then random* masking (fill to mask_ratio),
and export outputs as GeoTIFF for easy downstream comparison.

What this script fixes vs the older "paste" version:
1) Mask is constructed from each tile's LCC mask (patch-level) + random non-LCC patches to reach mask_ratio.
2) The MAE forward pass uses this explicit mask (so visualization exactly matches the mask you intend).
3) Denormalization uses the same Normalize(mean/std) used in build_pretrain_transform (if available),
   otherwise falls back to values you provide.
4) Outputs are written as GeoTIFF (reconstruction and paste DEM), preserving georeference if input DEM is GeoTIFF.

Usage example:
python mae_visualize_lcc.py \
  --dem /path/to/tile_dem.tif \
  --lcc /path/to/tile_lcc_mask.tif \
  --ckpt /path/to/checkpoint.pth \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --mask_ratio 0.35 \
  --seed 123 \
  --out_dir ./viz_out \
  --save_png

Outputs in out_dir:
- input_dem.tif
- lcc_mask.tif
- patch_mask.tif              (1=masked at pixel level, nearest-neighbor upsampled)
- recon_dem.tif               (model reconstruction for all pixels; channel-0)
- paste_dem.tif               (visible patches from input + masked patches from reconstruction; channel-0)
- (optional) quicklook PNGs
"""
import argparse
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import re
import csv

def _extract_id_and_basin(stem: str):
    # e.g. Select_tile_Basin_1m_BadgerFinNull_ID22
    m_id = re.search(r"_ID(\d+)", stem)
    tid = m_id.group(1) if m_id else None
    m_bs = re.search(r"1m_([^_]+)_ID", stem)
    basin = m_bs.group(1) if m_bs else None
    return basin, tid

def iter_pairs(dem_dir: Path, lcc_dir: Path, recursive: bool):
    dem_files = dem_dir.rglob("*.tif") if recursive else dem_dir.glob("*.tif")
    lcc_index = list(lcc_dir.rglob("*.tif")) if recursive else list(lcc_dir.glob("*.tif"))

    for dem_path in sorted(dem_files):
        basin, tid = _extract_id_and_basin(dem_path.stem)
        if tid is None:
            continue

        # prioritize basin+ID match, then fallback to ID-only
        cand = []
        if basin is not None:
            pat = re.compile(rf"{re.escape(basin)}.*_ID{tid}.*LCC", re.IGNORECASE)
            cand = [p for p in lcc_index if pat.search(p.stem)]
        if not cand:
            pat = re.compile(rf"_ID{tid}.*LCC", re.IGNORECASE)
            cand = [p for p in lcc_index if pat.search(p.stem)]

        if cand:
            yield dem_path, cand[0]
            
# ---- Optional GeoTIFF IO backends ----
def _try_import_rasterio():
    try:
        import rasterio  # noqa
        return True
    except Exception:
        return False

def _read_raster(path: Path) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Returns (array, profile). array is float32 for DEM, uint8/float32 for mask.
    profile is a rasterio profile dict if readable via rasterio, else None.
    """
    suffix = path.suffix.lower()
    if suffix in [".tif", ".tiff"] and _try_import_rasterio():
        import rasterio
        with rasterio.open(path) as ds:
            arr = ds.read(1)
            prof = ds.profile.copy()
        return arr, prof

    # fallback: numpy / image
    if suffix in [".npy"]:
        return np.load(path), None

    # image fallback (png/jpg)
    from PIL import Image
    im = Image.open(path)
    if im.mode not in ("L", "I;16", "I", "F"):
        im = im.convert("L")
    arr = np.array(im)
    return arr, None

def _write_raster(path: Path, arr: np.ndarray, ref_profile: Optional[dict], nodata: Optional[float] = None):
    """
    Write single-band GeoTIFF if rasterio profile is available; otherwise write a plain TIFF via tifffile.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)

    if ref_profile is not None and _try_import_rasterio():
        import rasterio
        prof = ref_profile.copy()
        prof.update(
            count=1,
            dtype=str(arr.dtype),
            nodata=nodata if nodata is not None else ref_profile.get("nodata", None),
            compress=ref_profile.get("compress", "lzw"),
        )
        with rasterio.open(path, "w", **prof) as ds:
            ds.write(arr, 1)
        return

    # fallback: write TIFF without georef
    try:
        import tifffile
        tifffile.imwrite(str(path), arr)
    except Exception as e:
        raise RuntimeError(
            f"Failed to write TIFF (no rasterio profile available, tifffile error: {e}). "
            f"Install rasterio or tifffile, or provide GeoTIFF inputs."
        )

# ---- Try to reuse project helpers (preferred) ----
# These should match your training pipeline.
try:
    from datasets_lcc import build_pretrain_transform, _dem_to_tensor, _mask_to_tensor  # type: ignore
except Exception:
    build_pretrain_transform = None
    _dem_to_tensor = None
    _mask_to_tensor = None

# ---- Model import (MAE) ----
import models_mae  # your modified models_mae.py must be on PYTHONPATH


def _find_normalize_params(tf) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Best-effort: extract Normalize(mean,std) used for DEM from your build_pretrain_transform().
    We look for torchvision.transforms.Normalize inside nested objects.
    """
    try:
        import torchvision.transforms as T
        Normalize = T.Normalize
    except Exception:
        Normalize = None

    mean = std = None

    def walk(obj):
        nonlocal mean, std
        if obj is None or (mean is not None and std is not None):
            return
        # torchvision Normalize
        if Normalize is not None and isinstance(obj, Normalize):
            m = torch.tensor(obj.mean, dtype=torch.float32)
            s = torch.tensor(obj.std, dtype=torch.float32)
            mean, std = m, s
            return
        # common containers
        for attr in ("transforms", "t", "dem_tf", "img_tf", "dem_transform", "image_transform"):
            if hasattr(obj, attr):
                walk(getattr(obj, attr))
        # list/tuple
        if isinstance(obj, (list, tuple)):
            for it in obj:
                walk(it)
        # dict
        if isinstance(obj, dict):
            for it in obj.values():
                walk(it)

    walk(tf)
    return mean, std


def denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Inverse of torchvision Normalize: x * std + mean, channel-wise.
    x: [B,C,H,W]
    mean/std: [C]
    """
    mean = mean.to(x.device).view(1, -1, 1, 1)
    std = std.to(x.device).view(1, -1, 1, 1)
    return x * std + mean


def make_lcc_then_random_mask(
    lcc_nchw: torch.Tensor,
    patch_size: int,
    mask_ratio: float,
    seed: int,
    lcc_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a MAE-style mask driven by LCC first, then random fill to reach mask_ratio.

    Inputs:
      lcc_nchw: [B,1,H,W] float 0/1
      patch_size: int
      mask_ratio: fraction of patches to mask (0..1)
      seed: RNG seed (deterministic)
      lcc_threshold: patch is considered "LCC" if patch-average >= threshold

    Returns:
      patch_mask: [B,L] with 1=masked, 0=visible
      ids_keep:   [B, L_keep] indices of visible patches
      ids_restore:[B, L] inverse permutation for decoder unshuffle
    """
    B, _, H, W = lcc_nchw.shape
    p = patch_size
    assert H % p == 0 and W % p == 0, f"H,W must be divisible by patch_size={p}. Got {(H,W)}"

    # patch-level LCC indicator (same idea as max_pool/avg_pool in many pipelines)
    # Here we use average to be robust; set threshold=0.5 for binary mask.
    lcc_patch = F.avg_pool2d(lcc_nchw.float(), kernel_size=p, stride=p)  # [B,1,h,w]
    lcc_patch = (lcc_patch.squeeze(1) >= lcc_threshold).flatten(1)       # [B,L] bool

    L = lcc_patch.shape[1]
    len_mask = int(round(L * mask_ratio))
    len_keep = L - len_mask

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    ids_keep_list = []
    ids_restore_list = []
    patch_mask_list = []

    for b in range(B):
        idx_lcc = torch.where(lcc_patch[b])[0]
        idx_non = torch.where(~lcc_patch[b])[0]

        if len_mask <= 0:
            # nothing masked
            ids_keep = torch.arange(L)
            ids_mask = torch.empty((0,), dtype=torch.long)
        elif idx_lcc.numel() >= len_mask:
            # LCC alone exceeds required masked patches -> randomly choose a subset of LCC patches to mask
            perm = idx_lcc[torch.randperm(idx_lcc.numel(), generator=g)]
            ids_mask = perm[:len_mask]
            # keep everything else (including remaining LCC patches)
            keep_mask = torch.ones(L, dtype=torch.bool)
            keep_mask[ids_mask] = False
            ids_keep = torch.where(keep_mask)[0]
        else:
            # mask all LCC patches, and randomly mask additional non-LCC to reach len_mask
            remaining = len_mask - idx_lcc.numel()
            if remaining > idx_non.numel():
                remaining = idx_non.numel()
            perm_non = idx_non[torch.randperm(idx_non.numel(), generator=g)]
            ids_mask = torch.cat([idx_lcc, perm_non[:remaining]], dim=0)
            keep_mask = torch.ones(L, dtype=torch.bool)
            keep_mask[ids_mask] = False
            ids_keep = torch.where(keep_mask)[0]

        # MAE expects ids_keep length == len_keep
        # If due to rounding we got slight mismatch, fix deterministically.
        if ids_keep.numel() > len_keep:
            ids_keep = ids_keep[:len_keep]
        elif ids_keep.numel() < len_keep:
            # add more keep from currently masked (rare; just in case)
            ids_extra = torch.tensor([i for i in range(L) if i not in set(ids_keep.tolist())], dtype=torch.long)
            need = len_keep - ids_keep.numel()
            ids_keep = torch.cat([ids_keep, ids_extra[:need]], dim=0)

        ids_keep = ids_keep.to(torch.long)
        ids_mask = torch.tensor([i for i in range(L) if i not in set(ids_keep.tolist())], dtype=torch.long)

        # shuffle order: keep first then mask, then compute restore
        ids_shuffle = torch.cat([ids_keep, ids_mask], dim=0)
        ids_restore = torch.argsort(ids_shuffle)

        # patch_mask in original order
        patch_mask = torch.ones(L, dtype=torch.float32)
        patch_mask[ids_keep] = 0.0

        ids_keep_list.append(ids_keep)
        ids_restore_list.append(ids_restore)
        patch_mask_list.append(patch_mask)

    return (
        torch.stack(patch_mask_list, dim=0),     # [B,L]
        torch.stack(ids_keep_list, dim=0),       # [B,L_keep]
        torch.stack(ids_restore_list, dim=0),    # [B,L]
    )


def forward_encoder_with_ids(model, imgs: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
    """
    MAE encoder forward with explicit visible patch indices (ids_keep).
    imgs: [B,3,H,W] normalized
    ids_keep: [B,L_keep]
    """
    x = model.patch_embed(imgs)                       # [B, L, D]
    x = x + model.pos_embed[:, 1:, :]                 # add pos (no cls)
    B, L, D = x.shape

    # gather visible tokens
    ids_keep_exp = ids_keep.unsqueeze(-1).repeat(1, 1, D)
    x = torch.gather(x, dim=1, index=ids_keep_exp)    # [B, L_keep, D]

    # append cls
    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # transformer
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return x


@torch.no_grad()
def run_one(
    model,
    dem_nchw: torch.Tensor,
    lcc_nchw: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    mask_ratio: float,
    seed: int,
    out_dir: Path,
    save_png: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    p = model.patch_embed.patch_size[0]
    # build mask (LCC-first + random fill)
    patch_mask, ids_keep, ids_restore = make_lcc_then_random_mask(
        lcc_nchw=lcc_nchw, patch_size=p, mask_ratio=mask_ratio, seed=seed, lcc_threshold=0.5
    )
    patch_mask = patch_mask.to(dem_nchw.device)
    ids_keep = ids_keep.to(dem_nchw.device)
    ids_restore = ids_restore.to(dem_nchw.device)

    # forward
    latent = forward_encoder_with_ids(model, dem_nchw, ids_keep)
    pred = model.forward_decoder(latent, ids_restore)     # [B,L,p*p*3]
    recon = model.unpatchify(pred)                        # [B,3,H,W]

    # pixel-level mask image (1=masked)
    mask_img = model.unpatchify(patch_mask.unsqueeze(-1).repeat(1, 1, p * p * 3))  # [B,3,H,W]
    mask_img = (mask_img > 0.5).float()

    # masked input (masked -> mean value after denorm; here we just zero in normalized space)
    masked_input = dem_nchw * (1.0 - mask_img)

    # paste = visible from input + masked from recon  (THIS is the final "model output" you want to compare)
    paste = dem_nchw * (1.0 - mask_img) + recon * mask_img

    # denorm to original scaling
    dem_dn = denorm(dem_nchw, mean, std)
    recon_dn = denorm(recon, mean, std)
    paste_dn = denorm(paste, mean, std)
    print("[units check] dem_dn min/max:", dem_dn[0,0].min().item(), dem_dn[0,0].max().item())

    errs = compute_lcc_errors(dem_dn, recon_dn, paste_dn, lcc_nchw, mask_img)
    print("[err]", errs)
    lcc_pixels = errs["lcc_pixels"]
    lcc_masked_pixels = errs["lcc_masked_pixels"]
    print(f"[LCC coverage] masked_in_LCC = {lcc_masked_pixels}/{lcc_pixels} = {lcc_masked_pixels/max(lcc_pixels,1):.3f}")

    import json
    metrics = {
        "mask_ratio_target": float(mask_ratio),
        "seed": int(seed),
        **errs
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # take channel 0 as DEM (your DEM is replicated to 3-ch)
    dem0 = dem_dn[0, 0].detach().cpu().numpy().astype(np.float32)
    recon0 = recon_dn[0, 0].detach().cpu().numpy().astype(np.float32)
    paste0 = paste_dn[0, 0].detach().cpu().numpy().astype(np.float32)
    mask0 = (mask_img[0, 0] > 0.5).detach().cpu().numpy().astype(np.uint8)

    # quick stats
    L = patch_mask.shape[1]
    masked_frac = float(patch_mask.mean().item())
    # how many masked patches are LCC patches
    lcc_patch = (F.avg_pool2d(lcc_nchw.float(), kernel_size=p, stride=p) >= 0.5).flatten(1)
    river_masked_frac = float(((patch_mask > 0.5) & lcc_patch).sum().item() / max(1, (patch_mask > 0.5).sum().item()))

    print(f"[mask] patches={L}  mask_ratio(target)={mask_ratio:.3f}  masked_frac(actual)={masked_frac:.3f}  "
          f"river_masked_frac(of masked)={river_masked_frac:.3f}")

    # Save TIFFs (georef handled by caller)
    return dem0, recon0, paste0, mask0, masked_input, recon_dn, paste_dn


def load_checkpoint(model, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # strip possible "module." prefix
    new_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_state[nk] = v

    msg = model.load_state_dict(new_state, strict=False)
    print(f"[ckpt] loaded: {ckpt_path.name}")
    print(f"[ckpt] missing_keys={len(msg.missing_keys)} unexpected_keys={len(msg.unexpected_keys)}")
    if len(msg.unexpected_keys) > 0:
        print("  unexpected (first 10):", msg.unexpected_keys[:10])
    if len(msg.missing_keys) > 0:
        print("  missing (first 10):", msg.missing_keys[:10])

# Error
def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> float:
    denom = float(m.sum().item())
    if denom < 1.0:
        return float("nan")
    return float((x * m).sum().item() / denom)

def compute_lcc_errors(dem_dn, recon_dn, paste_dn, lcc_nchw, mask_img):
    """
    dem_dn/recon_dn/paste_dn: [1,3,H,W] (denorm)
    lcc_nchw: [1,1,H,W] 0/1
    mask_img: [1,3,H,W] 0/1 (1=masked)
    Return dict of metrics (float) + counts.
    """
    dem = dem_dn[:, :1]          # [1,1,H,W]
    recon = recon_dn[:, :1]
    paste = paste_dn[:, :1]

    lcc = (lcc_nchw > 0.5).float()
    msk = (mask_img[:, :1] > 0.5).float()

    lcc_msk = lcc * msk

    abs_recon = (recon - dem).abs()
    sq_recon = (recon - dem) ** 2

    abs_paste = (paste - dem).abs()
    sq_paste = (paste - dem) ** 2

    out = {}
    out["lcc_pixels"] = float(lcc.sum().item())
    out["lcc_masked_pixels"] = float(lcc_msk.sum().item())

    # recon errors
    out["recon_mae_lcc"] = _masked_mean(abs_recon, lcc)
    out["recon_rmse_lcc"] = float(_masked_mean(sq_recon, lcc) ** 0.5)

    out["recon_mae_lcc_masked"] = _masked_mean(abs_recon, lcc_msk)
    out["recon_rmse_lcc_masked"] = float(_masked_mean(sq_recon, lcc_msk) ** 0.5)

    # optional: paste errors (often optimistic)
    out["paste_mae_lcc"] = _masked_mean(abs_paste, lcc)
    out["paste_rmse_lcc"] = float(_masked_mean(sq_paste, lcc) ** 0.5)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True, type=str, help="DEM tile path (.tif/.png/.npy)")
    ap.add_argument("--lcc", required=True, type=str, help="LCC mask tile path (.tif/.png/.npy), 0/1")
    ap.add_argument("--ckpt", required=True, type=str, help="MAE checkpoint (.pth)")
    ap.add_argument("--model", default="mae_vit_large_patch16", type=str, help="Model name in models_mae.py")
    ap.add_argument("--input_size", default=336, type=int, help="Expected input H=W")
    ap.add_argument("--mask_ratio", default=0.35, type=float, help="Fraction of patches to mask")
    ap.add_argument("--seed", default=None, type=int, help="Seed; if omitted, derived from DEM filename")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--out_dir", default="./viz_out", type=str)
    ap.add_argument("--save_png", action="store_true", help="Also save quicklook PNGs")
    ap.add_argument("--fallback_mean", default=None, type=float, help="If cannot extract Normalize(mean/std), use this mean")
    ap.add_argument("--fallback_std", default=None, type=float, help="If cannot extract Normalize(mean/std), use this std")
    ap.add_argument("--dem_dir", default=None, type=str, help="Batch: DEM directory")
    ap.add_argument("--lcc_dir", default=None, type=str, help="Batch: LCC directory")
    ap.add_argument("--recursive", action="store_true", help="Batch: recurse into subfolders")
    ap.add_argument("--out_csv", default=None, type=str, help="Batch: write summary CSV under out_dir if set")

    args = ap.parse_args()

    dem_path = Path(args.dem)
    lcc_path = Path(args.lcc)
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)

    # seed
    if args.seed is None:
        # deterministic per tile, but different across tiles
        args.seed = (abs(hash(dem_path.name)) % (2**31 - 1))

    # read rasters
    dem_arr, dem_prof = _read_raster(dem_path)
    lcc_arr, lcc_prof = _read_raster(lcc_path)

    # enforce sizes (simple center-crop or pad if needed)
    H, W = dem_arr.shape[:2]
    if (H, W) != (args.input_size, args.input_size):
        # center crop/pad to input_size
        tgt = args.input_size
        out = np.zeros((tgt, tgt), dtype=dem_arr.dtype)
        h0 = max(0, (H - tgt) // 2)
        w0 = max(0, (W - tgt) // 2)
        h1 = min(H, h0 + tgt)
        w1 = min(W, w0 + tgt)
        ch0 = max(0, (tgt - (h1 - h0)) // 2)
        cw0 = max(0, (tgt - (w1 - w0)) // 2)
        out[ch0:ch0+(h1-h0), cw0:cw0+(w1-w0)] = dem_arr[h0:h1, w0:w1]
        dem_arr = out
        # lcc match
        outm = np.zeros((tgt, tgt), dtype=lcc_arr.dtype)
        outm[ch0:ch0+(h1-h0), cw0:cw0+(w1-w0)] = lcc_arr[h0:h1, w0:w1]
        lcc_arr = outm
        print(f"[warn] input resized via center crop/pad to {tgt}x{tgt}. "
              f"For exact georef consistency, provide tiles already at input_size.")

    # convert to tensors using training helpers if available
    if _dem_to_tensor is None or _mask_to_tensor is None or build_pretrain_transform is None:
        raise RuntimeError(
            "Cannot import datasets_lcc helpers (build_pretrain_transform, _dem_to_tensor, _mask_to_tensor). "
            "Put datasets_lcc.py on PYTHONPATH (same as training)."
        )

    dem_t = _dem_to_tensor(dem_arr)   # expected [3,H,W] float
    lcc_t = _mask_to_tensor(lcc_arr)  # expected [1,H,W] float 0/1

    tf = build_pretrain_transform(args.input_size)
    # Extract normalization used by tf (so we can denorm outputs correctly)
    mean, std = _find_normalize_params(tf)
    if mean is None or std is None:
        if args.fallback_mean is None or args.fallback_std is None:
            raise RuntimeError(
                "Could not extract Normalize(mean/std) from build_pretrain_transform(). "
                "Provide --fallback_mean and --fallback_std, or modify datasets_lcc to expose DEM_MEAN/DEM_STD."
            )
        mean = torch.tensor([args.fallback_mean]*3, dtype=torch.float32)
        std = torch.tensor([args.fallback_std]*3, dtype=torch.float32)
        print(f"[warn] using fallback mean/std: mean={mean.tolist()} std={std.tolist()}")

    # make deterministic wrt transforms if any randomness exists
    torch.manual_seed(int(args.seed))
    dem_t, lcc_t = tf(dem_t, lcc_t)

    dem_nchw = dem_t.unsqueeze(0).to(args.device)  # [1,3,H,W]
    lcc_nchw = lcc_t.unsqueeze(0).to(args.device)  # [1,1,H,W]

    # build model
    if args.model not in models_mae.__dict__:
        raise ValueError(f"Unknown model {args.model}. Available keys: {sorted(models_mae.__dict__.keys())[:20]} ...")
    model = models_mae.__dict__[args.model](img_size=args.input_size)
    model.to(args.device)
    model.eval()
    load_checkpoint(model, ckpt_path)

    # run
    dem0, recon0, paste0, mask0, masked_input, recon_dn, paste_dn = run_one(
        model=model,
        dem_nchw=dem_nchw,
        lcc_nchw=lcc_nchw,
        mean=mean,
        std=std,
        mask_ratio=float(args.mask_ratio),
        seed=int(args.seed),
        out_dir=out_dir,
        save_png=args.save_png,
    )
    # sanity check on denorm ch0
    abs_diff = np.abs(paste0 - dem0)
    vis_max = float((abs_diff * (1 - mask0)).max())
    msk_sum = float(mask0.sum())
    msk_mae = float((abs_diff * mask0).sum() / (msk_sum + 1e-9))

    print(f"[check] visible max abs diff (should ~0): {vis_max:.6g}")
    print(f"[check] masked MAE (paste vs dem): {msk_mae:.6g}")

    print(f"[check] visible max abs diff (should ~0): {vis_max:.6g}")
    print(f"[check] masked MAE (paste vs dem): {msk_mae:.6g}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # choose reference profile (prefer DEM)
    ref_prof = dem_prof if dem_prof is not None else lcc_prof

    _write_raster(out_dir / "input_dem.tif", dem0, ref_prof)
    #_write_raster(out_dir / "lcc_mask.tif", (lcc_arr > 0).astype(np.uint8), ref_prof)
    lcc0 = (lcc_nchw[0, 0] > 0.5).detach().cpu().numpy().astype(np.uint8)
    _write_raster(out_dir / "lcc_mask.tif", lcc0, ref_prof)
    _write_raster(out_dir / "patch_mask.tif", mask0, ref_prof)      # 1=masked
    _write_raster(out_dir / "recon_dem.tif", recon0, ref_prof)
    _write_raster(out_dir / "paste_dem.tif", paste0, ref_prof)

    # optional quicklook PNGs
    if args.save_png:
        import matplotlib.pyplot as plt

        def save_png(arr2d, title, fn):
            plt.figure(figsize=(6, 6))
            plt.imshow(arr2d, cmap="gray")
            plt.title(title)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / fn, dpi=150)
            plt.close()

        save_png(dem0, "DEM (denorm, ch0)", "1_dem.png")
        save_png(lcc0, "LCC mask (0/1)", "2_lcc_mask.png")
        save_png(mask0, "Patch mask (1=masked)", "3_patch_mask.png")
        save_png(recon0, "Reconstruction DEM (denorm, ch0)", "4_recon.png")
        save_png(paste0, "Paste DEM (denorm, ch0)", "5_paste.png")

    print(f"[done] outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
