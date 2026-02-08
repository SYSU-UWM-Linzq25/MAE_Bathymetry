# mae_visualize_lcc.py
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import models_mae
from datasets_lcc import _read_array_any, _dem_to_tensor, _mask_to_tensor, build_pretrain_transform


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denorm(x: torch.Tensor) -> torch.Tensor:
    # x: [N,3,H,W] normalized
    return (x * IMAGENET_STD.to(x.device) + IMAGENET_MEAN.to(x.device)).clamp(0, 1)


def save_gray(img_hw: np.ndarray, out_png: Path, title: str = ""):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(img_hw, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_rgb(img_hwc: np.ndarray, out_png: Path, title: str = ""):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(img_hwc)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def prepare_model(checkpoint_path: str, arch: str, img_size: int, device: str):
    model = models_mae.__dict__[arch](img_size=img_size, save_path="")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    model.to(device)
    model.eval()
    return model


def load_one_pair(dem_path: str, mask_path: str, input_size: int, seed: int):
    dem = _dem_to_tensor(_read_array_any(Path(dem_path)))   # [3,H,W], ~[0,1]
    msk = _mask_to_tensor(_read_array_any(Path(mask_path))) # [1,H,W], {0,1}

    tf = build_pretrain_transform(input_size)
    torch.manual_seed(seed)
    dem, msk = tf(dem, msk)  # dem normalized, msk still 0/1 float

    dem = dem.unsqueeze(0)  # [1,3,H,W]
    msk = msk.unsqueeze(0)  # [1,1,H,W]
    return dem, msk


@torch.no_grad()
def run_and_save(
    model,
    dem_nchw: torch.Tensor,
    lcc_nchw: torch.Tensor,
    out_dir: Path,
    tag: str,
    mask_ratio: float,
    lcc_priority: float,
    loss_on_lcc_only: bool,
):
    device = next(model.parameters()).device
    dem_nchw = dem_nchw.to(device, non_blocking=True)
    lcc_nchw = lcc_nchw.to(device, non_blocking=True)

    # run MAE (new forward signature)
    loss, pred, patch_mask = model(
        dem_nchw.float(),
        mask_ratio=mask_ratio,
        lcc_mask=lcc_nchw.float(),
        loss_on_lcc_only=loss_on_lcc_only,
        lcc_priority=lcc_priority,
    )

    # reconstruct image
    recon = model.unpatchify(pred)  # [1,3,H,W] in normalized space
    p = model.patch_embed.patch_size[0]

    # patch_mask: [1,L] -> image mask [1,3,H,W]
    pm = patch_mask.unsqueeze(-1).repeat(1, 1, p * p * 3)
    pm_img = model.unpatchify(pm)  # [1,3,H,W], 1=masked,0=visible

    # de-normalize for display
    dem_dn = denorm(dem_nchw)
    recon_dn = denorm(recon)

    masked_dn = dem_dn * (1 - pm_img)
    paste_dn  = dem_dn * (1 - pm_img) + recon_dn * pm_img

    # also compute lcc at patch level for stats (same as model)
    lcc_patch = (F.max_pool2d(lcc_nchw.float(), kernel_size=p, stride=p) > 0.5).flatten(1)  # [1,L]
    masked = patch_mask > 0.5
    river_masked = (masked & (lcc_patch > 0.5))
    nonriver_masked = (masked & (lcc_patch <= 0.5))

    # compute per-patch mse like training
    target = model.patchify(dem_nchw)   # NOTE: dem_nchw is normalized already
    mse_patch = ((pred - target) ** 2).mean(dim=-1)  # [1,L]

    def safe_mean(x, sel):
        denom = sel.sum().item()
        if denom == 0:
            return float("nan")
        return (x[sel].mean().item())

    mse_all = safe_mean(mse_patch[0], masked[0])
    mse_riv = safe_mean(mse_patch[0], river_masked[0])
    mse_non = safe_mean(mse_patch[0], nonriver_masked[0])

    print(f"[{tag}] loss(tensor)={loss.item():.6g}  mse_masked={mse_all:.6g}  mse_river={mse_riv:.6g}  mse_nonriver={mse_non:.6g}")
    if masked[0].sum().item() > 0:
        frac_river = river_masked[0].sum().item() / masked[0].sum().item()
        print(f"[{tag}] masked_patches={int(masked[0].sum())}, river_masked_frac={frac_river:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # save images (use channel-0 as gray DEM view)
    save_gray(dem_dn[0, 0].detach().cpu().numpy(), out_dir / f"{tag}_1_dem.png", "DEM (denorm, ch0)")
    save_gray(lcc_nchw[0, 0].detach().cpu().numpy(), out_dir / f"{tag}_2_lcc_mask.png", "LCC mask")
    save_gray(pm_img[0, 0].detach().cpu().numpy(), out_dir / f"{tag}_3_patch_mask.png", "Patch mask (1=masked)")
    save_gray(masked_dn[0, 0].detach().cpu().numpy(), out_dir / f"{tag}_4_masked_input.png", "Masked input")
    save_gray(recon_dn[0, 0].detach().cpu().numpy(), out_dir / f"{tag}_5_recon.png", "Reconstruction")
    save_gray(paste_dn[0, 0].detach().cpu().numpy(), out_dir / f"{tag}_6_paste.png", "Recon + visible (paste)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch", default="mae_vit_large_patch16_dec512d8b")
    ap.add_argument("--img_size", type=int, default=336)
    ap.add_argument("--dem", required=True, help="Path to one DEM tile (tif/png/jpg)")
    ap.add_argument("--mask", required=True, help="Path to matching LCC mask tile")
    ap.add_argument("--out_dir", default="visualize_lcc")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--mask_ratio", type=float, default=0.35)
    ap.add_argument("--lcc_priority", type=float, default=10.0)
    ap.add_argument("--loss_on_lcc_only", action="store_true")
    args = ap.parse_args()

    model = prepare_model(args.checkpoint, args.arch, args.img_size, args.device)
    dem, msk = load_one_pair(args.dem, args.mask, args.img_size, args.seed)

    tag = Path(args.dem).stem
    run_and_save(
        model,
        dem,
        msk,
        Path(args.out_dir),
        tag=tag,
        mask_ratio=args.mask_ratio,
        lcc_priority=args.lcc_priority,
        loss_on_lcc_only=args.loss_on_lcc_only,
    )


if __name__ == "__main__":
    main()
