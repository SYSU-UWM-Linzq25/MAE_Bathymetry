# datasets_lcc.py
"""
Paired DEM + LCC-mask dataset for MAE.

Returns:
  img:      torch.FloatTensor [3, H, W]
  lcc_mask: torch.FloatTensor [1, H, W]  (1 = river)

- DEM can be 1-band or 3-band. 1-band will be replicated to 3.
- Mask is binarized: any value > 0 becomes 1.
- IMPORTANT: random crop/flip are applied identically to image and mask.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio  # preferred for GeoTIFF
except Exception:
    rasterio = None

from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF


# ---- filename pairing ----
_KEY_RE = re.compile(r"(?P<res>\d+m)_(?P<river>.+?)_(?P<id>ID\d+)")

def _key_from_path(p: Path) -> str:
    """
    Extract '1m_BadgerFinNull_ID10' from filenames like:
      - Select_tile_Basin_1m_BadgerFinNull_ID10.tif
      - Select_tile_1m_BadgerFinNull_ID10_LCC_Mask.tif
    """
    m = _KEY_RE.search(p.stem)
    if not m:
        raise ValueError(f"Cannot parse key from filename: {p.name}")
    return f"{m.group('res')}_{m.group('river')}_{m.group('id')}"


def _collect_files(root: str | Path, exts: Sequence[str]) -> List[Path]:
    root = Path(root)
    out: List[Path] = []
    for ext in exts:
        out += list(root.rglob(f"*{ext}"))
    return sorted([p for p in out if p.is_file()])


# ---- raster reading ----
def _read_array_any(path: Path) -> np.ndarray:
    """Read tif/png/jpg -> ndarray (H,W) or (H,W,C)."""
    if rasterio is not None:
        with rasterio.open(path) as src:
            arr = src.read()  # (C,H,W)
        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))  # (H,W,C)
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
        return arr

    # fallback PIL (may fail on some float GeoTIFF)
    from PIL import Image
    with Image.open(path) as im:
        arr = np.array(im)
    return arr


def _dem_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """DEM -> FloatTensor [3,H,W] in roughly [0,1]."""
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = arr.astype(np.float32)

    vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.0
    if vmax > 1.5:
        if vmax <= 255.0:
            arr = arr / 255.0
        elif vmax <= 65535.0:
            arr = arr / 65535.0
        else:
            # robust scaling for float DEM
            flat = arr.reshape(-1)
            flat = flat[np.isfinite(flat)]
            if flat.size > 0:
                lo, hi = np.percentile(flat, [1, 99])
                if hi > lo:
                    arr = (arr - lo) / (hi - lo)
                arr = np.clip(arr, 0.0, 1.0)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)

    # ensure 3 channels
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"Unexpected DEM shape: {arr.shape}")

    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # 3,H,W


def _mask_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Mask -> FloatTensor [1,H,W] in {0,1}."""
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = arr.astype(np.float32)
    ten = torch.from_numpy(arr)[None, ...]
    ten = (ten > 0).to(torch.float32)
    return ten.contiguous()


# ---- joint transform ----
class JointMAETransform:
    """Same RandomResizedCrop + HFlip for (img, mask)."""

    def __init__(
        self,
        input_size: int,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        scale: Tuple[float, float] = (0.2, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        hflip_p: float = 0.5,
    ):
        self.size = (input_size, input_size)
        self.mean = mean
        self.std = std
        self.scale = scale
        self.ratio = ratio
        self.hflip_p = hflip_p

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        i, j, h, w = RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)
        img = TF.resized_crop(
            img, i, j, h, w, self.size,
            interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        mask = TF.resized_crop(
            mask, i, j, h, w, self.size,
            interpolation=InterpolationMode.NEAREST
        )
        if torch.rand(1).item() < self.hflip_p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        img = TF.normalize(img, mean=self.mean, std=self.std)
        mask = (mask > 0.5).to(dtype=img.dtype)
        return img, mask


def build_pretrain_transform(input_size: int) -> JointMAETransform:
    return JointMAETransform(input_size=input_size)


# ---- dataset ----
class DEMLCCPairDataset(Dataset):
    def __init__(
        self,
        dem_root: str,
        mask_root: str,
        transform: Optional[JointMAETransform] = None,
        exts: Sequence[str] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    ):
        self.transform = transform

        dem_files = _collect_files(dem_root, exts)
        mask_files = _collect_files(mask_root, exts)

        dem_map: Dict[str, Path] = {_key_from_path(p): p for p in dem_files}
        mask_map: Dict[str, Path] = {_key_from_path(p): p for p in mask_files}

        keys = sorted(set(dem_map.keys()) & set(mask_map.keys()))
        if len(keys) == 0:
            raise RuntimeError(
                f"No matched DEM/mask pairs found!\nDEM={dem_root}\nMASK={mask_root}\n"
                f"Example DEM: {dem_files[0].name if dem_files else 'None'}\n"
                f"Example MASK: {mask_files[0].name if mask_files else 'None'}"
            )

        self.pairs = [(dem_map[k], mask_map[k]) for k in keys]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        dem_path, mask_path = self.pairs[idx]

        dem = _dem_to_tensor(_read_array_any(dem_path))
        mask = _mask_to_tensor(_read_array_any(mask_path))

        # sanity: spatial alignment
        if dem.shape[1:] != mask.shape[1:]:
            raise ValueError(
                f"Shape mismatch: DEM {dem_path.name} {dem.shape} vs MASK {mask_path.name} {mask.shape}"
            )

        if self.transform is not None:
            dem, mask = self.transform(dem, mask)

        return dem, mask
