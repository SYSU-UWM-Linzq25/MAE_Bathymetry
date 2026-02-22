# DEM Tile Dataset + Global Stats Utilities
# --------------------------------------------------------
# Tailored for single-band DEM GeoTIFF tiles (e.g., 336x336).
#
# Key features for your retraining workflow:
#   - Read single-band GeoTIFF tiles (Pillow-first; tifffile fallback)
#   - Global normalization computed from TRAIN set (mean/std or min/max)
#   - Dataset returns normalized tensor [1, H, W]
#   - JSON helpers (so main_pretrain_dem.py can record/restore train stats)
#
# NOTE on LZW-compressed TIFF:
#   `tifffile` needs `imagecodecs` to decode LZW. To avoid adding that
#   dependency, we read with Pillow first (libtiff-based, usually supports LZW).
# --------------------------------------------------------

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, List

import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

try:
    import tifffile  # optional fallback
except Exception:
    tifffile = None


def load_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open('r') as f:
        return json.load(f)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _read_dem_tiff(path: str | Path) -> np.ndarray:
    """Read a single-band GeoTIFF into numpy array (H, W) float32."""
    path = str(path)

    # 1) Pillow first
    arr = None
    try:
        with Image.open(path) as im:
            arr = np.array(im)
    except Exception:
        arr = None

    # 2) tifffile fallback
    if arr is None:
        if tifffile is None:
            raise RuntimeError('Failed to read GeoTIFF with Pillow and tifffile is unavailable.')
        arr = tifffile.imread(path)

    # squeeze (H, W, 1) -> (H, W)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D DEM tile, got shape={arr.shape} for {path}')

    return arr.astype(np.float32, copy=False)


def _apply_nodata(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    if nodata is not None:
        a = np.where(a == nodata, np.nan, a)
    a = np.where(np.isfinite(a), a, np.nan)
    return a


class DEMTileDataset(Dataset):
    """Dataset for single-band DEM GeoTIFF tiles.

    Supports two modes:
      1) `list_path`: a txt file listing (absolute or relative) tile paths
      2) `dir_path` : a directory containing tiles (recursively searched)

    The signature is kept compatible with `main_pretrain_dem.py`.
    """

    def __init__(
        self,
        dir_path: Optional[str] = None,
        list_path: Optional[str] = None,
        input_size: int = 336,
        nodata: Optional[float] = None,
        random_flip: bool = False,
        return_path: bool = False,
    ):
        if (not dir_path) and (not list_path):
            raise ValueError('DEMTileDataset: either dir_path or list_path must be provided')

        self.dir_path = str(dir_path) if dir_path else ''
        self.list_path = str(list_path) if list_path else ''
        self.input_size = int(input_size)
        self.nodata = nodata
        self.random_flip = bool(random_flip)
        self.return_path = bool(return_path)

        self.files: List[str] = []
        if self.list_path:
            lp = Path(self.list_path)
            if not lp.is_file():
                raise FileNotFoundError(f'List file not found: {lp}')
            items = [ln.strip() for ln in lp.open() if ln.strip()]
            if len(items) == 0:
                raise ValueError(f'Empty list file: {lp}')
            # if list contains relative paths and dir_path is provided, join them
            if self.dir_path:
                self.files = [p if os.path.isabs(p) else os.path.join(self.dir_path, p) for p in items]
            else:
                self.files = items
        else:
            dp = Path(self.dir_path)
            if not dp.is_dir():
                raise NotADirectoryError(f'dir_path is not a directory: {dp}')
            # recurse for tif/tiff
            pats = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
            for pat in pats:
                self.files.extend([str(p) for p in dp.rglob(pat)])
            self.files.sort()
            if len(self.files) == 0:
                raise ValueError(f'No GeoTIFF tiles found under: {dp}')

        # normalization settings
        self.norm_method = 'none'  # 'meanstd' or 'minmax' or 'none'
        self.norm_a = 0.0          # mean or min
        self.norm_b = 1.0          # std  or max

    def __len__(self) -> int:
        return len(self.files)

    def set_norm(self, a: float, b: float, method: str = 'meanstd') -> None:
        """Set normalization parameters.

        - method='meanstd': a=mean, b=std
        - method='minmax' : a=vmin, b=vmax
        """
        method = method.lower()
        if method not in ('meanstd', 'minmax'):
            raise ValueError(f'Unknown norm method: {method}')
        self.norm_method = method
        self.norm_a = float(a)
        self.norm_b = float(b)

    def get_norm(self) -> dict:
        return {'method': self.norm_method, 'a': self.norm_a, 'b': self.norm_b}

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if self.norm_method == 'none':
            return arr
        if self.norm_method == 'meanstd':
            mean = self.norm_a
            std = self.norm_b if self.norm_b != 0 else 1.0
            return (arr - mean) / std
        # minmax
        vmin = self.norm_a
        vmax = self.norm_b
        denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        return (arr - vmin) / denom

    def __getitem__(self, idx: int) -> torch.Tensor:
        f = self.files[idx]
        arr = _apply_nodata(_read_dem_tiff(f), self.nodata)

        if np.isnan(arr).any():
            if self.norm_method == 'meanstd':
                fill = float(self.norm_a)  # global mean
            elif self.norm_method == 'minmax':
                fill = float(self.norm_a)  # vmin
            else:
                fill = float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else 0.0
            arr = np.where(np.isfinite(arr), arr, fill).astype(np.float32, copy=False)

        # Ensure fixed input_size (crop/pad). Most of your tiles are already input_size.
        h, w = arr.shape
        s = self.input_size
        if h != s or w != s:
            # crop if larger
            if h >= s and w >= s:
                if self.random_flip:
                    # random crop for training
                    top = np.random.randint(0, h - s + 1)
                    left = np.random.randint(0, w - s + 1)
                else:
                    # center crop for eval
                    top = (h - s) // 2
                    left = (w - s) // 2
                arr = arr[top:top + s, left:left + s]
            else:
                # pad if smaller
                m = float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else 0.0
                out = np.full((s, s), m, dtype=np.float32)
                out[:h, :w] = arr
                arr = out

        # augmentation: random flip
        if self.random_flip:
            if np.random.rand() < 0.5:
                arr = np.flip(arr, axis=1)
            if np.random.rand() < 0.5:
                arr = np.flip(arr, axis=0)

        # np.flip creates negative strides; make contiguous for torch
        arr = np.ascontiguousarray(arr)

        arrn = self._normalize(arr)
        arrn = arrn.astype(np.float32, copy=False)

        arrn = np.ascontiguousarray(arrn)

        # 1 channel
        x = torch.from_numpy(arrn).unsqueeze(0)
        if self.return_path:
            return x, f
        return x


def compute_global_stats(
    files: Sequence[str],
    nodata: Optional[float] = None,
    max_files: Optional[int] = 5000,
    max_pixels_per_file: int = 5000,
    seed: int = 0,
) -> Dict[str, float]:
    """Compute global mean/std and min/max on a sampled subset of files/pixels."""
    rng = np.random.default_rng(seed)

    files = list(files)
    if len(files) == 0:
        raise ValueError('compute_global_stats: empty file list')

    if (max_files is None) or (max_files <= 0) or (len(files) <= max_files):
        sample_files = files
    else:
        idx = rng.choice(len(files), size=int(max_files), replace=False)
        sample_files = [files[i] for i in idx]

    # Welford running mean/variance on sampled pixels
    n = 0
    mean = 0.0
    M2 = 0.0
    gmin = math.inf
    gmax = -math.inf

    for fp in sample_files:
        try:
            arr = _apply_nodata(_read_dem_tiff(fp), nodata)
        except Exception:
            continue

        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            continue

        vmin = float(np.nanmin(valid))
        vmax = float(np.nanmax(valid))
        gmin = min(gmin, vmin)
        gmax = max(gmax, vmax)

        # subsample pixels
        if valid.size > max_pixels_per_file:
            pix = rng.choice(valid, size=max_pixels_per_file, replace=False)
        else:
            pix = valid

        pix = pix.astype(np.float64, copy=False)
        for x in pix:
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2

    if n < 2:
        raise RuntimeError(f'compute_global_stats failed (n_pixels={n}).')

    var = M2 / (n - 1)
    std = float(math.sqrt(max(var, 1e-12)))

    return {
        'mean': float(mean),
        'std': float(std),
        'min': float(gmin),
        'max': float(gmax),
        'n_files_used': float(len(sample_files)),
        'n_pixels_used': float(n),
    }


def compute_dem_stats(
    files: Sequence[str],
    nodata: Optional[float] = None,
    max_files: int = 5000,
    method: str = 'meanstd',
    seed: int = 0,
) -> Dict[str, float]:
    """Backward-compatible wrapper expected by main_pretrain_dem.py."""
    stats = compute_global_stats(files, nodata=nodata, max_files=max_files, seed=seed)
    method = method.lower()
    if method not in ('meanstd', 'minmax'):
        raise ValueError(f'Unknown stats method: {method}')
    stats['method'] = method
    return stats
