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


def _valid_mask_from_values(
    arr: np.ndarray,
    nodata: Optional[float],
    nodata_threshold: Optional[float] = -9999.0,
) -> np.ndarray:
    """Return a robust pixel-validity mask for DEM/bathymetry values.

    A pixel is invalid when any of the following is true:
      1) the value is NaN/Inf;
      2) it matches the declared NoData value;
      3) it is <= ``nodata_threshold`` (default -9999).

    The threshold rule catches mixed sentinels such as -9999, -99999 and
    -999999. Values this low are outside the physically meaningful range of
    the bathymetry/topography tiles used by this project.
    """
    a = arr.astype(np.float32, copy=False)
    valid = np.isfinite(a)

    if nodata is not None:
        nd = float(nodata)
        # Exact comparison handles integer-like GeoTIFF sentinels. isclose also
        # protects against float32 representation differences.
        atol = max(1.0e-6, abs(nd) * 1.0e-7)
        valid &= ~np.isclose(a, nd, rtol=0.0, atol=atol)

    if nodata_threshold is not None:
        valid &= a > float(nodata_threshold)

    return valid


def _apply_nodata(
    arr: np.ndarray,
    nodata: Optional[float],
    nodata_threshold: Optional[float] = -9999.0,
) -> np.ndarray:
    """Convert all detected NoData/invalid values to NaN."""
    a = arr.astype(np.float32, copy=False)
    valid = _valid_mask_from_values(a, nodata, nodata_threshold)
    return np.where(valid, a, np.nan).astype(np.float32, copy=False)


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
        nodata_threshold: Optional[float] = -9999.0,
        random_flip: bool = False,
        return_path: bool = False,
        tile_norm: bool = False,
        tile_norm_eps: float = 1e-3,
        return_meta: bool = False,
        tile_norm_std_scale: float = 1.0,
    ):
        if (not dir_path) and (not list_path):
            raise ValueError('DEMTileDataset: either dir_path or list_path must be provided')

        self.dir_path = str(dir_path) if dir_path else ''
        self.list_path = str(list_path) if list_path else ''
        self.input_size = int(input_size)
        self.nodata = nodata
        self.nodata_threshold = nodata_threshold
        self.random_flip = bool(random_flip)
        self.return_path = bool(return_path)
        self.tile_norm = bool(tile_norm)
        self.tile_norm_eps = float(tile_norm_eps)
        self.return_meta = bool(return_meta)
        self.tile_norm_std_scale = float(tile_norm_std_scale)
        if self.tile_norm_std_scale <= 0:
            raise ValueError(
                f"tile_norm_std_scale must be > 0, got {self.tile_norm_std_scale}"
            )

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

    def _normalize_tile_instance(self, arr: np.ndarray):
        """
        Tile-wise instance normalization in meter space:
            arr_tile = (arr - tile_mean_m) / tile_std_safe
        """
        tile_mean_m = float(np.mean(arr))
        tile_std_m = float(np.std(arr))
        tile_std_scaled = tile_std_m * self.tile_norm_std_scale
        tile_std_safe = max(tile_std_scaled, self.tile_norm_eps)

        arr_tile = (arr - tile_mean_m) / tile_std_safe

        return arr_tile.astype(np.float32, copy=False), tile_mean_m, tile_std_m, tile_std_safe

    def __getitem__(self, idx: int):
        f = self.files[idx]
        arr = _apply_nodata(_read_dem_tiff(f), self.nodata, self.nodata_threshold)

        if np.isnan(arr).any():
            if self.norm_method == 'meanstd':
                fill = float(self.norm_a)
            elif self.norm_method == 'minmax':
                fill = float(self.norm_a)
            else:
                fill = float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else 0.0
            arr = np.where(np.isfinite(arr), arr, fill).astype(np.float32, copy=False)

        h, w = arr.shape
        s = self.input_size
        if h != s or w != s:
            if h >= s and w >= s:
                if self.random_flip:
                    top = np.random.randint(0, h - s + 1)
                    left = np.random.randint(0, w - s + 1)
                else:
                    top = (h - s) // 2
                    left = (w - s) // 2
                arr = arr[top:top + s, left:left + s]
            else:
                m = float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else 0.0
                out = np.full((s, s), m, dtype=np.float32)
                out[:h, :w] = arr
                arr = out

        if self.random_flip:
            if np.random.rand() < 0.5:
                arr = np.flip(arr, axis=1)
            if np.random.rand() < 0.5:
                arr = np.flip(arr, axis=0)

        arr = np.ascontiguousarray(arr).astype(np.float32, copy=False)

        # ---- keep original meter-space tile for meta / denorm ----
        arr_m = arr

        # ---- model input normalization ----
        if self.tile_norm:
            arr_model, tile_mean_m, tile_std_m, tile_std_safe = self._normalize_tile_instance(arr_m)
        else:
            arr_model = self._normalize(arr_m).astype(np.float32, copy=False)
            tile_mean_m = float(np.mean(arr_m))
            tile_std_m = float(np.std(arr_m))
            tile_std_scaled = tile_std_m * self.tile_norm_std_scale
            tile_std_safe = max(tile_std_scaled, self.tile_norm_eps)

        arr_model = np.ascontiguousarray(arr_model)
        x = torch.from_numpy(arr_model).unsqueeze(0)  # [1,H,W]

        meta = {
            "path": f,
            "tile_mean_m": tile_mean_m,
            "tile_std_m": tile_std_m,
            "tile_std_safe": tile_std_safe,
            "tile_norm": bool(self.tile_norm),
            "tile_norm_std_scale": float(self.tile_norm_std_scale),
            "global_norm_method": self.norm_method,
            "global_norm_a": float(self.norm_a),
            "global_norm_b": float(self.norm_b),
        }

        if self.return_meta and self.return_path:
            return x, meta, f
        elif self.return_meta:
            return x, meta
        elif self.return_path:
            return x, f
        else:
            return x

def compute_global_stats(
    files: Sequence[str],
    nodata: Optional[float] = None,
    nodata_threshold: Optional[float] = -9999.0,
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
            arr = _apply_nodata(_read_dem_tiff(fp), nodata, nodata_threshold)
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
    nodata_threshold: Optional[float] = -9999.0,
    max_files: int = 5000,
    method: str = 'meanstd',
    seed: int = 0,
) -> Dict[str, float]:
    """Backward-compatible wrapper expected by main_pretrain_dem.py."""
    stats = compute_global_stats(files, nodata=nodata, nodata_threshold=nodata_threshold, max_files=max_files, seed=seed)
    method = method.lower()
    if method not in ('meanstd', 'minmax'):
        raise ValueError(f'Unknown stats method: {method}')
    stats['method'] = method
    return stats

# --------------------------------------------------------
# Paired DEM + LCC-mask dataset for downstream bathymetry adaptation
# --------------------------------------------------------
# Added for Stage2 bathymetry/LCC downstream training.
# It keeps DEM and LCC mask spatial transforms synchronized and can
# normalize each DEM tile using only visible/known pixels outside LCC.

import re
from typing import Tuple

_LCC_KEY_RE = re.compile(r"(?P<res>\d+m)_(?P<body>.+?)_(?P<id>ID\d+)")


def _key_from_lcc_pair_name(path: str | Path) -> str:
    p = Path(path)
    m = _LCC_KEY_RE.search(p.stem)
    if not m:
        raise ValueError(f"Cannot parse DEM/LCC key from filename: {p.name}")
    return f"{m.group('res')}_{m.group('body')}_{m.group('id')}"


def _collect_tiff_files(root: str | Path) -> List[str]:
    root = Path(root)
    files: List[str] = []
    for pat in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        files.extend(str(p) for p in root.rglob(pat) if p.is_file())
    files.sort()
    return files


def _read_list_file(list_path: str | Path, base_dir: Optional[str] = None) -> List[str]:
    lp = Path(list_path)
    if not lp.is_file():
        raise FileNotFoundError(f"List file not found: {lp}")
    items = [ln.strip() for ln in lp.open() if ln.strip() and not ln.strip().startswith('#')]
    if len(items) == 0:
        raise ValueError(f"Empty list file: {lp}")
    if base_dir:
        items = [p if os.path.isabs(p) else os.path.join(base_dir, p) for p in items]
    return items


def _read_lcc_mask_tiff(path: str | Path) -> np.ndarray:
    """Read LCC mask as binary uint8 array [H,W], where 1 means masked/river."""
    arr = _read_dem_tiff(path)
    arr = np.where(np.isfinite(arr), arr, 0)
    return (arr > 0).astype(np.uint8, copy=False)


def _center_or_pad_triplet(
    arr: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
    input_size: int,
    random_crop: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same crop/pad to data, river mask, and validity mask.

    Padding is deliberately marked invalid. It must never become a visible
    encoder patch or a supervised prediction target.
    """
    h, w = arr.shape
    s = int(input_size)
    if h == s and w == s:
        return arr, mask, valid

    if h >= s and w >= s:
        if random_crop:
            top = np.random.randint(0, h - s + 1)
            left = np.random.randint(0, w - s + 1)
        else:
            top = (h - s) // 2
            left = (w - s) // 2
        sl = np.s_[top:top + s, left:left + s]
        return arr[sl], mask[sl], valid[sl]

    out_arr = np.full((s, s), np.nan, dtype=np.float32)
    out_mask = np.zeros((s, s), dtype=np.uint8)
    out_valid = np.zeros((s, s), dtype=np.uint8)
    hh = min(h, s)
    ww = min(w, s)
    out_arr[:hh, :ww] = arr[:hh, :ww]
    out_mask[:hh, :ww] = mask[:hh, :ww]
    out_valid[:hh, :ww] = valid[:hh, :ww]
    return out_arr, out_mask, out_valid


def _center_or_pad_pair(arr: np.ndarray, mask: np.ndarray, input_size: int, random_crop: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for older callers."""
    valid = np.isfinite(arr).astype(np.uint8)
    arr2, mask2, _ = _center_or_pad_triplet(arr, mask, valid, input_size, random_crop)
    return arr2, mask2


def _patch_status_from_masks(
    lcc_mask: np.ndarray,
    valid_mask: np.ndarray,
    patch_size: int = 16,
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Build mutually exclusive patch states.

    Rules used by the downstream exact-mask task:
      * candidate patch: any river/final-mask pixel occurs in the patch;
      * valid patch: every pixel in the patch is valid (no NoData at all);
      * prediction patch: candidate AND valid;
      * visible patch: non-candidate AND valid;
      * ignored patch: any NoData pixel occurs in the patch.
    """
    if lcc_mask.shape != valid_mask.shape:
        raise ValueError(f"Mask/valid shape mismatch: {lcc_mask.shape} vs {valid_mask.shape}")

    h, w = lcc_mask.shape
    p = int(patch_size)
    hh = (h // p) * p
    ww = (w // p) * p
    if hh <= 0 or ww <= 0:
        z = np.zeros((0, 0), dtype=bool)
        return {
            "candidate": z, "valid": z, "prediction": z,
            "visible": z, "ignored": z,
        }

    lcc_blocks = lcc_mask[:hh, :ww].reshape(hh // p, p, ww // p, p)
    valid_blocks = valid_mask[:hh, :ww].reshape(hh // p, p, ww // p, p)

    candidate = lcc_blocks.max(axis=(1, 3)) > float(threshold)
    valid_patch = valid_blocks.min(axis=(1, 3)) > 0
    prediction = candidate & valid_patch
    visible = (~candidate) & valid_patch
    ignored = ~valid_patch

    return {
        "candidate": candidate,
        "valid": valid_patch,
        "prediction": prediction,
        "visible": visible,
        "ignored": ignored,
    }


def _center_core_patch_mask(grid_shape, radius: int = 3) -> np.ndarray:
    """Return a centered square patch mask with half-width ``radius``.

    For the current 21x21 patch grid and radius=3, this is a 7x7 core.
    The full tile is still used by the encoder/decoder; this mask only defines
    the optional loss/evaluation region and tile-quality checks.
    """
    gh, gw = int(grid_shape[0]), int(grid_shape[1])
    if gh <= 0 or gw <= 0:
        return np.zeros((max(0, gh), max(0, gw)), dtype=bool)
    r = int(radius)
    if r < 0:
        raise ValueError(f"core_patch_radius must be >= 0, got {r}")
    cy, cx = gh // 2, gw // 2
    y0, y1 = max(0, cy - r), min(gh, cy + r + 1)
    x0, x1 = max(0, cx - r), min(gw, cx + r + 1)
    out = np.zeros((gh, gw), dtype=bool)
    out[y0:y1, x0:x1] = True
    return out


def _core_patch_metrics(status: Dict[str, np.ndarray], radius: int = 3) -> Dict[str, float]:
    """Summarize validity and prediction coverage inside the centered core."""
    core = _center_core_patch_mask(status["valid"].shape, radius=radius)
    n_core = int(core.sum())
    if n_core <= 0:
        return {
            "core_patch_count": 0,
            "core_valid_patch_count": 0,
            "core_prediction_patch_count": 0,
            "core_valid_patch_ratio": 0.0,
            "core_prediction_patch_ratio": 0.0,
        }
    n_valid = int((status["valid"] & core).sum())
    n_pred = int((status["prediction"] & core).sum())
    return {
        "core_patch_count": n_core,
        "core_valid_patch_count": n_valid,
        "core_prediction_patch_count": n_pred,
        "core_valid_patch_ratio": float(n_valid / n_core),
        # Unknown/prediction share is defined among usable core patches.
        "core_prediction_patch_ratio": float(n_pred / max(1, n_valid)),
    }


def _patch_ratio_from_mask(mask: np.ndarray, patch_size: int = 16, threshold: float = 0.5) -> float:
    """Approximate LCC patch ratio after max-pooling mask to patch grid."""
    valid = np.ones_like(mask, dtype=np.uint8)
    status = _patch_status_from_masks(mask, valid, patch_size=patch_size, threshold=threshold)
    return float(status["candidate"].mean()) if status["candidate"].size else 0.0


class DEMLCCPairDataset(Dataset):
    """Paired single-band DEM/bathymetry GeoTIFF + LCC mask dataset.

    Returns by default:
        x:        FloatTensor [1,H,W], normalized DEM/bathy tile
        meta:     dict with path, tile stats, and LCC ratios
        lcc_mask:   FloatTensor [1,H,W], 1=LCC/river/masked region
        valid_mask: FloatTensor [1,H,W], 1=valid data, 0=NoData/invalid

    If return_path=True, returns (x, meta, path, lcc_mask, valid_mask).

    Pairing modes:
      - If dem_list_path and lcc_list_path are both provided, pairs are matched line-by-line.
      - Otherwise masks are paired by filename key, e.g.
          Select_tile_Basin_1m_BadgerFinNull_ID10.tif
          Select_tile_1m_BadgerFinNull_ID10_LCC_Mask.tif
        -> key = 1m_BadgerFinNull_ID10
    """

    def __init__(
        self,
        dem_dir: Optional[str] = None,
        lcc_dir: Optional[str] = None,
        dem_list_path: Optional[str] = None,
        lcc_list_path: Optional[str] = None,
        input_size: int = 336,
        nodata: Optional[float] = None,
        nodata_threshold: Optional[float] = -9999.0,
        random_flip: bool = False,
        return_path: bool = False,
        tile_norm: bool = False,
        tile_norm_eps: float = 1e-3,
        tile_norm_std_scale: float = 1.0,
        return_meta: bool = True,
        tile_norm_visible_only: bool = False,
        min_lcc_patch_ratio: float = 0.0,
        max_lcc_patch_ratio: float = 1.0,
        min_valid_visible_patch_ratio: float = 0.0,
        loss_region_mode: str = "all",
        core_patch_radius: int = 3,
        min_core_valid_patch_ratio: float = 0.0,
        min_core_prediction_patch_ratio: float = 0.0,
        max_core_prediction_patch_ratio: float = 1.0,
        patch_size: int = 16,
        lcc_patch_threshold: float = 0.5,
    ):
        if (not dem_dir) and (not dem_list_path):
            raise ValueError('DEMLCCPairDataset: dem_dir or dem_list_path must be provided')
        if (not lcc_dir) and (not lcc_list_path):
            raise ValueError('DEMLCCPairDataset: lcc_dir or lcc_list_path must be provided')

        self.dem_dir = str(dem_dir) if dem_dir else ''
        self.lcc_dir = str(lcc_dir) if lcc_dir else ''
        self.input_size = int(input_size)
        self.nodata = nodata
        self.nodata_threshold = nodata_threshold
        self.random_flip = bool(random_flip)
        self.return_path = bool(return_path)
        self.tile_norm = bool(tile_norm)
        self.tile_norm_eps = float(tile_norm_eps)
        self.tile_norm_std_scale = float(tile_norm_std_scale)
        if self.tile_norm_std_scale <= 0:
            raise ValueError(
                f"tile_norm_std_scale must be > 0, got {self.tile_norm_std_scale}"
            )
        self.return_meta = bool(return_meta)
        self.tile_norm_visible_only = bool(tile_norm_visible_only)
        self.patch_size = int(patch_size)
        self.lcc_patch_threshold = float(lcc_patch_threshold)
        self.min_valid_visible_patch_ratio = float(min_valid_visible_patch_ratio)
        if not (0.0 <= self.min_valid_visible_patch_ratio <= 1.0):
            raise ValueError(
                "min_valid_visible_patch_ratio must be in [0,1], "
                f"got {self.min_valid_visible_patch_ratio}"
            )
        self.loss_region_mode = str(loss_region_mode).lower()
        if self.loss_region_mode not in {"all", "core"}:
            raise ValueError(
                f"loss_region_mode must be 'all' or 'core', got {loss_region_mode}"
            )
        self.core_patch_radius = int(core_patch_radius)
        if self.core_patch_radius < 0:
            raise ValueError("core_patch_radius must be >= 0")
        self.min_core_valid_patch_ratio = float(min_core_valid_patch_ratio)
        self.min_core_prediction_patch_ratio = float(min_core_prediction_patch_ratio)
        self.max_core_prediction_patch_ratio = float(max_core_prediction_patch_ratio)
        for name, value in (
            ("min_core_valid_patch_ratio", self.min_core_valid_patch_ratio),
            ("min_core_prediction_patch_ratio", self.min_core_prediction_patch_ratio),
            ("max_core_prediction_patch_ratio", self.max_core_prediction_patch_ratio),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {value}")
        if self.min_core_prediction_patch_ratio > self.max_core_prediction_patch_ratio:
            raise ValueError(
                "min_core_prediction_patch_ratio cannot exceed "
                "max_core_prediction_patch_ratio"
            )

        if dem_list_path:
            dem_files = _read_list_file(dem_list_path, base_dir=self.dem_dir if self.dem_dir else None)
        else:
            dem_files = _collect_tiff_files(self.dem_dir)
        if len(dem_files) == 0:
            raise ValueError('No DEM/bathy GeoTIFF files found.')

        if lcc_list_path:
            lcc_files = _read_list_file(lcc_list_path, base_dir=self.lcc_dir if self.lcc_dir else None)
        else:
            lcc_files = _collect_tiff_files(self.lcc_dir)
        if len(lcc_files) == 0:
            raise ValueError('No LCC mask GeoTIFF files found.')

        if dem_list_path and lcc_list_path:
            if len(dem_files) != len(lcc_files):
                raise ValueError(f'DEM list and LCC list length mismatch: {len(dem_files)} vs {len(lcc_files)}')
            pairs = list(zip(dem_files, lcc_files))
        else:
            mask_map: Dict[str, str] = {}
            for mp in lcc_files:
                key = _key_from_lcc_pair_name(mp)
                if key in mask_map:
                    raise ValueError(f'Duplicate LCC key={key}: {mask_map[key]} and {mp}')
                mask_map[key] = mp
            pairs = []
            missing = []
            for dp in dem_files:
                key = _key_from_lcc_pair_name(dp)
                mp = mask_map.get(key)
                if mp is None:
                    missing.append(Path(dp).name)
                else:
                    pairs.append((dp, mp))
            if len(pairs) == 0:
                raise RuntimeError(
                    f'No matched DEM/LCC pairs found. DEM root/list={dem_dir or dem_list_path}; LCC root/list={lcc_dir or lcc_list_path}'
                )
            if missing:
                print(f'[DEMLCCPairDataset] WARNING: {len(missing)} DEM files have no paired LCC mask. Example: {missing[:5]}')

        # Patch-aware quality filter. A patch containing ANY NoData pixel is
        # ignored completely: it is neither visible encoder input nor a
        # prediction/loss target. Ratios below are computed after that removal.
        min_r = float(min_lcc_patch_ratio)
        max_r = float(max_lcc_patch_ratio)
        min_visible_r = self.min_valid_visible_patch_ratio
        use_core_filter = self.loss_region_mode == "core"
        if (min_r > 0.0 or max_r < 1.0 or min_visible_r > 0.0
                or use_core_filter):
            kept = []
            drop_reason = {
                "read_error": 0,
                "prediction_ratio": 0,
                "visible_ratio": 0,
                "no_prediction_patch": 0,
                "core_valid_ratio": 0,
                "core_no_prediction_patch": 0,
                "core_prediction_ratio": 0,
            }
            examples = []

            for dp, mp in pairs:
                try:
                    raw = _read_dem_tiff(dp)
                    valid = _valid_mask_from_values(
                        raw, self.nodata, self.nodata_threshold
                    ).astype(np.uint8, copy=False)
                    arr = np.where(valid > 0, raw, np.nan).astype(np.float32, copy=False)
                    m = _read_lcc_mask_tiff(mp)
                    if arr.shape != m.shape:
                        raise ValueError(
                            f"Shape mismatch: DEM {arr.shape} vs LCC {m.shape}"
                        )

                    # Deterministic filtering. For the current 336x336 tiles this
                    # does not crop; it also marks any padding as invalid.
                    _, mm, vv = _center_or_pad_triplet(
                        arr, m, valid, self.input_size, random_crop=False
                    )
                    status = _patch_status_from_masks(
                        mm, vv, patch_size=self.patch_size,
                        threshold=self.lcc_patch_threshold,
                    )
                    n_total = int(status["valid"].size)
                    if n_total <= 0:
                        raise ValueError("No complete patches after crop/pad")

                    pred_r = float(status["prediction"].sum() / n_total)
                    visible_r = float(status["visible"].sum() / n_total)
                    n_pred = int(status["prediction"].sum())
                    core_metrics = _core_patch_metrics(
                        status, radius=self.core_patch_radius
                    )
                    core_valid_r = core_metrics["core_valid_patch_ratio"]
                    core_pred_r = core_metrics["core_prediction_patch_ratio"]
                    core_n_pred = core_metrics["core_prediction_patch_count"]

                    reason = None
                    if n_pred == 0:
                        reason = "no_prediction_patch"
                    elif not (min_r <= pred_r <= max_r):
                        reason = "prediction_ratio"
                    elif visible_r < min_visible_r:
                        reason = "visible_ratio"
                    elif use_core_filter and core_valid_r < self.min_core_valid_patch_ratio:
                        reason = "core_valid_ratio"
                    elif use_core_filter and core_n_pred == 0:
                        reason = "core_no_prediction_patch"
                    elif use_core_filter and not (
                        self.min_core_prediction_patch_ratio
                        <= core_pred_r
                        <= self.max_core_prediction_patch_ratio
                    ):
                        reason = "core_prediction_ratio"

                    if reason is None:
                        kept.append((dp, mp))
                    else:
                        drop_reason[reason] += 1
                        if len(examples) < 20:
                            examples.append(
                                f"{Path(dp).name}: reason={reason}, "
                                f"prediction_ratio={pred_r:.6f}, "
                                f"visible_valid_ratio={visible_r:.6f}, "
                                f"ignored_ratio={float(status['ignored'].mean()):.6f}, "
                                f"core_valid_ratio={core_valid_r:.6f}, "
                                f"core_prediction_ratio={core_pred_r:.6f}"
                            )
                except Exception as exc:
                    drop_reason["read_error"] += 1
                    if len(examples) < 20:
                        examples.append(
                            f"{Path(dp).name}: reason=read_error, error={exc!r}"
                        )

            dropped = sum(drop_reason.values())
            print(
                '[DEMLCCPairDataset] patch-quality filter: '
                f'kept={len(kept)} dropped={dropped} '
                f'prediction_ratio=[{min_r},{max_r}] '
                f'min_valid_visible_patch_ratio={min_visible_r} '
                f'loss_region_mode={self.loss_region_mode} '
                f'core_patch_radius={self.core_patch_radius} '
                f'core_valid_min={self.min_core_valid_patch_ratio} '
                f'core_prediction_range=['
                f'{self.min_core_prediction_patch_ratio},'
                f'{self.max_core_prediction_patch_ratio}] '
                f'drop_reasons={drop_reason}'
            )
            if examples:
                print('[DEMLCCPairDataset] dropped examples:')
                for msg in examples:
                    print('  ' + msg)

            pairs = kept
            if len(pairs) == 0:
                raise RuntimeError(
                    'All DEM/LCC pairs were removed by patch-quality filtering.'
                )

        self.pairs: List[Tuple[str, str]] = [(str(a), str(b)) for a, b in pairs]
        self.files: List[str] = [a for a, _ in self.pairs]
        self.mask_files: List[str] = [b for _, b in self.pairs]

        self.norm_method = 'none'
        self.norm_a = 0.0
        self.norm_b = 1.0

        print(f'[DEMLCCPairDataset] matched pairs: {len(self.pairs)}')

    def __len__(self) -> int:
        return len(self.pairs)

    def set_norm(self, a: float, b: float, method: str = 'meanstd') -> None:
        method = method.lower()
        if method not in ('meanstd', 'minmax'):
            raise ValueError(f'Unknown norm method: {method}')
        self.norm_method = method
        self.norm_a = float(a)
        self.norm_b = float(b)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if self.norm_method == 'none':
            return arr
        if self.norm_method == 'meanstd':
            std = self.norm_b if self.norm_b != 0 else 1.0
            return (arr - self.norm_a) / std
        denom = (self.norm_b - self.norm_a) if (self.norm_b - self.norm_a) != 0 else 1.0
        return (arr - self.norm_a) / denom

    def _normalize_tile_instance(
        self,
        arr: np.ndarray,
        lcc_mask: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
    ):
        valid = np.isfinite(arr)
        if valid_mask is not None:
            valid &= valid_mask.astype(bool)

        if self.tile_norm_visible_only and lcc_mask is not None:
            known = (lcc_mask == 0) & valid
            vals = arr[known]
            if vals.size < 2:
                vals = arr[valid]
        else:
            vals = arr[valid]

        if vals.size < 2:
            raise ValueError(
                'Tile has fewer than two valid values after NoData removal.'
            )

        tile_mean_m = float(np.mean(vals))
        tile_std_m = float(np.std(vals))
        tile_std_scaled = tile_std_m * self.tile_norm_std_scale
        tile_std_safe = max(tile_std_scaled, self.tile_norm_eps)

        arr_tile = (arr - tile_mean_m) / tile_std_safe
        # Invalid pixels are only tensor placeholders. Their patches are removed
        # from encoder input, prediction targets, loss, RMSE, and visualization.
        arr_tile = np.where(valid, arr_tile, 0.0)

        return arr_tile.astype(np.float32, copy=False), tile_mean_m, tile_std_m, tile_std_safe

    def __getitem__(self, idx: int):
        dem_path, mask_path = self.pairs[idx]
        raw = _read_dem_tiff(dem_path)
        valid = _valid_mask_from_values(
            raw, self.nodata, self.nodata_threshold
        ).astype(np.uint8, copy=False)
        arr = np.where(valid > 0, raw, np.nan).astype(np.float32, copy=False)
        lcc = _read_lcc_mask_tiff(mask_path)

        if arr.shape != lcc.shape:
            raise ValueError(
                f'Shape mismatch: DEM {Path(dem_path).name} {arr.shape} '
                f'vs LCC {Path(mask_path).name} {lcc.shape}'
            )

        arr, lcc, valid = _center_or_pad_triplet(
            arr, lcc, valid, self.input_size, random_crop=self.random_flip
        )

        if self.random_flip:
            flip_x = np.random.rand() < 0.5
            flip_y = np.random.rand() < 0.5
            if flip_x:
                arr = np.flip(arr, axis=1)
                lcc = np.flip(lcc, axis=1)
                valid = np.flip(valid, axis=1)
            if flip_y:
                arr = np.flip(arr, axis=0)
                lcc = np.flip(lcc, axis=0)
                valid = np.flip(valid, axis=0)

        arr_m = np.ascontiguousarray(arr).astype(np.float32, copy=False)
        lcc = np.ascontiguousarray(lcc).astype(np.uint8, copy=False)
        valid = np.ascontiguousarray(valid).astype(np.uint8, copy=False)

        if self.tile_norm:
            arr_model, tile_mean_m, tile_std_m, tile_std_safe = (
                self._normalize_tile_instance(
                    arr_m, lcc_mask=lcc, valid_mask=valid
                )
            )
        else:
            valid_bool = valid.astype(bool)
            arr_model = self._normalize(arr_m).astype(np.float32, copy=False)
            arr_model = np.where(valid_bool, arr_model, 0.0).astype(
                np.float32, copy=False
            )
            vals = arr_m[valid_bool]
            if vals.size < 2:
                raise ValueError(
                    f'{Path(dem_path).name} has fewer than two valid pixels.'
                )
            tile_mean_m = float(np.mean(vals))
            tile_std_m = float(np.std(vals))
            tile_std_scaled = tile_std_m * self.tile_norm_std_scale
            tile_std_safe = max(tile_std_scaled, self.tile_norm_eps)

        status = _patch_status_from_masks(
            lcc, valid, patch_size=self.patch_size,
            threshold=self.lcc_patch_threshold,
        )
        n_total = max(1, int(status["valid"].size))
        lcc_pixel_ratio = float(lcc.mean())
        candidate_patch_ratio = float(status["candidate"].sum() / n_total)
        prediction_patch_ratio = float(status["prediction"].sum() / n_total)
        visible_valid_patch_ratio = float(status["visible"].sum() / n_total)
        ignored_patch_ratio = float(status["ignored"].sum() / n_total)
        core_metrics = _core_patch_metrics(
            status, radius=self.core_patch_radius
        )

        x = torch.from_numpy(np.ascontiguousarray(arr_model)).unsqueeze(0)
        lcc_t = torch.from_numpy(lcc.astype(np.float32, copy=False)).unsqueeze(0)
        valid_t = torch.from_numpy(valid.astype(np.float32, copy=False)).unsqueeze(0)

        meta = {
            "path": dem_path,
            "lcc_path": mask_path,
            "tile_mean_m": tile_mean_m,
            "tile_std_m": tile_std_m,
            "tile_std_safe": tile_std_safe,
            "tile_norm_std_scale": float(self.tile_norm_std_scale),
            "tile_norm": bool(self.tile_norm),
            "tile_norm_visible_only": bool(self.tile_norm_visible_only),
            "global_norm_method": self.norm_method,
            "global_norm_a": float(self.norm_a),
            "global_norm_b": float(self.norm_b),
            "nodata_value": float(self.nodata) if self.nodata is not None else float('nan'),
            "nodata_threshold": (
                float(self.nodata_threshold)
                if self.nodata_threshold is not None else float('nan')
            ),
            "valid_pixel_ratio": float(valid.mean()),
            "lcc_pixel_ratio": lcc_pixel_ratio,
            # Keep old key for compatibility, but it now means usable prediction
            # patches after removing every patch containing any NoData pixel.
            "lcc_patch_ratio": prediction_patch_ratio,
            "candidate_lcc_patch_ratio": candidate_patch_ratio,
            "prediction_patch_ratio": prediction_patch_ratio,
            "visible_valid_patch_ratio": visible_valid_patch_ratio,
            "ignored_patch_ratio": ignored_patch_ratio,
            "loss_region_mode": self.loss_region_mode,
            "core_patch_radius": int(self.core_patch_radius),
            "core_valid_patch_ratio": float(
                core_metrics["core_valid_patch_ratio"]
            ),
            "core_prediction_patch_ratio": float(
                core_metrics["core_prediction_patch_ratio"]
            ),
            "core_valid_patch_count": int(
                core_metrics["core_valid_patch_count"]
            ),
            "core_prediction_patch_count": int(
                core_metrics["core_prediction_patch_count"]
            ),
        }

        if self.return_path:
            return x, meta, dem_path, lcc_t, valid_t
        if self.return_meta:
            return x, meta, lcc_t, valid_t
        return x, lcc_t, valid_t
