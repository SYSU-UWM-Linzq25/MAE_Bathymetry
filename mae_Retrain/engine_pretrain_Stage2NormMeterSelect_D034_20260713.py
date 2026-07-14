"""Training and evaluation loops for MAE pre-training.

Original MAE code only provided `train_one_epoch`.
For DEM retraining we also add `evaluate_one_epoch` so we can monitor
validation loss (and optionally RMSE in meters) each epoch and save the best checkpoint.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def _unwrap_samples(batch):
    """Support multiple dataset return formats.

    - ImageFolder returns (samples, label)
    - Our DEM dataset returns (samples, path)
    - Some loaders may return samples directly
    """
    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        return batch[0]
    return batch

def _unwrap_batch(batch):
    """Unpack datasets with optional hidden/prediction mask and validity masks.

    Returns:
        samples, meta, path, hidden_or_lcc_mask, valid_mask, loss_pixel_mask

    Backward compatibility:
      * old DEMLCCPairDataset returns no loss_pixel_mask.
      * new DEMDualMaskDataset returns hidden mask + pixel loss mask.
    """
    if isinstance(batch, dict):
        samples = batch.get("image", batch.get("x"))
        meta = batch.get("meta", None)
        path = batch.get("path", None)
        hidden_mask = batch.get("hidden_mask", batch.get("lcc_mask", None))
        valid_mask = batch.get("valid_mask", None)
        loss_pixel_mask = batch.get("loss_pixel_mask", None)
        return samples, meta, path, hidden_mask, valid_mask, loss_pixel_mask

    if not isinstance(batch, (tuple, list)):
        return batch, None, None, None, None, None

    if len(batch) == 1:
        return batch[0], None, None, None, None, None

    if len(batch) == 2:
        if torch.is_tensor(batch[1]):
            return batch[0], None, None, batch[1], None, None
        if isinstance(batch[1], dict):
            return batch[0], batch[1], None, None, None, None
        return batch[0], None, batch[1], None, None, None

    if len(batch) == 3:
        # no-meta form: (x, hidden/lcc_mask, valid_mask)
        if torch.is_tensor(batch[1]) and torch.is_tensor(batch[2]):
            return batch[0], None, None, batch[1], batch[2], None
        if isinstance(batch[1], dict) and torch.is_tensor(batch[2]):
            return batch[0], batch[1], None, batch[2], None, None
        if torch.is_tensor(batch[1]) and isinstance(batch[2], dict):
            return batch[0], batch[2], None, batch[1], None, None
        return batch[0], batch[1], batch[2], None, None, None

    if len(batch) == 4:
        # (x, hidden, valid, loss_pixel)
        if all(torch.is_tensor(batch[i]) for i in (1, 2, 3)):
            return batch[0], None, None, batch[1], batch[2], batch[3]
        # old/new paired no-path format: (x, meta, lcc/hidden, valid)
        if (isinstance(batch[1], dict) and torch.is_tensor(batch[2])
                and torch.is_tensor(batch[3])):
            return batch[0], batch[1], None, batch[2], batch[3], None
        # old paired path format: (x, meta, path, lcc)
        if isinstance(batch[1], dict) and torch.is_tensor(batch[3]):
            return batch[0], batch[1], batch[2], batch[3], None, None
        if torch.is_tensor(batch[1]):
            return batch[0], batch[2], batch[3], batch[1], None, None
        return batch[0], batch[1], batch[2], None, None, None

    if len(batch) == 5:
        # New DEMDualMaskDataset no-path format: (x, meta, hidden, valid, loss_pixel)
        if (isinstance(batch[1], dict) and torch.is_tensor(batch[2])
                and torch.is_tensor(batch[3]) and torch.is_tensor(batch[4])):
            return batch[0], batch[1], None, batch[2], batch[3], batch[4]
        # Old paired path format: (x, meta, path, lcc, valid)
        if torch.is_tensor(batch[3]) and torch.is_tensor(batch[4]):
            return batch[0], batch[1], batch[2], batch[3], batch[4], None

    if len(batch) >= 6:
        # New DEMDualMaskDataset path format: (x, meta, path, hidden, valid, loss_pixel)
        if (torch.is_tensor(batch[3]) and torch.is_tensor(batch[4])
                and torch.is_tensor(batch[5])):
            return batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

    return batch[0], None, None, None, None, None


def _model_forward(model, samples, args, lcc_mask=None, valid_mask=None, loss_pixel_mask=None):
    """Centralized model call so mask rules match train and evaluation."""
    if lcc_mask is not None:
        return model(
            samples,
            mask_ratio=getattr(args, "mask_ratio", 0.75),
            lcc_mask=lcc_mask,
            valid_mask=valid_mask,
            loss_pixel_mask=loss_pixel_mask,
            loss_on_lcc_only=getattr(args, "loss_on_lcc_only", False),
            lcc_priority=getattr(args, "lcc_priority", 10.0),
            lcc_mask_mode=getattr(args, "lcc_mask_mode", "exact"),
            lcc_patch_threshold=getattr(args, "lcc_patch_threshold", 0.5),
            loss_region_mode=getattr(args, "loss_region_mode", "all"),
            core_patch_radius=getattr(args, "core_patch_radius", 3),
            return_aux_masks=True,
        )
    return model(
        samples, mask_ratio=getattr(args, "mask_ratio", 0.75),
        return_aux_masks=True,
    )


def _meta_to_tile_std_tensor(meta, device, dtype=torch.float32):
    """
    meta can be:
      - dict of batched tensors/lists from default collate
    """
    if meta is None:
        return None

    if isinstance(meta, dict):
        vals = meta["tile_std_safe"]
        if torch.is_tensor(vals):
            return vals.to(device=device, dtype=dtype)
        return torch.as_tensor(vals, device=device, dtype=dtype)

    raise TypeError(f"Unsupported meta type: {type(meta)}")

def _meta_to_tile_mean_tensor(meta, device, dtype=torch.float32):
    if meta is None:
        return None
    if isinstance(meta, dict):
        vals = meta["tile_mean_m"]
        if torch.is_tensor(vals):
            return vals.to(device=device, dtype=dtype)
        return torch.as_tensor(vals, device=device, dtype=dtype)
    raise TypeError(f"Unsupported meta type: {type(meta)}")

@torch.no_grad()
def _rmse_meters_from_pred(
    model, samples, pred, mask, valid_mask=None, meta=None,
    norm_scale_m: float = 1.0, prediction_mask=None,
):
    target = model.patchify(samples)
    pred_f = pred.float()
    target_f = target.float()

    keep = (mask == 0)
    pred_paste = pred_f.clone()
    pred_paste[keep] = target_f[keep]
    err = pred_paste - target_f

    if valid_mask is None:
        valid_patch = torch.ones_like(mask, dtype=torch.float32)
    else:
        valid_patch = model._valid_patch_from_mask(valid_mask).float()

    if meta is None:
        mse = (err ** 2).mean(dim=-1)
        mask_f = mask.float() * valid_patch
        rmse_mask = torch.sqrt(
            (mse * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        )
        rmse_all = torch.sqrt(
            (mse * valid_patch).sum() / valid_patch.sum().clamp(min=1.0)
        )
        scale = torch.as_tensor(
            float(norm_scale_m), device=samples.device, dtype=rmse_mask.dtype
        )
        return rmse_mask * scale, rmse_all * scale

    tile_std = _meta_to_tile_std_tensor(
        meta, device=samples.device, dtype=err.dtype
    )
    err_m = err * tile_std[:, None, None]
    mse_m = (err_m ** 2).mean(dim=-1)
    mask_f = mask.float() * valid_patch

    rmse_mask = torch.sqrt(
        (mse_m * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    )
    rmse_all = torch.sqrt(
        (mse_m * valid_patch).sum() / valid_patch.sum().clamp(min=1.0)
    )
    return rmse_mask, rmse_all

@torch.no_grad()

@torch.no_grad()
def _rmse_meters_from_pred_pixel_mask(
    model, samples, pred, loss_pixel_mask, valid_mask=None, meta=None,
    norm_scale_m: float = 1.0, patch_loss_mask=None,
):
    """RMSE in meters on the same pixel-level region used by v2 loss.

    Important for MAE v2:
      * loss_pixel_mask is the exact pixel-level supervision mask.
      * patch_loss_mask is the patch-level loss region returned by model.forward.
        In core-loss mode it is prediction_patch AND core_patch.
      * Therefore rmse_m_mask must use:
            loss_pixel_mask AND patch_loss_mask AND valid_patch
        so early stopping and logs match the actual optimized loss region.
    """
    target = model.patchify(samples)
    err = pred.float() - target.float()
    pix_w = model.patchify(loss_pixel_mask.float())
    pix_w = (pix_w > 0.5).float()

    if patch_loss_mask is not None:
        pix_w = pix_w * patch_loss_mask.float().unsqueeze(-1)

    if valid_mask is not None:
        valid_patch = model._valid_patch_from_mask(valid_mask).float()
        pix_w = pix_w * valid_patch.unsqueeze(-1)

    if meta is None:
        err_m = err * float(norm_scale_m)
    else:
        tile_std = _meta_to_tile_std_tensor(
            meta, device=samples.device, dtype=err.dtype
        )
        err_m = err * tile_std[:, None, None]

    denom = pix_w.sum().clamp(min=1.0)
    rmse_mask = torch.sqrt(((err_m ** 2) * pix_w).sum() / denom)

    # Also report all-valid-patch RMSE for context.
    if valid_mask is None:
        valid_patch = torch.ones(err.shape[:2], device=err.device, dtype=err.dtype)
    else:
        valid_patch = model._valid_patch_from_mask(valid_mask).float()
    mse_patch = (err_m ** 2).mean(dim=-1)
    rmse_all = torch.sqrt(
        (mse_patch * valid_patch).sum() / valid_patch.sum().clamp(min=1.0)
    )
    return rmse_mask, rmse_all

def _rmse_meters_visible_median_bias_from_pred(
    model, samples, pred, mask, valid_mask=None, meta=None,
    norm_scale_m: float = 1.0, prediction_mask=None,
):
    target = model.patchify(samples)
    pred_f = pred.float()
    target_f = target.float()

    if valid_mask is None:
        valid_patch = torch.ones_like(mask, dtype=torch.bool)
    else:
        valid_patch = model._valid_patch_from_mask(valid_mask).bool()

    # Core-loss mode still masks river patches across the full tile. Only
    # genuinely visible patches may estimate bias.
    model_mask = mask if prediction_mask is None else prediction_mask
    keep_patch = (model_mask == 0) & valid_patch
    e = pred_f - target_f

    bias_list = []
    for i in range(e.shape[0]):
        vals = e[i][keep_patch[i]].reshape(-1)
        if vals.numel() == 0:
            bias_list.append(torch.zeros((), device=e.device, dtype=e.dtype))
        else:
            bias_list.append(vals.median())
    bias = torch.stack(bias_list, dim=0)

    pred_corr = pred_f - bias[:, None, None]
    pred_paste = pred_corr.clone()
    pred_paste[model_mask == 0] = target_f[model_mask == 0]
    err = pred_paste - target_f

    valid_patch_f = valid_patch.float()
    mask_f = mask.float() * valid_patch_f

    if meta is None:
        mse = (err ** 2).mean(dim=-1)
        rmse_mask = torch.sqrt(
            (mse * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        )
        rmse_all = torch.sqrt(
            (mse * valid_patch_f).sum()
            / valid_patch_f.sum().clamp(min=1.0)
        )
        scale = torch.as_tensor(
            float(norm_scale_m), device=samples.device, dtype=rmse_mask.dtype
        )
        return rmse_mask * scale, rmse_all * scale, bias.mean() * scale

    tile_std = _meta_to_tile_std_tensor(
        meta, device=samples.device, dtype=err.dtype
    )
    err_m = err * tile_std[:, None, None]
    bias_m = bias * tile_std
    mse_m = (err_m ** 2).mean(dim=-1)

    rmse_mask = torch.sqrt(
        (mse_m * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    )
    rmse_all = torch.sqrt(
        (mse_m * valid_patch_f).sum()
        / valid_patch_f.sum().clamp(min=1.0)
    )
    return rmse_mask, rmse_all, bias_m.mean()


def _meter_objective_components(
    model,
    samples,
    pred,
    patch_loss_mask,
    loss_pixel_mask,
    valid_mask,
    meta,
):
    """Return differentiable exact-mask objectives and meter-space diagnostics.

    The supervision region exactly matches the MAE-v2 dual-mask objective:
        Loss_Mask_Pixel
        AND decoder/prediction patch mask
        AND centered core patch mask (already folded into patch_loss_mask)
        AND fully valid patch mask.

    With tile-wise normalization:
        x_norm = (x_m - tile_mean_m) / tile_std_safe

    therefore:
        |pred_m - target_m|
        = |pred_norm - target_norm| * tile_std_safe

    The tile mean cancels in the error. Large-STD tiles consequently receive
    proportionally larger gradients when optimization_loss=meter_mae.
    """
    target = model.patchify(samples).float()
    pred_f = pred.float()
    err_norm = pred_f - target

    if meta is None:
        raise RuntimeError(
            "meter_mae requires dataset metadata containing tile_std_safe. "
            "Use DEMDualMaskDataset with --tile_norm."
        )
    tile_std = _meta_to_tile_std_tensor(
        meta, device=samples.device, dtype=err_norm.dtype
    )
    if tile_std is None:
        raise RuntimeError("Missing tile_std_safe for meter-space objective.")
    if torch.any(~torch.isfinite(tile_std)) or torch.any(tile_std <= 0):
        raise RuntimeError("Invalid tile_std_safe encountered in meter-space objective.")

    err_m = err_norm * tile_std[:, None, None]

    if loss_pixel_mask is not None:
        pixel_w = model.patchify(loss_pixel_mask.float())
        pixel_w = (pixel_w > 0.5).to(dtype=err_norm.dtype)
    else:
        pixel_w = torch.ones_like(err_norm)

    if patch_loss_mask is not None:
        pixel_w = pixel_w * patch_loss_mask.float().unsqueeze(-1)

    if valid_mask is not None:
        valid_patch = model._valid_patch_from_mask(valid_mask).float()
        pixel_w = pixel_w * valid_patch.unsqueeze(-1)

    count = pixel_w.sum()
    if count <= 0:
        # Defensive differentiable zero. Dataset filtering should prevent this.
        zero = err_m.mean() * 0.0
        return {
            "meter_mae_loss": zero,
            "normalized_mse_loss": zero,
            "abs_m_sum": zero.detach(),
            "sq_m_sum": zero.detach(),
            "sq_norm_sum": zero.detach(),
            "count": count.detach(),
        }

    abs_m_sum = (err_m.abs() * pixel_w).sum()
    sq_m_sum = ((err_m ** 2) * pixel_w).sum()
    sq_norm_sum = ((err_norm ** 2) * pixel_w).sum()

    return {
        "meter_mae_loss": abs_m_sum / count,
        "normalized_mse_loss": sq_norm_sum / count,
        "abs_m_sum": abs_m_sum.detach(),
        "sq_m_sum": sq_m_sum.detach(),
        "sq_norm_sum": sq_norm_sum.detach(),
        "count": count.detach(),
    }


def _sync_exact_pixel_sums(
    abs_m_sum: float,
    sq_m_sum: float,
    sq_norm_sum: float,
    count: float,
    device: torch.device,
):
    """Synchronize exact pixel sums across distributed workers."""
    t = torch.tensor(
        [abs_m_sum, sq_m_sum, sq_norm_sum, count],
        dtype=torch.float64,
        device=device,
    )
    if misc.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(t)
    return [float(x) for x in t.cpu().tolist()]


def _finalize_exact_meter_stats(stats, sums, optimization_loss: str):
    abs_m_sum, sq_m_sum, sq_norm_sum, count = sums
    if count > 0:
        mae_m = abs_m_sum / count
        rmse_m = math.sqrt(max(sq_m_sum / count, 0.0))
        normalized_mse = sq_norm_sum / count
    else:
        mae_m = float("nan")
        rmse_m = float("nan")
        normalized_mse = float("nan")

    stats["mae_m_mask"] = mae_m
    stats["rmse_m_mask"] = rmse_m
    stats["normalized_mse_mask"] = normalized_mse
    stats["supervised_pixel_count"] = count

    # Keep `loss` as the actual optimization objective so old logging/checkpoint
    # code remains meaningful. For either objective it is the exact global pixel-weighted objective.
    if optimization_loss == "meter_mae":
        stats["loss"] = mae_m
    elif optimization_loss == "normalized_mse":
        stats["loss"] = normalized_mse
    return stats


def _update_mask_ratio_metrics(metric_logger, model, mask, prediction_mask, valid_mask):
    if valid_mask is not None:
        valid_patch = model._valid_patch_from_mask(valid_mask).float()
        actual_ratio = mask.float().sum() / valid_patch.sum().clamp(min=1.0)
        prediction_ratio = (
            prediction_mask.float().sum() / valid_patch.sum().clamp(min=1.0)
        )
        ignored_ratio = 1.0 - valid_patch.mean()
        metric_logger.update(actual_mask_ratio=float(actual_ratio.item()))
        metric_logger.update(
            actual_prediction_mask_ratio=float(prediction_ratio.item())
        )
        metric_logger.update(ignored_patch_ratio=float(ignored_ratio.item()))
    else:
        metric_logger.update(actual_mask_ratio=float(mask.float().mean().item()))


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = f"Epoch: [{epoch}]"
    print_freq = 20
    accum_iter = getattr(args, "accum_iter", 1)
    optimization_loss = getattr(args, "optimization_loss", "normalized_mse")

    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print("log_dir:", log_writer.log_dir)

    exact_abs_m = 0.0
    exact_sq_m = 0.0
    exact_sq_norm = 0.0
    exact_count = 0.0

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples, meta, _, lcc_mask, valid_mask, loss_pixel_mask = _unwrap_batch(batch)
        samples = samples.to(device, non_blocking=True)
        if lcc_mask is not None:
            lcc_mask = lcc_mask.to(device, non_blocking=True)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device, non_blocking=True)
        if loss_pixel_mask is not None:
            loss_pixel_mask = loss_pixel_mask.to(device, non_blocking=True)

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        with torch.cuda.amp.autocast(enabled=getattr(args, "amp", True)):
            normalized_model_loss, pred, mask, prediction_mask = _model_forward(
                model,
                samples,
                args,
                lcc_mask=lcc_mask,
                valid_mask=valid_mask,
                loss_pixel_mask=loss_pixel_mask,
            )

        components = _meter_objective_components(
            model,
            samples,
            pred,
            patch_loss_mask=mask,
            loss_pixel_mask=loss_pixel_mask,
            valid_mask=valid_mask,
            meta=meta,
        )

        if optimization_loss == "meter_mae":
            loss = components["meter_mae_loss"]
        elif optimization_loss == "normalized_mse":
            loss = components["normalized_mse_loss"]
        else:
            raise ValueError(f"Unsupported optimization_loss={optimization_loss}")

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is not finite: {loss_value}")

        count_value = float(components["count"].item())
        exact_abs_m += float(components["abs_m_sum"].item())
        exact_sq_m += float(components["sq_m_sum"].item())
        exact_sq_norm += float(components["sq_norm_sum"].item())
        exact_count += count_value

        batch_mae_m = (
            float(components["abs_m_sum"].item()) / count_value
            if count_value > 0 else 0.0
        )
        batch_rmse_m = (
            math.sqrt(float(components["sq_m_sum"].item()) / count_value)
            if count_value > 0 else 0.0
        )
        batch_norm_mse = (
            float(components["sq_norm_sum"].item()) / count_value
            if count_value > 0 else 0.0
        )

        scaled_loss = loss / accum_iter
        loss_scaler(
            scaled_loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        metric_logger.update(
            loss=loss_value,
            mae_m_mask=batch_mae_m,
            rmse_m_mask=batch_rmse_m,
            normalized_mse_mask=batch_norm_mse,
        )
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if args is not None and getattr(args, "log_rmse", False):
            # Context metric over all valid patches. Primary masked RMSE is
            # finalized exactly from pixel sums at epoch end.
            if loss_pixel_mask is not None:
                _, rmse_all_m = _rmse_meters_from_pred_pixel_mask(
                    model,
                    samples,
                    pred,
                    loss_pixel_mask,
                    valid_mask=valid_mask,
                    meta=meta,
                    norm_scale_m=getattr(args, "norm_scale_m", 1.0),
                    patch_loss_mask=mask,
                )
            else:
                _, rmse_all_m = _rmse_meters_from_pred(
                    model,
                    samples,
                    pred,
                    mask,
                    valid_mask=valid_mask,
                    meta=meta,
                    norm_scale_m=getattr(args, "norm_scale_m", 1.0),
                )
            metric_logger.update(rmse_m_all=float(rmse_all_m.item()))

        if lcc_mask is not None:
            _update_mask_ratio_metrics(
                metric_logger, model, mask, prediction_mask, valid_mask
            )

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000
            )
            log_writer.add_scalar("train_loss", loss_value, epoch_1000x)
            log_writer.add_scalar(
                "train_mae_m_mask", batch_mae_m, epoch_1000x
            )
            log_writer.add_scalar(
                "train_rmse_m_mask", batch_rmse_m, epoch_1000x
            )
            log_writer.add_scalar(
                "train_normalized_mse_mask", batch_norm_mse, epoch_1000x
            )
            log_writer.add_scalar("lr", lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    sums = _sync_exact_pixel_sums(
        exact_abs_m, exact_sq_m, exact_sq_norm, exact_count, device
    )
    stats = _finalize_exact_meter_stats(stats, sums, optimization_loss)

    print("Averaged stats:", stats)
    return stats


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    log_writer=None,
    args=None,
    prefix: str = "val",
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"{prefix.capitalize()}: [{epoch}]"
    print_freq = 50
    optimization_loss = getattr(args, "optimization_loss", "normalized_mse")

    exact_abs_m = 0.0
    exact_sq_m = 0.0
    exact_sq_norm = 0.0
    exact_count = 0.0

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples, meta, _, lcc_mask, valid_mask, loss_pixel_mask = _unwrap_batch(batch)
        samples = samples.to(device, non_blocking=True)
        if lcc_mask is not None:
            lcc_mask = lcc_mask.to(device, non_blocking=True)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device, non_blocking=True)
        if loss_pixel_mask is not None:
            loss_pixel_mask = loss_pixel_mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=getattr(args, "amp", True)):
            normalized_model_loss, pred, mask, prediction_mask = _model_forward(
                model,
                samples,
                args,
                lcc_mask=lcc_mask,
                valid_mask=valid_mask,
                loss_pixel_mask=loss_pixel_mask,
            )

        components = _meter_objective_components(
            model,
            samples,
            pred,
            patch_loss_mask=mask,
            loss_pixel_mask=loss_pixel_mask,
            valid_mask=valid_mask,
            meta=meta,
        )

        if optimization_loss == "meter_mae":
            loss = components["meter_mae_loss"]
        elif optimization_loss == "normalized_mse":
            loss = components["normalized_mse_loss"]
        else:
            raise ValueError(f"Unsupported optimization_loss={optimization_loss}")

        count_value = float(components["count"].item())
        exact_abs_m += float(components["abs_m_sum"].item())
        exact_sq_m += float(components["sq_m_sum"].item())
        exact_sq_norm += float(components["sq_norm_sum"].item())
        exact_count += count_value

        batch_mae_m = (
            float(components["abs_m_sum"].item()) / count_value
            if count_value > 0 else 0.0
        )
        batch_rmse_m = (
            math.sqrt(float(components["sq_m_sum"].item()) / count_value)
            if count_value > 0 else 0.0
        )
        batch_norm_mse = (
            float(components["sq_norm_sum"].item()) / count_value
            if count_value > 0 else 0.0
        )

        metric_logger.update(
            loss=float(loss.item()),
            mae_m_mask=batch_mae_m,
            rmse_m_mask=batch_rmse_m,
            normalized_mse_mask=batch_norm_mse,
        )

        if args is not None and getattr(args, "log_rmse", False):
            if loss_pixel_mask is not None:
                _, rmse_all_m = _rmse_meters_from_pred_pixel_mask(
                    model,
                    samples,
                    pred,
                    loss_pixel_mask,
                    valid_mask=valid_mask,
                    meta=meta,
                    norm_scale_m=getattr(args, "norm_scale_m", 1.0),
                    patch_loss_mask=mask,
                )
            else:
                _, rmse_all_m = _rmse_meters_from_pred(
                    model,
                    samples,
                    pred,
                    mask,
                    valid_mask=valid_mask,
                    meta=meta,
                    norm_scale_m=getattr(args, "norm_scale_m", 1.0),
                )
            metric_logger.update(rmse_m_all=float(rmse_all_m.item()))

            if loss_pixel_mask is None:
                rmse_mask_vis_m, rmse_all_vis_m, bias_vis_m = (
                    _rmse_meters_visible_median_bias_from_pred(
                        model,
                        samples,
                        pred,
                        mask,
                        valid_mask=valid_mask,
                        meta=meta,
                        norm_scale_m=getattr(args, "norm_scale_m", 1.0),
                        prediction_mask=prediction_mask,
                    )
                )
                metric_logger.update(
                    rmse_m_mask_viscorr=float(rmse_mask_vis_m.item())
                )
                metric_logger.update(
                    rmse_m_all_viscorr=float(rmse_all_vis_m.item())
                )
                metric_logger.update(bias_m_vis_med=float(bias_vis_m.item()))

        if lcc_mask is not None:
            _update_mask_ratio_metrics(
                metric_logger, model, mask, prediction_mask, valid_mask
            )

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    sums = _sync_exact_pixel_sums(
        exact_abs_m, exact_sq_m, exact_sq_norm, exact_count, device
    )
    stats = _finalize_exact_meter_stats(stats, sums, optimization_loss)

    if log_writer is not None:
        log_writer.add_scalar(f"{prefix}_loss", stats["loss"], epoch)
        log_writer.add_scalar(
            f"{prefix}_mae_m_mask", stats["mae_m_mask"], epoch
        )
        log_writer.add_scalar(
            f"{prefix}_rmse_m_mask", stats["rmse_m_mask"], epoch
        )
        log_writer.add_scalar(
            f"{prefix}_normalized_mse_mask",
            stats["normalized_mse_mask"],
            epoch,
        )
        if args is not None and getattr(args, "log_rmse", False):
            log_writer.add_scalar(
                f"{prefix}_rmse_m_all",
                stats.get("rmse_m_all", float("nan")),
                epoch,
            )
            log_writer.add_scalar(
                f"{prefix}_rmse_m_mask_viscorr",
                stats.get("rmse_m_mask_viscorr", float("nan")),
                epoch,
            )
            log_writer.add_scalar(
                f"{prefix}_rmse_m_all_viscorr",
                stats.get("rmse_m_all_viscorr", float("nan")),
                epoch,
            )
            log_writer.add_scalar(
                f"{prefix}_bias_m_vis_med",
                stats.get("bias_m_vis_med", float("nan")),
                epoch,
            )

    print(f"{prefix} stats:", stats)
    return stats
