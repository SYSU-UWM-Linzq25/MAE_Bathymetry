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
    """
    Support:
      - x
      - (x, path)
      - (x, meta)
      - (x, meta, path)
    """
    samples = None
    meta = None
    path = None

    if isinstance(batch, dict):
        samples = batch["image"]
        meta = batch.get("meta", None)
        path = batch.get("path", None)
        return samples, meta, path

    if not isinstance(batch, (tuple, list)):
        return batch, None, None

    if len(batch) == 1:
        return batch[0], None, None
    if len(batch) == 2:
        # heuristic: second is meta dict or path list
        if isinstance(batch[1], dict):
            return batch[0], batch[1], None
        return batch[0], None, batch[1]
    if len(batch) >= 3:
        return batch[0], batch[1], batch[2]

    return batch[0], None, None

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
def _rmse_meters_from_pred(model, samples, pred, mask, meta=None, norm_scale_m: float = 1.0):
    target = model.patchify(samples)  # [N, L, P]

    pred_f = pred.float()
    target_f = target.float()

    keep = (mask == 0)
    pred_paste = pred_f.clone()
    pred_paste[keep] = target_f[keep]

    # error in normalized/model space
    err = pred_paste - target_f  # [N,L,P]

    if meta is None:
        # fallback to old global-scale logic
        mse = (err ** 2).mean(dim=-1)
        mask_f = mask.float()
        mask_sum = mask_f.sum().clamp(min=1.0)
        rmse_mask = torch.sqrt((mse * mask_f).sum() / mask_sum)
        rmse_all = torch.sqrt(mse.mean())
        scale = torch.as_tensor(float(norm_scale_m), device=samples.device, dtype=rmse_mask.dtype)
        return rmse_mask * scale, rmse_all * scale

    # tile-wise meter conversion
    tile_std = _meta_to_tile_std_tensor(meta, device=samples.device, dtype=err.dtype)  # [N]
    err_m = err * tile_std[:, None, None]  # [N,L,P]

    mse_m = (err_m ** 2).mean(dim=-1)  # [N,L]
    mask_f = mask.float()

    rmse_mask = torch.sqrt((mse_m * mask_f).sum() / mask_f.sum().clamp(min=1.0))
    rmse_all = torch.sqrt(mse_m.mean())

    return rmse_mask, rmse_all

@torch.no_grad()
def _rmse_meters_visible_median_bias_from_pred(model, samples, pred, mask, meta=None, norm_scale_m: float = 1.0):
    target = model.patchify(samples)
    pred_f = pred.float()
    target_f = target.float()

    keep_patch = (mask == 0)
    e = pred_f - target_f  # normalized/model space

    bias_list = []
    for i in range(e.shape[0]):
        ei = e[i]
        ki = keep_patch[i]
        if ki.sum() == 0:
            bias_list.append(torch.zeros((), device=e.device, dtype=e.dtype))
        else:
            vals = ei[ki].reshape(-1)
            bias_list.append(vals.median())
    bias = torch.stack(bias_list, dim=0)  # [N]

    pred_corr = pred_f - bias[:, None, None]
    pred_paste = pred_corr.clone()
    pred_paste[keep_patch] = target_f[keep_patch]

    err = pred_paste - target_f  # normalized/model space

    if meta is None:
        mse = (err ** 2).mean(dim=-1)
        mask_f = mask.float()
        rmse_mask = torch.sqrt((mse * mask_f).sum() / mask_f.sum().clamp(min=1.0))
        rmse_all = torch.sqrt(mse.mean())
        scale = torch.as_tensor(float(norm_scale_m), device=samples.device, dtype=rmse_mask.dtype)
        bias_m_vis_med = bias.mean() * scale
        return rmse_mask * scale, rmse_all * scale, bias_m_vis_med

    tile_std = _meta_to_tile_std_tensor(meta, device=samples.device, dtype=err.dtype)
    err_m = err * tile_std[:, None, None]
    bias_m = bias * tile_std  # [N]

    mse_m = (err_m ** 2).mean(dim=-1)
    mask_f = mask.float()

    rmse_mask = torch.sqrt((mse_m * mask_f).sum() / mask_f.sum().clamp(min=1.0))
    rmse_all = torch.sqrt(mse_m.mean())
    bias_m_vis_med = bias_m.mean()

    return rmse_mask, rmse_all, bias_m_vis_med
    
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
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    accum_iter = getattr(args, "accum_iter", 1)

    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print('log_dir:', log_writer.log_dir)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, meta, _ = _unwrap_batch(batch)
        samples = samples.to(device, non_blocking=True)

        # per-iteration lr schedule
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # NOTE: FutureWarning about autocast API is harmless; we keep compatibility.
        with torch.cuda.amp.autocast(enabled=getattr(args, "amp", True)):
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {} stopping training".format(loss_value))
            raise RuntimeError(f"Loss is not finite: {loss_value}")

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if args is not None and getattr(args, "log_rmse", False):
            rmse_mask_m, rmse_all_m = _rmse_meters_from_pred(
                model, samples, pred, mask,
                meta=meta,
                norm_scale_m=getattr(args, "norm_scale_m", 1.0)
            )
            
            metric_logger.update(rmse_m_mask=float(rmse_mask_m.item()))
            metric_logger.update(rmse_m_all=float(rmse_all_m.item()))

        # logging
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            if args is not None and getattr(args, "log_rmse", False):
                log_writer.add_scalar('train_rmse_m_mask', rmse_mask_m.item(), epoch_1000x)
                log_writer.add_scalar('train_rmse_m_all', rmse_all_m.item(), epoch_1000x)
                
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
    header = f'{prefix.capitalize()}: [{epoch}]'
    print_freq = 50

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, meta, _ = _unwrap_batch(batch)
        samples = samples.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=getattr(args, "amp", True)):
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        if args is not None and getattr(args, "log_rmse", False):
            rmse_mask_m, rmse_all_m = _rmse_meters_from_pred(
                model, samples, pred, mask,
                meta=meta,
                norm_scale_m=getattr(args, "norm_scale_m", 1.0)
            )
            
            metric_logger.update(rmse_m_mask=float(rmse_mask_m.item()))
            metric_logger.update(rmse_m_all=float(rmse_all_m.item()))
            
            # New: visible-median bias correction (deployment-style)
            rmse_mask_vis_m, rmse_all_vis_m, bias_vis_m = _rmse_meters_visible_median_bias_from_pred(
                model, samples, pred, mask,
                meta=meta,
                norm_scale_m=getattr(args, "norm_scale_m", 1.0)
            )
            
            metric_logger.update(rmse_m_mask_viscorr=float(rmse_mask_vis_m.item()))
            metric_logger.update(rmse_m_all_viscorr=float(rmse_all_vis_m.item()))
            metric_logger.update(bias_m_vis_med=float(bias_vis_m.item()))
            
    metric_logger.synchronize_between_processes()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if log_writer is not None:
        log_writer.add_scalar(f'{prefix}_loss', stats['loss'], epoch)
        if args is not None and getattr(args, "log_rmse", False):
            log_writer.add_scalar(f'{prefix}_rmse_m_mask', stats.get('rmse_m_mask', float('nan')), epoch)
            log_writer.add_scalar(f'{prefix}_rmse_m_all', stats.get('rmse_m_all', float('nan')), epoch)
            log_writer.add_scalar(f'{prefix}_rmse_m_mask_viscorr', stats.get('rmse_m_mask_viscorr', float('nan')), epoch)
            log_writer.add_scalar(f'{prefix}_rmse_m_all_viscorr',  stats.get('rmse_m_all_viscorr',  float('nan')), epoch)
            log_writer.add_scalar(f'{prefix}_bias_m_vis_med',      stats.get('bias_m_vis_med',      float('nan')), epoch)
            
    print(f"{prefix} stats:", stats)
    return stats
