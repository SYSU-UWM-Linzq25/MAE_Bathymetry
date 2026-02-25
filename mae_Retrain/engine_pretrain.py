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


@torch.no_grad()
def _rmse_meters_from_pred(model, samples, pred, mask, norm_scale_m: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RMSE in meters (masked-only and all-pixels) from MAE outputs.

    Assumes `samples` are globally normalized.

    - mean/std:  x_norm = (x - mean) / std       => RMSE_m = RMSE_norm * std
    - min/max:   x_norm = (x - min) / (max-min) => RMSE_m = RMSE_norm * (max-min)

    Important:
      `pred` is produced under autocast and can be float16/bfloat16.
      `target` from patchify(samples) is usually float32.
      We compute RMSE in float32 to avoid dtype mismatch and improve numerical stability.
    """
    target = model.patchify(samples)  # [N, L, P]

    # Compute metrics in float32 (safe + stable)
    pred_f = pred.float()
    target_f = target.float()

    # paste keep patches from target into pred
    # mask: 0 keep, 1 remove
    keep = (mask == 0)
    pred_paste = pred_f.clone()
    pred_paste[keep] = target_f[keep]

    mse = (pred_paste - target_f) ** 2
    mse = mse.mean(dim=-1)  # [N, L] mean over patch pixels

    mask_f = mask.float()
    mask_sum = mask_f.sum().clamp(min=1.0)

    rmse_mask = torch.sqrt((mse * mask_f).sum() / mask_sum)
    rmse_all = torch.sqrt(mse.mean())

    scale = torch.as_tensor(float(norm_scale_m), device=samples.device, dtype=rmse_mask.dtype)
    return rmse_mask * scale, rmse_all * scale


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
        samples = _unwrap_samples(batch)
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
                model, samples, pred, mask, norm_scale_m=getattr(args, "norm_scale_m", 1.0)
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
        samples = _unwrap_samples(batch)
        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=getattr(args, "amp", True)):
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        if args is not None and getattr(args, "log_rmse", False):
            rmse_mask_m, rmse_all_m = _rmse_meters_from_pred(
                model, samples, pred, mask, norm_scale_m=getattr(args, "norm_scale_m", 1.0)
            )
            metric_logger.update(rmse_m_mask=float(rmse_mask_m.item()))
            metric_logger.update(rmse_m_all=float(rmse_all_m.item()))

    metric_logger.synchronize_between_processes()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if log_writer is not None:
        log_writer.add_scalar(f'{prefix}_loss', stats['loss'], epoch)
        if args is not None and getattr(args, "log_rmse", False):
            log_writer.add_scalar(f'{prefix}_rmse_m_mask', stats.get('rmse_m_mask', float('nan')), epoch)
            log_writer.add_scalar(f'{prefix}_rmse_m_all', stats.get('rmse_m_all', float('nan')), epoch)

    print(f"{prefix} stats:", stats)
    return stats
