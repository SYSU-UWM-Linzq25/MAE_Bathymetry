"""MAE pre-training for single-channel DEM GeoTIFF tiles.

Compared to the original `main_pretrain.py`, this script:
1) Reads **1-channel float** DEM tiles from GeoTIFF, not 3-channel JPEG.
2) Computes **global normalization** from the TRAIN split at start (mean/std or min/max).
3) Runs **validation evaluation** each epoch and saves the **best** checkpoint by val loss.
4) Optionally evaluates the final/best model on TEST (and extra eval dirs) using TRAIN normalization.

Intended data layout (recommended):
  <DATA_ROOT>/train/**/*.tif
  <DATA_ROOT>/val/**/*.tif
  <DATA_ROOT>/test/**/*.tif

You can also provide explicit file lists via --train_list / --val_list / --test_list.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import csv
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # tensorboard is optional
    SummaryWriter = None

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_pretrain import train_one_epoch, evaluate_one_epoch
from util.dem_dataset import DEMTileDataset, compute_dem_stats, load_json, save_json


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training (DEM GeoTIFF)', add_help=False)

    # --- data ---
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root folder containing train/val/test subfolders (or use *_list).')
    parser.add_argument('--train_dir', type=str, default='', help='Override train dir (else data_root/train)')
    parser.add_argument('--val_dir', type=str, default='', help='Override val dir (else data_root/val)')
    parser.add_argument('--test_dir', type=str, default='', help='Override test dir (else data_root/test)')
    parser.add_argument('--train_list', type=str, default='', help='TXT list of train GeoTIFF paths')
    parser.add_argument('--val_list', type=str, default='', help='TXT list of val GeoTIFF paths')
    parser.add_argument('--test_list', type=str, default='', help='TXT list of test GeoTIFF paths')
    parser.add_argument('--extra_eval_dir', type=str, default='',
                        help='An extra directory to evaluate at the end (e.g., KY holdout tiles).')

    parser.add_argument('--input_size', default=336, type=int,
                        help='images input size (must be divisible by patch size)')
    parser.add_argument('--in_chans', default=1, type=int,
                        help='Input channels. DEM should be 1.')

    # Nodata handling
    parser.add_argument('--nodata', default=-9999.0, type=float,
                        help='Nodata value in tiles (e.g., -9999). If set, nodata pixels are filled by tile mean.')

    # --- normalization ---
    parser.add_argument('--norm_method', default='meanstd', choices=['meanstd', 'minmax'],
                        help='Global normalization computed from TRAIN split.')
    parser.add_argument('--norm_json', default='', type=str,
                        help='If provided, load normalization stats from this JSON instead of computing.')
    parser.add_argument('--stats_max_files', default=0, type=int,
                        help='If >0, compute normalization using only first N train files (for quick experiments).')

    # --- model ---
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Normalize target patches per-sample (original MAE). For DEM, usually keep False.')
    parser.add_argument('--use_instance_norm', action='store_true', help='Apply InstanceNorm2d on input (per-tile). Usually OFF if you already use global normalization.')

    # --- training ---
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='gradient accumulation iterations')

    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    # evaluation
    parser.add_argument('--eval_rmse', action='store_true',
                        help='Also compute RMSE (masked + all) in meters during val/test evaluation.')

    # curves / early-stop helpers
    parser.add_argument('--history_csv', default='', type=str,
                        help='Where to save epoch-level history CSV (default: <output_dir>/history.csv).')
    parser.add_argument('--plot_every', default=1, type=int,
                        help='Update curve PNGs every N epochs (default: 1).')
    parser.add_argument('--no_plot_curves', action='store_true',
                        help='Disable generating curve PNGs during training.')
    parser.add_argument('--early_stop_patience', default=0, type=int,
                        help='If >0, enable early stopping when metric does not improve for N epochs.')
    parser.add_argument('--early_stop_metric', default='val_loss',
                        choices=['val_loss', 'val_rmse_m_mask', 'val_rmse_m_all'],
                        help='Metric to monitor for early stopping.')
    parser.add_argument('--early_stop_min_delta', default=0.0, type=float,
                        help='Minimum improvement to reset early stop patience.')
    parser.add_argument('--early_stop_warmup_epochs', default=0, type=int,
                        help='Do not allow early stopping until this epoch (warmup).')
    parser.add_argument('--early_stop_start_threshold', default=0.0, type=float,
                        help=('Only start counting early-stop patience after the monitored metric '
                              'reaches this threshold (e.g., RMSE <= threshold). 0 disables.'))

    # best checkpoint selection
    parser.add_argument('--best_metric', default='',
                        choices=['', 'val_loss', 'val_rmse_m_mask', 'val_rmse_m_all'],
                        help=('Metric used to save checkpoint-best.pth. '
                              'If empty: use val_rmse_m_mask when --eval_rmse, else val_loss.'))

    # misc
    parser.add_argument('--output_dir', default='./output_dem', type=str)
    parser.add_argument('--log_dir', default='./output_dem', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')

    # distributed
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def _resolve_split_dir(data_root: str, split: str, override: str) -> str:
    if override:
        return override
    return os.path.join(data_root, split)


def _save_checkpoint(path: str, model, optimizer, loss_scaler, epoch: int, args, dem_norm: Dict):
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'args': vars(args),
        'dem_norm': dem_norm,
    }
    misc.save_on_master(to_save, path)


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float('nan')


def _load_history(csv_path: Path) -> List[Dict[str, float]]:
    if not csv_path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: _safe_float(v) for k, v in r.items()})
    return rows


def _save_history(csv_path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, float('nan')) for k in fieldnames})


def _maybe_plot_curves(history: List[Dict[str, float]], out_dir: Path, plot_rmse: bool) -> None:
    """Generate/overwrite curve PNGs from history.

    This is best-effort: if matplotlib is not available, training continues.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib not available; skip plotting curves. ({e})")
        return

    if not history:
        return

    epochs = [int(r.get('epoch', 0)) for r in history]

    def _plot_one(y_train_key: str, y_val_key: str, ylabel: str, out_name: str):
        y_tr = [r.get(y_train_key, float('nan')) for r in history]
        y_va = [r.get(y_val_key, float('nan')) for r in history]
        if all(np.isnan(y_tr)) and all(np.isnan(y_va)):
            return
        plt.figure()
        plt.plot(epochs, y_tr, label='train')
        plt.plot(epochs, y_va, label='val')
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / out_name, dpi=150)
        plt.close()

    _plot_one('train_loss', 'val_loss', 'loss (MSE, normalized)', 'curve_loss.png')

    if plot_rmse:
        _plot_one('train_rmse_m_mask', 'val_rmse_m_mask', 'RMSE (m) on masked patches', 'curve_rmse_mask.png')
        _plot_one('train_rmse_m_all', 'val_rmse_m_all', 'RMSE (m) on pasted full tile', 'curve_rmse_all.png')


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # ---- build datasets (no normalization yet) ----
    train_dir = _resolve_split_dir(args.data_root, 'train', args.train_dir)
    val_dir = _resolve_split_dir(args.data_root, 'val', args.val_dir)
    test_dir = _resolve_split_dir(args.data_root, 'test', args.test_dir)

    train_ds = DEMTileDataset(
        dir_path=train_dir if not args.train_list else None,
        list_path=args.train_list if args.train_list else None,
        input_size=args.input_size,
        nodata=args.nodata,
        random_flip=True,
        return_path=False,
    )
    val_ds = DEMTileDataset(
        dir_path=val_dir if not args.val_list else None,
        list_path=args.val_list if args.val_list else None,
        input_size=args.input_size,
        nodata=args.nodata,
        random_flip=False,
        return_path=False,
    )

    # ---- compute / load global normalization (TRAIN only) ----
    norm_path = args.norm_json
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not norm_path:
        norm_path = str(out_dir / 'norm_stats_train.json')

    if args.norm_json and os.path.isfile(args.norm_json):
        dem_norm = load_json(args.norm_json)
        if misc.is_main_process():
            print(f"[NORM] loaded from: {args.norm_json}")
    else:
        if misc.is_main_process():
            max_files = args.stats_max_files if args.stats_max_files and args.stats_max_files > 0 else None
            dem_norm = compute_dem_stats(
                train_ds.files,
                nodata=args.nodata,
                max_files=max_files,
                method=args.norm_method,
            )
            if args.nodata is None and float(dem_norm.get("min", 0)) <= -9000:
                raise ValueError(
                    f"[NORM] Detected extreme min={dem_norm['min']} (likely nodata). "
                    f"Please rerun with --nodata -9999 (or correct nodata)."
                )

            save_json(dem_norm, norm_path)
            print(f"[NORM] computed from train split and saved to: {norm_path}")
        else:
            dem_norm = {}

        # broadcast to all ranks
        if args.distributed:
            obj_list = [dem_norm]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            dem_norm = obj_list[0]

    # attach normalization to datasets
    if args.norm_method == 'meanstd':
        mean = float(dem_norm['mean'])
        std = float(dem_norm['std'])
        train_ds.set_norm(mean, std)
        val_ds.set_norm(mean, std)
    else:
        # minmax scaling
        vmin = float(dem_norm['min'])
        vmax = float(dem_norm['max'])
        train_ds.set_norm(vmin, vmax, method='minmax')
        val_ds.set_norm(vmin, vmax, method='minmax')

    # also store on args for RMSE(m) conversion
    args.dem_norm = dem_norm
    args.norm_mean = float(dem_norm.get('mean', 0.0))
    args.norm_std = float(dem_norm.get('std', 1.0))
    # engine_pretrain uses args.log_rmse
    args.log_rmse = bool(args.eval_rmse)
    # meters-scale factor depends on normalization method
    if args.norm_method == 'meanstd':
        args.norm_scale_m = args.norm_std
    else:
        args.norm_scale_m = float(dem_norm.get('max', 1.0)) - float(dem_norm.get('min', 0.0))

    # ---- samplers & loaders ----
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        train_sampler = torch.utils.data.DistributedSampler(train_ds, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_ds, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_ds)
        val_sampler = torch.utils.data.SequentialSampler(val_ds)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # ---- model ----
    model = models_mae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        img_size=args.input_size,
        in_chans=args.in_chans,
        use_instance_norm=args.use_instance_norm,
    )
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # ---- optimizer ----
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if misc.is_main_process():
        print(f"base_lr: {args.blr:.2e}")
        print(f"actual_lr: {args.lr:.2e}")
        print(f"effective_batch_size: {eff_batch_size}")

    import timm.optim.optim_factory as optim_factory
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()

    # ---- resume ----
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        if misc.is_main_process():
            print(f"[RESUME] loaded: {args.resume} (start_epoch={args.start_epoch})")

    # ---- logging ----
    log_writer = None
    if misc.is_main_process() and SummaryWriter is not None:
        log_writer = SummaryWriter(log_dir=args.log_dir)

    # history & curve outputs (main process only)
    history_csv = Path(args.history_csv) if args.history_csv else (out_dir / 'history.csv')
    history: List[Dict[str, float]] = _load_history(history_csv) if misc.is_main_process() else []

    # early stopping state
    best_metric = float('inf')
    bad_epochs = 0
    es_activated = False

    print_freq = 20
    # select best checkpoint metric
    best_metric_name = args.best_metric
    if best_metric_name == '':
        best_metric_name = 'val_rmse_m_mask' if args.eval_rmse else 'val_loss'

    best_val = float('inf')  # best value for best_metric_name
    best_epoch = -1

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args
        )

        val_stats = evaluate_one_epoch(
            model, val_loader, device, epoch, log_writer=log_writer, args=args, prefix='val'
        )

        val_loss = float(val_stats.get('loss', float('inf')))

        # metric used for selecting checkpoint-best
        if best_metric_name == 'val_loss':
            val_best_metric = val_loss
        else:
            key = best_metric_name.replace('val_', '')
            val_best_metric = float(val_stats.get(key, float('inf')))
            if not np.isfinite(val_best_metric):
                if misc.is_main_process():
                    print(f"[WARN] best_metric={best_metric_name} unavailable (did you forget --eval_rmse?). Fallback to val_loss.")
                best_metric_name = 'val_loss'
                val_best_metric = val_loss

        # ---- epoch-level tensorboard (clean curves) ----
        if log_writer is not None and misc.is_main_process():
            log_writer.add_scalar('epoch/train_loss', float(train_stats.get('loss', float('nan'))), epoch)
            log_writer.add_scalar('epoch/val_loss', val_loss, epoch)
            if args.eval_rmse:
                log_writer.add_scalar('epoch/train_rmse_m_mask', float(train_stats.get('rmse_m_mask', float('nan'))), epoch)
                log_writer.add_scalar('epoch/val_rmse_m_mask', float(val_stats.get('rmse_m_mask', float('nan'))), epoch)
                log_writer.add_scalar('epoch/train_rmse_m_all', float(train_stats.get('rmse_m_all', float('nan'))), epoch)
                log_writer.add_scalar('epoch/val_rmse_m_all', float(val_stats.get('rmse_m_all', float('nan'))), epoch)

        # save checkpoints
        if misc.is_main_process():
            _save_checkpoint(str(out_dir / f'checkpoint-{epoch:04d}.pth'), model_without_ddp, optimizer, loss_scaler, epoch, args, dem_norm)

            if val_best_metric < best_val:
                best_val = val_best_metric
                best_epoch = epoch
                _save_checkpoint(str(out_dir / 'checkpoint-best.pth'), model_without_ddp, optimizer, loss_scaler, epoch, args, dem_norm)

            with open(out_dir / 'log.txt', 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'epoch': epoch,
                    'train': train_stats,
                    'val': val_stats,
                    'best_metric': best_metric_name,
                    'best_metric_value': best_val,
                    'best_epoch': best_epoch,
                }) + '\n')

            # ---- update history.csv (epoch-level) ----
            row = {
                'epoch': float(epoch),
                'train_loss': float(train_stats.get('loss', float('nan'))),
                'val_loss': float(val_stats.get('loss', float('nan'))),
                'lr': float(train_stats.get('lr', float('nan'))),
            }
            if args.eval_rmse:
                row.update({
                    'train_rmse_m_mask': float(train_stats.get('rmse_m_mask', float('nan'))),
                    'val_rmse_m_mask': float(val_stats.get('rmse_m_mask', float('nan'))),
                    'train_rmse_m_all': float(train_stats.get('rmse_m_all', float('nan'))),
                    'val_rmse_m_all': float(val_stats.get('rmse_m_all', float('nan'))),
                })

            # replace existing epoch row if present (safe for resume)
            history = [r for r in history if int(r.get('epoch', -1)) != epoch]
            history.append(row)
            history.sort(key=lambda r: int(r.get('epoch', 0)))

            fieldnames = ['epoch', 'lr', 'train_loss', 'val_loss']
            if args.eval_rmse:
                fieldnames += ['train_rmse_m_mask', 'val_rmse_m_mask', 'train_rmse_m_all', 'val_rmse_m_all']
            _save_history(history_csv, history, fieldnames)

            # ---- plot curves (best-effort) ----
            if (not args.no_plot_curves) and (args.plot_every > 0) and (epoch % args.plot_every == 0):
                _maybe_plot_curves(history, out_dir, plot_rmse=bool(args.eval_rmse))

        # ---- optional early stopping ----
        if args.early_stop_patience and args.early_stop_patience > 0:
            metric_name = args.early_stop_metric
            if metric_name == 'val_loss':
                cur = float(val_stats.get('loss', float('inf')))
            else:
                # val_stats uses keys: rmse_m_mask / rmse_m_all
                cur = float(val_stats.get(metric_name.replace('val_', ''), float('inf')))
                if not np.isfinite(cur):
                    cur = float(val_stats.get('loss', float('inf')))
                    metric_name = 'val_loss'

            # Activate early-stop only after:
            #   1) warmup epochs are finished, AND
            #   2) (optional) metric reaches a target threshold, e.g., RMSE <= threshold
            warmup_ok = epoch >= int(getattr(args, 'early_stop_warmup_epochs', 0) or 0)
            thr = float(getattr(args, 'early_stop_start_threshold', 0.0) or 0.0)
            thr_ok = True
            if thr > 0:
                thr_ok = cur <= thr

            if not es_activated:
                if warmup_ok and thr_ok:
                    es_activated = True
                    best_metric = cur
                    bad_epochs = 0
                    if misc.is_main_process():
                        if thr > 0:
                            print(f"[EARLY-STOP] Activated at epoch={epoch} (metric={metric_name}={cur:.4f} <= {thr}).")
                        else:
                            print(f"[EARLY-STOP] Activated at epoch={epoch} (metric={metric_name}={cur:.4f}).")
                else:
                    # Not activated yet: do not accumulate patience.
                    pass
            else:
                if cur < best_metric - float(args.early_stop_min_delta):
                    best_metric = cur
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(args.early_stop_patience):
                        if misc.is_main_process():
                            print(f"[EARLY-STOP] No improvement in {metric_name} for {bad_epochs} epochs. Stop at epoch={epoch}. best={best_metric:.4f}")
                        break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if misc.is_main_process():
        print(f"Training time {total_time_str}; best_epoch={best_epoch} best_{best_metric_name}={best_val:.6f}")

    # ---- final evaluations (optional) ----
    def _maybe_eval(split_name: str, dir_path: str, list_path: str, out_tag: str):
        if not dir_path and not list_path:
            return
        ds = DEMTileDataset(dir_path=dir_path if dir_path else None,
                            list_path=list_path if list_path else None,
                            input_size=args.input_size,
                            nodata=args.nodata,
                            random_flip=False,
                            return_path=False)
        # apply same TRAIN normalization
        if args.norm_method == 'meanstd':
            ds.set_norm(args.norm_mean, args.norm_std)
        else:
            ds.set_norm(float(dem_norm['min']), float(dem_norm['max']), method='minmax')

        if args.distributed:
            sampler = torch.utils.data.DistributedSampler(ds, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(ds)

        loader = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=args.batch_size,
                                             num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
        stats = evaluate_one_epoch(model, loader, device, epoch=args.epochs, log_writer=None, args=args, prefix=out_tag)
        if misc.is_main_process():
            save_json(stats, str(out_dir / f'eval_{out_tag}.json'))
            print(f"[EVAL] {split_name} -> {stats}")

    # test (optional; if you don't pass test_dir/test_list, this does nothing)
    _maybe_eval('test', test_dir if os.path.isdir(test_dir) else '', args.test_list, 'test')

    # KY holdout or any extra (optional)
    if args.extra_eval_dir and os.path.isdir(args.extra_eval_dir):
        _maybe_eval('extra', args.extra_eval_dir, '', 'extra')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MAE pre-training (DEM GeoTIFF)', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
