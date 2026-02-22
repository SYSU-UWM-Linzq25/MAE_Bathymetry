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
from pathlib import Path
from typing import Dict, Optional

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

    print_freq = 20
    best_val = float('inf')
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

        # save checkpoints
        if misc.is_main_process():
            _save_checkpoint(str(out_dir / f'checkpoint-{epoch:04d}.pth'), model_without_ddp, optimizer, loss_scaler, epoch, args, dem_norm)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                _save_checkpoint(str(out_dir / 'checkpoint-best.pth'), model_without_ddp, optimizer, loss_scaler, epoch, args, dem_norm)

            with open(out_dir / 'log.txt', 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'epoch': epoch,
                    'train': train_stats,
                    'val': val_stats,
                    'best_val': best_val,
                    'best_epoch': best_epoch,
                }) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if misc.is_main_process():
        print(f"Training time {total_time_str}; best_epoch={best_epoch} best_val_loss={best_val:.6f}")

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

    # test
    _maybe_eval('test', test_dir if os.path.isdir(test_dir) else '', args.test_list, 'test')

    # KY holdout or any extra
    if args.extra_eval_dir and os.path.isdir(args.extra_eval_dir):
        _maybe_eval('extra', args.extra_eval_dir, '', 'extra')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MAE pre-training (DEM GeoTIFF)', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
