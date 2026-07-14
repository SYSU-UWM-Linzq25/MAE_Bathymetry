A100 — Upstream/Downstream MAE tile elevation statistics
=========================================================

Purpose
-------
Generate distribution statistics first, without running the model and without
mixing prediction-error metrics into the diagnosis.

Definitions used
----------------
1. Upstream
   - Region: ALL valid DEM pixels in each listed tile.
   - No fixed random MAE mask is applied because the upstream random mask is
     generated during training and changes; the stable data distribution is
     the full valid tile.
   - Normalization center: all-valid-pixel mean.
   - Normalization denominator: max(all-valid-pixel std * 1.0, 1e-3).

2. Downstream
   - known:  valid DEM AND Hidden_Mask is defined AND Hidden_Mask < 0.5.
   - masked: valid DEM AND Hidden_Mask is defined AND Hidden_Mask >= 0.5.
   - loss:   valid DEM AND Loss_Mask_Pixel is defined AND Loss_Mask_Pixel >= 0.5.
   - Normalization center and denominator always come from known pixels:
       known_mean
       max(known_std * 1.5, 1e-3)
   - The same known-based normalization is then applied to masked/loss pixels.

The downstream Loss_Mask_Pixel result is included as an additional diagnostic.
It is not substituted for the full Hidden_Mask region.  This prevents the
model-hidden region and the actual supervised-loss subset from being confused.

Statistics produced for every region
-------------------------------------
Raw elevation in meters:
  count, min, max, mean, population std (ddof=0), p01, p05, p50, p95, p99

Normalized elevation:
  count, min, max, mean, population std (ddof=0), p01, p05, p50, p95, p99

Files
-----
A100_collect_mae_tile_statistics.py
  Main CPU-only Python program.

A101_run_upstream_tile_statistics.sh
  Slurm wrapper for the exact upstream Small_tilenorm_viscorr_336 split files:
    train.txt
    val.txt
    holdout_KY.txt (kept as a separate comparison label)

A102_run_downstream_tile_statistics.sh
  Slurm wrapper that scans the complete MAE-v2 1 m tile collection:
    Train_tile
    Hidden_Mask
    Loss_Mask_Pixel

Expected output
---------------
Z997_MAE_Tile_Statistics_20260711/
  Upstream_AllValid_ByState/
    upstream_tile_stats.csv
    upstream_state_summary.csv
    upstream_split_summary.csv
    by_state/<STATE>_tile_stats.csv
    by_split/<SPLIT>_tile_stats.csv
    run_config.json
    errors.csv                         only when problems exist

  Downstream_Known_Masked_ByRiver/
    downstream_tile_stats.csv
    downstream_river_summary.csv
    by_river/<RIVER>_tile_stats.csv
    run_config.json
    errors.csv                         only when problems exist

Important columns in upstream_tile_stats.csv
---------------------------------------------
split, state, tile_path
all_raw_min, all_raw_max, all_raw_mean, all_raw_std
normalization_mean_m, normalization_denominator_m
all_norm_min, all_norm_max, all_norm_mean, all_norm_std

Important columns in downstream_tile_stats.csv
-----------------------------------------------
river, tile_id, tile_path
known_raw_min/max/mean/std
masked_raw_min/max/mean/std
loss_raw_min/max/mean/std
normalization_mean_m, normalization_denominator_m
known_norm_min/max/mean/std
masked_norm_min/max/mean/std
loss_norm_min/max/mean/std
known_pixel_count, masked_pixel_count, loss_pixel_count
masked_nonloss_count, loss_outside_masked_count

Installation on HPC
-------------------
Place A100_collect_mae_tile_statistics.py in both locations, or change SCRIPT
in the wrappers:

  /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/
    Upstream_Model_ReTrain/scripts/A100_collect_mae_tile_statistics.py

  /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/
    Downstream_Task_Bathy/script/A100_collect_mae_tile_statistics.py

Suggested commands
------------------
sbatch A101_run_upstream_tile_statistics.sh
sbatch A102_run_downstream_tile_statistics.sh

The two jobs are independent and can run at the same time.  They need CPU and
raster I/O only; no checkpoint or GPU is used.

Validation behavior
-------------------
- Every upstream list path is resolved and checked.
- Downstream DEM/Hidden/Loss files are paired by resolution + river + tile ID.
- Missing pairs, duplicate keys, unreadable rasters, shape mismatches, and tiles
  with insufficient normalization pixels are written to errors.csv.
- With --fail-on-error, the Slurm job exits nonzero after preserving outputs,
  making incomplete statistics visible rather than silently accepted.
