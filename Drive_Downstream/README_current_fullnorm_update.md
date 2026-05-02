# Stage2 bathy + LCC current-task update

This update is for the current validation experiment where the full bathymetry tile is treated as known during training/validation. It keeps exact LCC masking for the MAE reconstruction target, but removes `--tile_norm_visible_only` from the smoke and full training scripts.

Use these scripts instead of the previous exact decoder scripts:

```bash
sbatch Downstream_Task_Bathy/scripts/C001_smoke_stage2_bathy_lcc_exact_decoder_fullnorm.sh
sbatch Downstream_Task_Bathy/scripts/D001_train_stage2_bathy_lcc_exact_decoder_fullnorm.sh
```

Optional safety patch: `decoder_only_safety_patch.diff` adds an assertion that, when `--freeze_encoder --freeze_last_n_encoder_blocks 0` is used, only decoder-side parameters remain trainable.
