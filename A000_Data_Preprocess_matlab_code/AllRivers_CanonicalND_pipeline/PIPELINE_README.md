# All-river Canonical NoData pipeline

## Why this chain is unit-safe

The pipeline starts from:

`Processed_Results/Bathy_1m_FixND/<river>/Bathy_1m.vrt`

It does **not** start from raw bathymetry and does **not** multiply elevations.

Therefore the existing corrections remain intact:

- `OR_MKRC_Topobathy_2021`
  - vertical elevation feet -> meters;
  - horizontal 2 ft grid -> true 1 m grid;
  - produced by `B001_02_fix_OR_MKRC_unit_only.m`.

- `KewaFix2Null`
  - vertical elevation feet -> meters;
  - horizontal grid already 1 m;
  - produced by `B001_05_fix_KewaFix2Null_unit_only.m`.

The master driver checks sentinel files for these corrections and refuses to run
if they are missing.

## Production chain

### One-time raw/unit preparation

1. General raw bathymetry -> `Bathy_1m_FixND`.
2. Run unit diagnostics:
   - `B001_03_Check_Bathy_3DEP_Unit_Ratio.m`
   - `B001_04_Check_Kewa_Point_Unit_Compare.m`
3. Apply the two confirmed unit fixes:
   - `B001_02_fix_OR_MKRC_unit_only.m`
   - `B001_05_fix_KewaFix2Null_unit_only.m`
4. Rebuild LCC to the corrected bathy grid:
   - `B002_Rebuild_LCC_To_FinalBathyGrid_All.m`
5. Check grids:
   - `B002c_Check_LCC_BathyGrid_All.m`

Do not rerun the generic raw bathymetry section afterward in a way that
overwrites the two corrected `Bathy_1m_FixND` products.

### Canonical NoData rebuild for all 12 MAE rivers

Run from MATLAB:

```matlab
patchDir = '/path/to/AllRivers_CanonicalND_pipeline';
addpath(patchDir);

Summary = B000_run_AllTrainingRivers_CanonicalND( ...
    'overwrite', true, ...
    'continueOnError', false);
```

The driver executes:

1. `B001_10_Canonicalize_Bathy_ForRiver`
   - preserves elevations and grids;
   - canonicalizes invalid pixels to `-999999`;
   - always writes Float32;
   - uses `-srcnodata` and `-vrtnodata`.

2. `B001_12_Rebuild_Bathy3DEP_Merge_ForRiver`
   - explicit bathy-priority merge;
   - output initialized to `-999999`;
   - 3DEP only fills invalid bathy;
   - both invalid remains `-999999`.

3. `B003s_10_Build_SimpleFinalMask_FromCanonicalBathy`
   - `final_mask = (LCC == 1) & bathy_valid`.

4. `B005b_10_Reextract_SelectedTiles_CanonicalND`
   - reuses existing selected center-point shapefiles;
   - PointID and filenames do not change;
   - no random resampling;
   - output is Float32 with `-999999`;
   - `TileOutRiver` uses `-999999`, not NaN.

Staged products:

- `Bathy_1m_CanonicalND`
- `Bathy3DEP_Merged_Tiff_1m_CanonicalND`
- `PredictionMask_LCCBathyValid_1m_CanonicalND`
- `Tiles_for_MAE_CanonicalND/Tiles_1m`

## Zero policy

Do not globally apply `value == 0 -> NoData`.

Zero can be a real elevation for coastal/low-elevation products. The registry
currently uses `ZeroIsNoData=false` for all rivers. Source-declared NoData,
NaN, Inf, and `-999999` are still canonicalized.

The Santiam source bathymetry audit found `N_zero_reclassified=0`; its old
tile zeros were caused by the old NoData write/VRT chain. Explicit output
initialization and metadata are the permanent fix.

## Auditing before publication

```bash
python B006_compare_old_new_tiles_all.py \
  --old_base /tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE/Tiles_1m \
  --new_base /tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_CanonicalND/Tiles_1m \
  --output_dir /tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Z021_CanonicalND_OldNew_Audit
```

Review:

- filename/count agreement;
- `N_new_zero`;
- `N_old_zero_to_new_nodata`;
- `N_common_valid_changed`;
- `N_mask_changed`;
- maximum difference on common valid pixels.

Expected:
- PointIDs and file counts match exactly;
- common valid elevations should normally remain unchanged;
- old false-zero backgrounds may become `-999999`;
- masks may change only where old NoData handling was wrong.

## Publish

Only after the audit passes:

```bash
bash B007_publish_CanonicalND_tiles_to_MAE.sh
```

The script validates counts, backs up all current MAE tiles/masks, and then
copies the staged files into the official MAE Data directories.

After publication:

1. rerun A004 train/val audit;
2. start a new downstream run from the upstream checkpoint;
3. do not resume a checkpoint trained with the old inputs;
4. rerun the per-tile evaluation.
