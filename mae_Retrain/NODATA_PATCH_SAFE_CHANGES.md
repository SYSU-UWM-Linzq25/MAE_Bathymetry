# NoData-safe exact-mask revision

## Detection
A pixel is invalid when it is non-finite, matches `--nodata`, or is less than or equal to `--nodata_threshold` (default `-9999`). This catches common sentinels including `-9999`, `-99999`, and `-999999`.

## Patch states
For each 16x16 patch:

- **visible**: every pixel is valid and the patch contains no final-mask pixel;
- **prediction**: every pixel is valid and the patch contains at least one final-mask pixel;
- **ignored**: the patch contains at least one invalid/NoData pixel.

Ignored patches are absent from encoder input and decoder attention. They do not participate in loss or RMSE and are written as NoData in visual outputs.

## Tile filtering
`--min_valid_visible_patch_ratio` removes tiles that do not retain enough valid known context after NoData patches are removed. A conservative starting value is `0.70` for 336x336 tiles with patch size 16.

The existing `--min_lcc_patch_ratio` and `--max_lcc_patch_ratio` now apply to usable prediction patches after NoData-patch removal.

## Visualization
GT, prediction, reconstruction, and error GeoTIFFs preserve invalid pixels as the configured NoData value. A `*_valid_mask.tif` is also written.

## Important
Use a new output directory/run name. Do not resume the old NoData-contaminated downstream checkpoints for the final comparison.
