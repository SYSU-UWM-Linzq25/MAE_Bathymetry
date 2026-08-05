# Representative bathymetry reconstruction figure

This folder reproduces the downstream-task result figure from the portable
`H054_AGU_SelectedReach_DataBundle` files. It does not require access to the
original `/tank/...` data tree.

## Run

On Mortimer, use the provided wrapper. It explicitly runs the same Python 3.12
environment used by the MAE training and evaluation workflow:

```bash
bash downstream_figure/run_figure.sh
```

The default interpreter is:

```text
/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn/bin/python
```

The wrapper prints the actual interpreter and package versions before drawing
the figure, so an old system Python cannot be used silently. To use another
compatible environment deliberately, override the interpreter explicitly:

```bash
PYTHON_BIN=/path/to/python bash downstream_figure/run_figure.sh
```

The default run creates:

- a 900 dpi PNG;
- a 900 dpi LZW-compressed TIFF for Word or journal submission;
- a vector PDF;
- an editable SVG, recommended for insertion into current versions of Word;
- a JSON audit containing the plotted metrics and display settings.

For the clearest result in Microsoft Word, insert the SVG directly with
**Insert > Pictures > This Device**. If a workflow does not accept SVG, use the
TIFF. Avoid copying the preview image from a browser or screenshot, because
Word then receives a lower-resolution copy.

## Useful options

```bash
# Use the original 0--2 m error range
PYTHON_BIN=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn/bin/python
"$PYTHON_BIN" downstream_figure/plot_representative_bathymetry.py \
  --data-dir downstream_figure/data/H054_AGU_SelectedReach_DataBundle \
  --output-dir downstream_figure/output_2m \
  --error-max-m 2

# Disable the compact display tilt
"$PYTHON_BIN" downstream_figure/plot_representative_bathymetry.py \
  --data-dir downstream_figure/data/H054_AGU_SelectedReach_DataBundle \
  --output-dir downstream_figure/output_no_tilt \
  --display-tilt-deg 0
```

The displayed reaches are rotated only for compact layout. Pixel size is
unchanged, all three panels in each row receive the same nearest-neighbor
rotation, and the exact same comparison mask, crop, array extent, and axis
limits are enforced for ground truth, prediction, and absolute error. No scale
bar is drawn.

## Suggested caption

**Figure 1. Representative bathymetry reconstruction for three rivers withheld
individually from model training. Ground truth and prediction share a common
elevation scale within each river. Absolute error is shown on a common 0--1 m
scale; values above 1 m use the saturated endpoint color. Within each river,
ground truth, prediction, and absolute error show the same spatial extent and
the same valid comparison pixels. Reaches are rotated only for compact
display.**
