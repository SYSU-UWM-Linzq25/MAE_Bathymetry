# CONUS upstream nested 3DEP sampling

Version: 2026-08-16 v10 (robust smoke-test sampling)

This package rebuilds the upstream DEM sampling workflow for all 48 conterminous U.S. states.
It keeps the terrain at its native analytical scale of 1 m and generates four concentric views
from every accepted random center:

| Label | Window | Ground extent at 1 m |
|---|---:|---:|
| S1 | 336 × 336 | 336 m × 336 m |
| S3 | 1008 × 1008 | 1008 m × 1008 m |
| S5 | 1680 × 1680 | 1680 m × 1680 m |
| S10 | 3360 × 3360 | 3360 m × 3360 m |

The target is exactly 1,000 accepted centers per state, or 48,000 centers in total. Kentucky is
handled exactly like every other state; there is no state holdout.

Data acquisition/sampling and train/validation assignment are deliberately separate. The download
and sampling workflow creates an unsplit 48-state sample pool. Only after that pool is complete and
passes sampling QA may the optional later processing step assign 80% training and 20% validation.
No test split is created.

## Why the old download snippet is not run once per random point

The earlier code has two reusable pieces:

1. the TNMAccess dataset name `Digital Elevation Model (DEM) 1 meter`;
2. resumable downloads with `wget -c`.

The following parts are replaced:

- `grep` is not used to parse JSON;
- API pagination is handled explicitly;
- the API is queried once per state bounding box, not once per proposed center;
- source URLs are deduplicated before download;
- repeated versions of the same reported footprint are reduced to the newest item;
- random centers are drawn only after selected source products are available;
- the entire S10 mask is checked, instead of checking one center pixel.

USGS Seamless 1-Meter DEM (S1M) is not used as the sole source in this version because production
is still in progress. The established TNMAccess 1 m product collection provides the broadest
practical basis for reaching 1,000 centers in every CONUS state. Every selected source is exposed
on a common EPSG:5070, north-up, 1 m grid before sampling.

## Random and overlap behavior

Sampling is random but spatially balanced in two stages.

1. For each state, source products are grouped into 20 km EPSG:5070 cells. The planner takes one
   randomly shuffled source from as many different cells as possible before taking a second source
   from any cell. The default is 150 candidate source products per state.
2. The state sampler visits those products in randomized round-robin order and draws random centers
   inside their valid interiors. It first tries a 3,024 m minimum center separation, then 1,512 m,
   then 756 m, and finally permits overlap while still rejecting identical centers.

The center point must lie inside its assigned state. The S10 footprint is allowed to cross a state
border because state assignment is center-based; it is accepted only when the complete 3360 × 3360
window contains valid DEM coverage. Thus small states can still reach 1,000 samples, while large
states remain geographically dispersed.

Only S10 is checked for validity. When S10 passes, S1/S3/S5 are guaranteed to be its concentric
subwindows. With the default `MIN_VALID_RATIO=1.0`, every pixel in S10 must be valid.

## Storage design

The default prepared source format and sample format are VRT:

- a prepared Warped VRT exposes each downloaded source as EPSG:5070 at 1 m;
- each S1/S3/S5/S10 sample is a small crop VRT that references its prepared source;
- raw source GeoTIFFs remain the only full pixel copies.

Four physical Float32 GeoTIFFs for 48,000 centers would be about 2.7 TiB before compression, in
addition to source downloads. Use `OUTPUT_MODE=GTiff` only when physical tile files are truly
required. VRTs work with GDAL/rasterio readers but must remain together with their referenced raw
and prepared source paths.

## Output tree

```text
Upstream_Model_ReTrain/
└── CONUS_3DEP_NestedNative1m_20260815/
    ├── boundaries/
    ├── inventory/                 # cached TNMAccess response per state
    ├── plan/
    │   ├── anchor_plan.csv
    │   └── download_manifest.tsv
    ├── source_downloads/raw/      # deduplicated USGS products
    ├── prepared_sources/
    │   ├── extracted/
    │   ├── prepared/              # EPSG:5070 1 m VRT/GeoTIFF sources
    │   └── source_index.csv
    ├── samples/
    │   ├── manifests/
    │   │   ├── centers_WI.csv
    │   │   └── nested_WI.csv
    │   └── tiles/{S1,S3,S5,S10}/WI/*.vrt
    ├── splits/
    │   ├── center_splits.csv
    │   └── lists/{S1,S3,S5,S10}/{train,val}.txt
    └── qa/
```

## Software

The Slurm scripts activate the Python environment inside each compute job. Do not run
`conda activate` on the submit node before `sbatch`. The defaults are:

```bash
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/conus_sampling_gdal
```

At job startup, the scripts print `CODE_ROOT`, the selected Python executable, its version, and the
GDAL version before doing any work.

The common runtime searches for `conda.sh` through `CONDA_SH`, `CONDA_EXE`, the current `conda`
executable, `~/miniconda3`, and `~/anaconda3`. If the cluster uses another conda installation, pass
its initialization file and environment name/path at submission time:

```bash
sbatch --export=ALL,CONDA_SH=/absolute/conda/etc/profile.d/conda.sh,CONDA_ENV=/absolute/conda/environment \
  scripts/C090_smoke_WI.slurm.sh
```

`CONDA_ENV` can also be the absolute path of the conda environment. If conda is provided by a
cluster module, pass `CONDA_MODULE=<module-name>` as another exported value. `C110` performs only
shell/wget downloads and therefore does not activate Python.

The provided `requirements.txt` lists Python dependencies. GeoPandas, Pandas, PyProj, and Shapely
are not required: state boundaries, geometric predicates, and coordinate transforms use the GDAL
Python package's OGR/OSR modules.
GDAL Python bindings must match the GDAL library installed on the HPC system.

Check the minimal runtime first:

```bash
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
ENV=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/conus_sampling_gdal
conda activate "$ENV"
python3 -c 'import requests, numpy, rasterio; from osgeo import gdal, ogr, osr; print("Required imports OK", gdal.VersionInfo())'
```

Only if that command reports a missing package, install or repair the minimal stack:

```bash
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
ENV=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/conus_sampling_gdal
conda install --prefix "$ENV" -c conda-forge \
  requests numpy rasterio gdal
```

Review conda's proposed transaction before confirming it. Package installation is a one-time
environment setup action; normal Slurm submissions still activate this environment inside the job.
The job preflight reports every missing or broken import in one run.

## First run: WI smoke test

Copy the package beneath `Upstream_Model_ReTrain`, enter the package directory, and submit:

```bash
cd /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/CONUS_Upstream_NestedSampling_20260815
mkdir -p logs
sbatch scripts/C090_smoke_WI.slurm.sh
```

`logs/` must exist before `sbatch`, because Slurm opens the files named by `#SBATCH --output` and
`#SBATCH --error` before the job body starts. Create it once in the package root as shown above.
The job itself is located through `SLURM_SUBMIT_DIR`; it never derives the project path from
`BASH_SOURCE`, because Slurm runs a temporary copy under `/var/spool/slurmd`.

The smoke test downloads five randomly selected WI source products and creates 20 centers × four
scales. It ends with:

```text
[C090] WI smoke PASS
```

Inspect the manifests and several VRTs before submitting the national run.

## Full CONUS data acquisition and sampling

Steps 1–6 below do not assign train/validation labels.

After the WI smoke test passes, the recommended full submission is the dependency-chain wrapper:

```bash
cd /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/CONUS_Upstream_NestedSampling_20260815
bash scripts/C170_submit_acquisition_pipeline.sh
```

`C170` checks the dedicated environment, obtains the state boundary file, and submits
`C100 -> C110 -> C120 -> C130 -> C140` with Slurm `afterok` dependencies. A downstream stage starts
only when the preceding stage succeeds. It stops after QA of the unsplit 48-state sampling pool;
it never submits `C150` or `C160`. The individual commands below remain available for inspection,
restarts, or manual execution.

### 1. Download official state boundaries

```bash
bash scripts/C000_download_state_boundaries.sh
```

### 2. Query the USGS inventory and plan 150 anchors per state

```bash
sbatch scripts/C100_inventory_and_plan.slurm.sh
```

This stage downloads metadata only. Review:

```bash
column -s, -t < \
  /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/CONUS_3DEP_NestedNative1m_20260815/plan/anchor_summary.csv \
  | less -S
```

### 3. Download only selected, deduplicated source products

```bash
sbatch scripts/C110_download_sources.slurm.sh
```

Rerunning is safe: completed files are skipped and `.part` files resume with `wget -c`.

### 4. Expose all sources as EPSG:5070, 1 m

```bash
sbatch scripts/C120_prepare_sources.slurm.sh
```

The default `PREP_FORMAT=VRT` avoids duplicating every source raster. To materialize prepared source
GeoTIFFs instead:

```bash
sbatch --export=ALL,PREP_FORMAT=GTiff scripts/C120_prepare_sources.slurm.sh
```

### 5. Sample 1,000 centers per state

```bash
sbatch scripts/C130_sample_states_array.slurm.sh
```

This submits a 48-task array, one state per task, with at most eight states running concurrently.
The script uses:

```text
TARGET_PER_STATE=1000
MIN_VALID_RATIO=1.0
OUTPUT_MODE=VRT
SEED=20260815
```

Every state task uses `--require-target`. If a state cannot reach 1,000 centers, that task exits
nonzero and preserves its diagnostics. Increase only that state's anchor inventory or rerun the
national plan with more reserve products, for example:

```bash
sbatch --export=ALL,ANCHORS_PER_STATE=250 scripts/C100_inventory_and_plan.slurm.sh
sbatch scripts/C110_download_sources.slurm.sh
sbatch scripts/C120_prepare_sources.slurm.sh
sbatch scripts/C130_sample_states_array.slurm.sh
```

Existing downloads and prepared sources are reused.

### 6. Verify the completed unsplit sampling pool

```bash
sbatch scripts/C140_verify_sampling.slurm.sh
```

This verifies 1,000 centers per state, the four required scales, concentric offsets, raster sizes,
1 m resolution, and readable sample files. It does not create or inspect a split. Expected line:

```text
QA PASS: states=48 centers=48000
```

At this point the USGS download and sampling task is complete.

## Later data processing: optional 80%/20% train/validation split

Run this only after the unsplit sample pool has passed `C140`:

```bash
sbatch scripts/C150_make_train_val_split.slurm.sh
```

This creates only `train` and `val`; there is no `test` set. Blocks are assigned approximately 80%
to training and 20% to validation on the national EPSG:5070 grid. The state name is not part of the
block hash, and all four scales from one center always share the same assignment. A 1,680 m guard
is applied where neighboring 33.6 km blocks receive different labels, so S10 windows from training
and validation cannot overlap across the block boundary.

The 80%/20% proportions are national approximate proportions, rather than forcing every state to
contain exactly 800 training and 200 validation centers.

Verify the later split separately:

```bash
sbatch scripts/C160_verify_train_val_split.slurm.sh
```

## Reproducibility and restart rules

- Random generators use `SEED` plus stable SHA-256-derived state keys; Python's process-randomized
  `hash()` is never used.
- Inventory JSON and the anchor plan are retained as an audit trail.
- Exact duplicate centers are never accepted, even after spacing is relaxed to zero.
- Repeating a completed state without `--overwrite` skips it when its center manifest already has
  the requested count.
- VRT paths are absolute by design. If the data tree is moved, rebuild prepared/sample VRTs or use
  `gdal_edit.py`/a controlled path-rewrite procedure before training.

## Script map

| Script | Purpose |
|---|---|
| `C000_download_state_boundaries.sh` | Download Census 2025 state polygons |
| `C010_query_tnm_inventory.py` | Paginated TNMAccess query and state cache |
| `C020_plan_anchor_downloads.py` | Random spatially balanced anchor selection and URL deduplication |
| `C030_download_selected_sources.sh` | Parallel resumable wget downloads |
| `C040_prepare_sources.py` | Extract and expose sources as EPSG:5070, 1 m |
| `C050_sample_nested_tiles.py` | Validate S10 and create concentric four-scale samples |
| `C060_make_spatial_splits.py` | Optional later national 80%/20% train/val split |
| `C070_verify_sampling.py` | Sampling QA, with optional train/val consistency checks |
| `C080_slurm_runtime.sh` | Resolve the submit directory and activate the in-job conda environment |
| `C090_smoke_WI.slurm.sh` | Small end-to-end WI test |
| `C100`–`C140` | USGS acquisition, sampling, and unsplit-pool QA |
| `C150_make_train_val_split.slurm.sh` | Separate later train/val assignment |
| `C160_verify_train_val_split.slurm.sh` | Separate train/val QA |
| `C170_submit_acquisition_pipeline.sh` | Submit C100–C140 as an afterok dependency chain; no split |
