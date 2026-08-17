#!/usr/bin/env bash
# Submit the complete CONUS acquisition/sampling chain through sampling QA.
# Run this directly with bash from the package root; do not submit C170 with sbatch.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
CODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
CONDA_SH=${CONDA_SH:-/home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/conus_sampling_gdal}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "[C170] run this submission wrapper on the submit node, not inside a Slurm job" >&2
  exit 2
fi
if [[ ! -f "$CONDA_SH" || ! -x "$CONDA_ENV/bin/python3" ]]; then
  echo "[C170] missing conda runtime: CONDA_SH=$CONDA_SH CONDA_ENV=$CONDA_ENV" >&2
  exit 2
fi

cd "$CODE_ROOT"
mkdir -p logs

echo "[C170] checking $CONDA_ENV"
set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
PYTHON_BIN=$(command -v python3)
"$PYTHON_BIN" -c \
  'import requests, numpy, rasterio; from osgeo import gdal, ogr, osr; print(f"[C170] Python environment OK; GDAL={gdal.VersionInfo()} rasterio={rasterio.__version__}")'

echo "[C170] downloading/checking state boundaries"
bash "$SCRIPT_DIR/C000_download_state_boundaries.sh"

export CONDA_SH CONDA_ENV

job_inventory=$(sbatch --parsable "$SCRIPT_DIR/C100_inventory_and_plan.slurm.sh")
job_inventory=${job_inventory%%;*}
job_download=$(sbatch --parsable --dependency="afterok:$job_inventory" "$SCRIPT_DIR/C110_download_sources.slurm.sh")
job_download=${job_download%%;*}
job_prepare=$(sbatch --parsable --dependency="afterok:$job_download" "$SCRIPT_DIR/C120_prepare_sources.slurm.sh")
job_prepare=${job_prepare%%;*}
job_sample=$(sbatch --parsable --dependency="afterok:$job_prepare" "$SCRIPT_DIR/C130_sample_states_array.slurm.sh")
job_sample=${job_sample%%;*}
job_qa=$(sbatch --parsable --dependency="afterok:$job_sample" "$SCRIPT_DIR/C140_verify_sampling.slurm.sh")
job_qa=${job_qa%%;*}

echo "[C170] submitted acquisition/sampling chain"
echo "[C170] C100 inventory/plan : $job_inventory"
echo "[C170] C110 downloads      : $job_download (afterok:$job_inventory)"
echo "[C170] C120 prepare        : $job_prepare (afterok:$job_download)"
echo "[C170] C130 state array    : $job_sample (afterok:$job_prepare)"
echo "[C170] C140 sampling QA    : $job_qa (afterok:$job_sample)"
echo "[C170] train/val split was not submitted; C150/C160 remain separate later steps"
