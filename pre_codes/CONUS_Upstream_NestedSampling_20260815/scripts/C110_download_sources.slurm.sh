#!/usr/bin/env bash
#SBATCH --job-name=C110_CONUS_download
#SBATCH --partition=HydroIntel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/C110_CONUS_download_%j.out
#SBATCH --error=logs/C110_CONUS_download_%j.err
set -euo pipefail

SUBMIT_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck disable=SC1090
source "$SUBMIT_ROOT/scripts/C080_slurm_runtime.sh"
conus_setup_paths "$SUBMIT_ROOT"
PROJECT_ROOT=${PROJECT_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/CONUS_3DEP_NestedNative1m_20260815}
PARALLEL=${PARALLEL:-${SLURM_CPUS_PER_TASK:-8}}

mkdir -p "$DATA_ROOT/source_downloads"
bash "$SCRIPT_DIR/C030_download_selected_sources.sh" \
  "$DATA_ROOT/plan/download_manifest.tsv" \
  "$DATA_ROOT/source_downloads" \
  "$PARALLEL"
