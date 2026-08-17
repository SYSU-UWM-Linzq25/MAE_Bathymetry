#!/usr/bin/env bash
#SBATCH --job-name=C120_CONUS_prepare
#SBATCH --partition=HydroIntel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/C120_CONUS_prepare_%j.out
#SBATCH --error=logs/C120_CONUS_prepare_%j.err
set -euo pipefail

SUBMIT_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck disable=SC1090
source "$SUBMIT_ROOT/scripts/C080_slurm_runtime.sh"
conus_setup_paths "$SUBMIT_ROOT"
conus_activate_python
PROJECT_ROOT=${PROJECT_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/CONUS_3DEP_NestedNative1m_20260815}
PREP_FORMAT=${PREP_FORMAT:-VRT}
WORKERS=${WORKERS:-4}

mkdir -p "$DATA_ROOT/prepared_sources"
"$PYTHON_BIN" "$SCRIPT_DIR/C040_prepare_sources.py" \
  --download-manifest "$DATA_ROOT/plan/download_manifest.tsv" \
  --data-root "$DATA_ROOT/source_downloads" \
  --out-dir "$DATA_ROOT/prepared_sources" \
  --format "$PREP_FORMAT" \
  --workers "$WORKERS"
