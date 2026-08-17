#!/usr/bin/env bash
#SBATCH --job-name=C140_CONUS_sampleqa
#SBATCH --partition=HydroIntel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/C140_CONUS_sampleqa_%j.out
#SBATCH --error=logs/C140_CONUS_sampleqa_%j.err
# Verify downloaded/prepared/sampled products only. No train/val split is created here.
set -euo pipefail

SUBMIT_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck disable=SC1090
source "$SUBMIT_ROOT/scripts/C080_slurm_runtime.sh"
conus_setup_paths "$SUBMIT_ROOT"
conus_activate_python
PROJECT_ROOT=${PROJECT_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/CONUS_3DEP_NestedNative1m_20260815}
TARGET_PER_STATE=${TARGET_PER_STATE:-1000}

mkdir -p "$DATA_ROOT/qa"
"$PYTHON_BIN" "$SCRIPT_DIR/C070_verify_sampling.py" \
  --sampling-root "$DATA_ROOT/samples" \
  --target-per-state "$TARGET_PER_STATE" \
  --report "$DATA_ROOT/qa/sampling_qa_errors.csv"

echo "[C140] sampling QA PASS"
