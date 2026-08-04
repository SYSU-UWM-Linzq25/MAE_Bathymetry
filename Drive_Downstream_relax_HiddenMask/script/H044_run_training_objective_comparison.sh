#!/usr/bin/env bash
#SBATCH -J H044_obj_train
#SBATCH -p HydroIntel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Analysis only. This script does not train or rerun any F-series workflow.
#
# Step 1:
# Compare the two single-stage objectives under strict and relaxed masks:
#   - normalized objective
#   - meter-domain objective
#
# Outputs include validation MAE/RMSE, optimization-loss curves, meter-MAE
# curves, and per-river/macro CSV summaries.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
SCRIPT=${SCRIPT:-$RELAX_ROOT/script/H044_compare_training_objectives.py}

STRICT_NORMALIZED_ROOT=${STRICT_NORMALIZED_ROOT:-$ROOT/Downstream_Task_Bathy/cross_validation_v2}
STRICT_METER_ROOT=${STRICT_METER_ROOT:-$ROOT/Downstream_Task_Bathy/cross_validation_v4_meterMAE_BaselineEval}
OUT_DIR=${OUT_DIR:-$RELAX_ROOT/results/H044_ObjectiveComparison_TrainVal}
OVERWRITE=${OVERWRITE:-0}
DPI=${DPI:-220}

CONDA_SH=${CONDA_SH:-/home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-$ROOT/conda_envs/mae_zequn}
PYTHON_BIN=${PYTHON_BIN:-$CONDA_ENV/bin/python}

[[ -f "$CONDA_SH" ]] || {
  echo "[ERROR] Missing conda initialization script: $CONDA_SH" >&2
  exit 2
}
[[ -x "$PYTHON_BIN" ]] || {
  echo "[ERROR] Python executable is missing or not executable: $PYTHON_BIN" >&2
  exit 2
}

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$LOG_DIR/H044_training_objective_comparison_${JOB_ID}.out" \
     2>"$LOG_DIR/H044_training_objective_comparison_${JOB_ID}.err"

module purge || true
source "$CONDA_SH"
conda activate "$CONDA_ENV"

for path in "$SCRIPT" "$STRICT_NORMALIZED_ROOT" "$STRICT_METER_ROOT" "$RELAX_ROOT/results/NormOnly" "$RELAX_ROOT/results/MeterOnly"; do
  [[ -e "$path" ]] || { echo "[ERROR] Missing: $path" >&2; exit 2; }
done

if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -not -path "$OUT_DIR/logs*" -print -quit | grep -q .; then
  if [[ "$OVERWRITE" == "1" ]]; then
    find "$OUT_DIR" -mindepth 1 -maxdepth 1 ! -name logs -exec rm -rf {} +
  else
    echo "[ERROR] Output exists: $OUT_DIR" >&2
    echo "Set OVERWRITE=1 to rebuild analysis figures." >&2
    exit 3
  fi
fi
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "H044 analysis only: normalized vs meter objective"
date
echo "STRICT_NORMALIZED_ROOT=$STRICT_NORMALIZED_ROOT"
echo "STRICT_METER_ROOT=$STRICT_METER_ROOT"
echo "RELAX_ROOT=$RELAX_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "============================================================"

"$PYTHON_BIN" -u "$SCRIPT" \
  --strict_normalized_root "$STRICT_NORMALIZED_ROOT" \
  --strict_meter_root "$STRICT_METER_ROOT" \
  --relax_root "$RELAX_ROOT" \
  --output_dir "$OUT_DIR" \
  --dpi "$DPI"

echo
echo "=== Metric source audit ==="
column -s, -t < "$OUT_DIR/H044_metric_source_audit.csv" | sed -n '1,30p' ||   cat "$OUT_DIR/H044_metric_source_audit.csv"

echo "=== DONE H044 ==="
echo "$OUT_DIR"
date
