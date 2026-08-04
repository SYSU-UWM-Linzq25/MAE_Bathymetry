#!/usr/bin/env bash
#SBATCH -J H047_report
#SBATCH -p HydroIntel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
SCRIPT=${SCRIPT:-$RELAX_ROOT/script/H047_build_analysis_report.py}

TRAINING_DIR=${TRAINING_DIR:-$RELAX_ROOT/results/H044_ObjectiveComparison_TrainVal}
FULLRIVER_DIR=${FULLRIVER_DIR:-$RELAX_ROOT/results/H045_FullRiver_CommonLossPixel_Analysis}
REACH_DIR=${REACH_DIR:-$RELAX_ROOT/results/H046_LocalReach_6Panel_Analysis}
OUT_DIR=${OUT_DIR:-$RELAX_ROOT/results/H047_Analysis_Report}
OVERWRITE=${OVERWRITE:-0}

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
exec >"$LOG_DIR/H047_report_${JOB_ID}.out" \
     2>"$LOG_DIR/H047_report_${JOB_ID}.err"

module purge || true
source "$CONDA_SH"
conda activate "$CONDA_ENV"

for path in "$SCRIPT" "$TRAINING_DIR" "$FULLRIVER_DIR" "$REACH_DIR"; do
  [[ -e "$path" ]] || { echo "[ERROR] Missing: $path" >&2; exit 2; }
done

if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -not -path "$OUT_DIR/logs*" -print -quit | grep -q .; then
  if [[ "$OVERWRITE" == "1" ]]; then
    find "$OUT_DIR" -mindepth 1 -maxdepth 1 ! -name logs -exec rm -rf {} +
  else
    echo "[ERROR] Output exists: $OUT_DIR" >&2
    exit 3
  fi
fi
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" -u "$SCRIPT" \
  --training_dir "$TRAINING_DIR" \
  --fullriver_dir "$FULLRIVER_DIR" \
  --reach_dir "$REACH_DIR" \
  --output_dir "$OUT_DIR"

echo "=== DONE H047 ==="
echo "$OUT_DIR/H047_analysis_report.html"
