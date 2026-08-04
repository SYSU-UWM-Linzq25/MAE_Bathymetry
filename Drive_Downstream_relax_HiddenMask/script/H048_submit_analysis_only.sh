#!/usr/bin/env bash
set -euo pipefail

# Analysis-only master submitter.
# No F-series inference or GT/error generation is submitted here.
#
# H044: training/validation objective comparison
# H045: full-river common-loss-pixel analysis
# H046: local continuous-reach six-panel figures
# H047: combined HTML/Markdown report

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
SCRIPT_DIR=${SCRIPT_DIR:-$RELAX_ROOT/script}
OVERWRITE=${OVERWRITE:-0}

RELAX_RESULTS=${RELAX_RESULTS:-$RELAX_ROOT/results}
RESOLVER=${RESOLVER:-$SCRIPT_DIR/H044_resolve_relaxed_normalized_prediction_root.py}
RELAX_NORMALIZED_PRED_ROOT=${RELAX_NORMALIZED_PRED_ROOT:-$RELAX_RESULTS/FullRiver_Predictions_F049_NormalizedObjective_D001cAnyVisiblePatch}
INPUT_AUDIT_JSON=${INPUT_AUDIT_JSON:-$RELAX_RESULTS/H048_input_resolution.json}

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

H044=${H044:-$SCRIPT_DIR/H044_run_training_objective_comparison.sh}
H045=${H045:-$SCRIPT_DIR/H045_run_fullriver_common_loss_analysis.sh}
H046=${H046:-$SCRIPT_DIR/H046_run_local_reach_6panel_analysis.sh}
H047=${H047:-$SCRIPT_DIR/H047_run_analysis_report.sh}

for script in "$RESOLVER" "$H044" "$H045" "$H046" "$H047"; do
  [[ -f "$script" ]] || { echo "[ERROR] Missing analysis script: $script" >&2; exit 2; }
done

if [[ -z "$RELAX_NORMALIZED_PRED_ROOT" ]]; then
  RELAX_NORMALIZED_PRED_ROOT=$(
    "$PYTHON_BIN" "$RESOLVER" \
      --relax-root "$RELAX_ROOT" \
      --output-json "$INPUT_AUDIT_JSON" \
      --print-root
  )
else
  RELAX_NORMALIZED_PRED_ROOT=$(
    "$PYTHON_BIN" "$RESOLVER" \
      --relax-root "$RELAX_ROOT" \
      --explicit-root "$RELAX_NORMALIZED_PRED_ROOT" \
      --output-json "$INPUT_AUDIT_JSON" \
      --print-root
  )
fi
export RELAX_NORMALIZED_PRED_ROOT

echo "Resolved relaxed-normalized prediction root:"
echo "  $RELAX_NORMALIZED_PRED_ROOT"
echo "Python environment:"
echo "  $PYTHON_BIN"
echo "Input audit:"
echo "  $INPUT_AUDIT_JSON"

j044=$(sbatch --parsable --export=ALL,OVERWRITE="$OVERWRITE" "$H044")
j045=$(sbatch --parsable --export=ALL,OVERWRITE="$OVERWRITE",RELAX_NORMALIZED_PRED_ROOT="$RELAX_NORMALIZED_PRED_ROOT" "$H045")
j046=$(sbatch --parsable --export=ALL,OVERWRITE="$OVERWRITE",RELAX_NORMALIZED_PRED_ROOT="$RELAX_NORMALIZED_PRED_ROOT" "$H046")
j047=$(sbatch --parsable \
  --dependency="afterok:${j044}:${j045}:${j046}" \
  --export=ALL,OVERWRITE="$OVERWRITE" \
  "$H047")

echo "============================================================"
echo "Analysis-only H-series submitted"
echo "H044 training/validation : $j044"
echo "H045 full-river analysis : $j045"
echo "H046 local 6-panel       : $j046"
echo "H047 combined report     : $j047 (after H044/H045/H046)"
echo
echo "No F-series job was submitted."
echo "============================================================"
