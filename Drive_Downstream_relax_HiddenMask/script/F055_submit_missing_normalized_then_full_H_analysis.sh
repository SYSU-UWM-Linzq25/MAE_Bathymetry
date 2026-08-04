#!/usr/bin/env bash
set -euo pipefail

# One-command completion of the missing branch and the full analysis:
#
#   F050/F052 for CA, CO, Santiam
#     -> H048 analysis-only master
#
# Internally this calls F053 with SUBMIT_H_ANALYSIS=1.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
SCRIPT_DIR=${SCRIPT_DIR:-$RELAX_ROOT/script}
F053=${F053:-$SCRIPT_DIR/F053_submit_three_holdout_relax_NormalizedObjective_inference_error.sh}

OVERWRITE_INFER=${OVERWRITE_INFER:-0}
RESUME_INFER=${RESUME_INFER:-1}
OVERWRITE_METRICS=${OVERWRITE_METRICS:-0}
OVERWRITE_H_ANALYSIS=${OVERWRITE_H_ANALYSIS:-1}

[[ -f "$F053" ]] || {
  echo "[ERROR] Missing F053 submitter: $F053" >&2
  exit 2
}

SUBMIT_H_ANALYSIS=1 \
OVERWRITE_INFER="$OVERWRITE_INFER" \
RESUME_INFER="$RESUME_INFER" \
OVERWRITE_METRICS="$OVERWRITE_METRICS" \
OVERWRITE_H_ANALYSIS="$OVERWRITE_H_ANALYSIS" \
bash "$F053"
