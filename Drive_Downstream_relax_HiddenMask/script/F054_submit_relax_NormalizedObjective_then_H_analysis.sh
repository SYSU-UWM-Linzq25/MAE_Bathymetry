#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
SCRIPT_DIR=${SCRIPT_DIR:-$RELAX_ROOT/script}
F053=${F053:-$SCRIPT_DIR/F053_submit_three_holdout_relax_NormalizedObjective_inference_error.sh}

[[ -f "$F053" ]] || {
  echo "[ERROR] Missing F053 submitter: $F053" >&2
  exit 2
}

SUBMIT_H_ANALYSIS=1 \
bash "$F053"
