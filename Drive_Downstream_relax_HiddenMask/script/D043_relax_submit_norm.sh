#!/usr/bin/env bash
set -euo pipefail

# RELAX PROJECT: isolated code/results under Downstream_Task_Bathy_relax_HiddenMask.
SCRIPT_DIR=${SCRIPT_DIR:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask/script}
RUN_SCRIPT=${RUN_SCRIPT:-$SCRIPT_DIR/D041_relax_holdout_norm.sh}
EPOCHS=${EPOCHS:-400}; PATIENCE=${PATIENCE:-60}; GPU_ID=${GPU_ID:-0}; FRESH_RUN=${FRESH_RUN:-1}
[[ -f "$RUN_SCRIPT" ]] || { echo "[ERROR] Missing $RUN_SCRIPT" >&2; exit 2; }
submit(){ sbatch --parsable --export=ALL,HOLDOUT_PRESET="$1",EPOCHS="$EPOCHS",PATIENCE="$PATIENCE",GPU_ID="$GPU_ID",FRESH_RUN="$FRESH_RUN" "$RUN_SCRIPT"; }
a=$(submit CA); c=$(submit CO); s=$(submit Santiam)
echo "D001c AnyVisiblePatch RELAX project NormOnly submitted: CA=$a CO=$c Santiam=$s"
echo "/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask/results/NormOnly"
