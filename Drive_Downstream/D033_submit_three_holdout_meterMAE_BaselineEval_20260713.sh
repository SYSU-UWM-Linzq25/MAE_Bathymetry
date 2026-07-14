#!/usr/bin/env bash
set -euo pipefail

# D033: submit the three formal holdout meter-MAE + baseline-evaluation jobs.
# Run this script from the Downstream_Task_Bathy/script directory or set SCRIPT_DIR.

SCRIPT_DIR=${SCRIPT_DIR:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/script}
RUN_SCRIPT=${RUN_SCRIPT:-$SCRIPT_DIR/D031_v2_dualmask_holdout_meterMAE_BaselineEval_RuntimeLogSafe_D001NoDataSafe_20260713.sh}

EPOCHS=${EPOCHS:-400}
PATIENCE=${PATIENCE:-60}
GPU_ID=${GPU_ID:-0}
FRESH_RUN=${FRESH_RUN:-1}
BASELINE_EVAL_BEFORE_TRAINING=${BASELINE_EVAL_BEFORE_TRAINING:-1}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "[ERROR] Missing run script: $RUN_SCRIPT" >&2
  exit 2
fi

submit_one() {
  local preset="$1"
  sbatch --parsable \
    --export=ALL,HOLDOUT_PRESET="$preset",EPOCHS="$EPOCHS",PATIENCE="$PATIENCE",GPU_ID="$GPU_ID",FRESH_RUN="$FRESH_RUN",BASELINE_EVAL_BEFORE_TRAINING="$BASELINE_EVAL_BEFORE_TRAINING",WARMUP_EPOCHS="$WARMUP_EPOCHS" \
    "$RUN_SCRIPT"
}

jid_ca=$(submit_one CA)
jid_co=$(submit_one CO)
jid_santiam=$(submit_one Santiam)

echo "Formal meter-MAE + baseline jobs submitted:"
echo "  CA      : $jid_ca"
echo "  CO      : $jid_co"
echo "  Santiam : $jid_santiam"
echo
echo "Logs:"
echo "  /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v4_meterMAE_BaselineEval/logs/"
