#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=${SCRIPT_DIR:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/script}
RUN_SCRIPT=${RUN_SCRIPT:-$SCRIPT_DIR/D035_v2_dualmask_holdout_Stage2NormMSE_MeterSelect_RuntimeLogSafe_20260713.sh}

EPOCHS=${EPOCHS:-120}
PATIENCE=${PATIENCE:-30}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-7}
GPU_ID=${GPU_ID:-0}
FRESH_RUN=${FRESH_RUN:-1}
OVERWRITE_STAGE2=${OVERWRITE_STAGE2:-0}

[[ -f "$RUN_SCRIPT" ]] || { echo "[ERROR] Missing run script: $RUN_SCRIPT" >&2; exit 2; }

submit_one() {
  local preset="$1"
  sbatch --parsable \
    --export=ALL,HOLDOUT_PRESET="$preset",EPOCHS="$EPOCHS",PATIENCE="$PATIENCE",LR="$LR",MIN_LR="$MIN_LR",GPU_ID="$GPU_ID",FRESH_RUN="$FRESH_RUN",OVERWRITE_STAGE2="$OVERWRITE_STAGE2" \
    "$RUN_SCRIPT"
}

jid_ca=$(submit_one CA)
jid_co=$(submit_one CO)
jid_santiam=$(submit_one Santiam)

echo "============================================================"
echo "Stage-2 normalized-MSE / meter-selected jobs submitted"
echo "CA      : $jid_ca"
echo "CO      : $jid_co"
echo "Santiam : $jid_santiam"
echo
echo "Defaults:"
echo "  optimization loss = exact normalized MSE"
echo "  checkpoint metric = validation meter MAE"
echo "  initialization    = Stage-1 meter-MAE checkpoint-best"
echo "  baseline safety   = Stage-1 model is epoch -1 checkpoint-best"
echo "  lr                 = $LR"
echo "  epochs             = $EPOCHS"
echo "  patience           = $PATIENCE"
echo
echo "Logs:"
echo "  /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v5_Stage2NormMSE_MeterSelect/logs/"
echo "============================================================"
