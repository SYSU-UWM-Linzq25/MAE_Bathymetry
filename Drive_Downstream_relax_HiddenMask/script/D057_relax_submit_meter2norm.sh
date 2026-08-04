#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
RUN_SCRIPT=${RUN_SCRIPT:-$RELAX_ROOT/script/D055_relax_holdout_meter2norm.sh}

EPOCHS=${EPOCHS:-120}
PATIENCE=${PATIENCE:-30}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-7}
GPU_ID=${GPU_ID:-0}
FRESH_RUN=${FRESH_RUN:-1}
OVERWRITE_STAGE2=${OVERWRITE_STAGE2:-0}

[[ -f "$RUN_SCRIPT" ]] || { echo "[ERROR] Missing $RUN_SCRIPT" >&2; exit 2; }

submit_one() {
  sbatch --parsable \
    --export=ALL,HOLDOUT_PRESET="$1",EPOCHS="$EPOCHS",PATIENCE="$PATIENCE",LR="$LR",MIN_LR="$MIN_LR",GPU_ID="$GPU_ID",FRESH_RUN="$FRESH_RUN",OVERWRITE_STAGE2="$OVERWRITE_STAGE2" \
    "$RUN_SCRIPT"
}

jid_ca=$(submit_one CA)
jid_co=$(submit_one CO)
jid_santiam=$(submit_one Santiam)

echo "D001c RELAX MeterThenNorm submitted:"
echo "  CA      : $jid_ca"
echo "  CO      : $jid_co"
echo "  Santiam : $jid_santiam"
echo "Stage 2 objective=normalized_mse; best/ES=val_mae_m_mask"
