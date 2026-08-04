#!/usr/bin/env bash
set -euo pipefail

# Submit CA, CO, and Santiam MeterOnly defreeze-last-1 experiments.
#
# Resource placement is configurable per river. Defaults keep all jobs on
# HydroIntel/execute-4006. Example moving Santiam to the public GPU:
#
#   SANTIAM_PARTITION=gpu SANTIAM_NODE=execute-3000 \
#     bash F073_relax_meterOnly_defreeze_submit.sh

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
SCRIPT_DIR=${SCRIPT_DIR:-$RELAX_ROOT/script}
RUN_SCRIPT=${RUN_SCRIPT:-$SCRIPT_DIR/F071_relax_meterOnly_defreeze_holdout.sh}

EPOCHS=${EPOCHS:-80}
PATIENCE=${PATIENCE:-20}
LR=${LR:-1e-6}
MIN_LR=${MIN_LR:-1e-8}
GPU_ID=${GPU_ID:-0}
FRESH_RUN=${FRESH_RUN:-1}
OVERWRITE_DEFREEZE=${OVERWRITE_DEFREEZE:-0}

CA_PARTITION=${CA_PARTITION:-HydroIntel}
CA_NODE=${CA_NODE:-execute-4006}
CO_PARTITION=${CO_PARTITION:-HydroIntel}
CO_NODE=${CO_NODE:-execute-4006}
SANTIAM_PARTITION=${SANTIAM_PARTITION:-HydroIntel}
SANTIAM_NODE=${SANTIAM_NODE:-execute-4006}

[[ -f "$RUN_SCRIPT" ]] || { echo "[ERROR] Missing run script: $RUN_SCRIPT" >&2; exit 2; }

submit_one() {
  local preset="$1"
  local partition="$2"
  local node="$3"
  local sbatch_args=(--parsable -p "$partition")

  if [[ -n "$node" ]]; then
    sbatch_args+=(-w "$node")
  fi

  sbatch "${sbatch_args[@]}" \
    --export=ALL,HOLDOUT_PRESET="$preset",EPOCHS="$EPOCHS",PATIENCE="$PATIENCE",LR="$LR",MIN_LR="$MIN_LR",GPU_ID="$GPU_ID",FRESH_RUN="$FRESH_RUN",OVERWRITE_DEFREEZE="$OVERWRITE_DEFREEZE" \
    "$RUN_SCRIPT"
}

jid_ca=$(submit_one CA "$CA_PARTITION" "$CA_NODE")
jid_co=$(submit_one CO "$CO_PARTITION" "$CO_NODE")
jid_santiam=$(submit_one Santiam "$SANTIAM_PARTITION" "$SANTIAM_NODE")

echo "============================================================"
echo "MeterOnly defreeze-last-1 jobs submitted"
echo "CA      : $jid_ca  [$CA_PARTITION ${CA_NODE:-any-node}]"
echo "CO      : $jid_co  [$CO_PARTITION ${CO_NODE:-any-node}]"
echo "Santiam : $jid_santiam  [$SANTIAM_PARTITION ${SANTIAM_NODE:-any-node}]"
echo
echo "Defaults:"
echo "  source       = matching D001c MeterOnly checkpoint-best"
echo "  trainable    = decoder + last 1 encoder block"
echo "  objective    = meter_mae"
echo "  best/ES      = val_mae_m_mask"
echo "  epoch -1     = untouched frozen MeterOnly"
echo "  epochs       = $EPOCHS"
echo "  lr           = $LR"
echo "  min_lr       = $MIN_LR"
echo "  patience     = $PATIENCE"
echo
echo "Results:"
echo "  $RELAX_ROOT/results/MeterOnly_DefreezeLast1"
echo "============================================================"
