#!/usr/bin/env bash
set -euo pipefail

# Submit:
#   E045 train/val evaluation
#   F045 E001c full-river inference
#   F047 GT/error unique-pixel metrics after F045
#
# Defaults use HydroIntel/execute-4006. Resources can be overridden per river:
#   SANTIAM_PARTITION=gpu SANTIAM_NODE=execute-3000 bash F048_...

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
SCRIPT_DIR=${SCRIPT_DIR:-$RELAX_ROOT/script}

EVAL_SCRIPT=${EVAL_SCRIPT:-$SCRIPT_DIR/E045_run_relax_MeterOnly_train_val_core_pixel_eval.sh}
INFER_SCRIPT=${INFER_SCRIPT:-$SCRIPT_DIR/F045_run_relax_fullriver_inference_MeterOnly_TileAvgVRT.sh}
METRIC_SCRIPT=${METRIC_SCRIPT:-$SCRIPT_DIR/F047_run_relax_fullriver_gt_error_MeterOnly_UniquePixel.sh}

GPU_ID=${GPU_ID:-0}
SUBMIT_EVAL=${SUBMIT_EVAL:-1}
OVERWRITE_EVAL=${OVERWRITE_EVAL:-0}
OVERWRITE_INFER=${OVERWRITE_INFER:-0}
RESUME_INFER=${RESUME_INFER:-1}
OVERWRITE_METRICS=${OVERWRITE_METRICS:-0}
NO_VISUALS=${NO_VISUALS:-0}

CA_PARTITION=${CA_PARTITION:-HydroIntel}
CA_NODE=${CA_NODE:-execute-4006}
CO_PARTITION=${CO_PARTITION:-HydroIntel}
CO_NODE=${CO_NODE:-execute-4006}
SANTIAM_PARTITION=${SANTIAM_PARTITION:-HydroIntel}
SANTIAM_NODE=${SANTIAM_NODE:-execute-4006}

for f in "$INFER_SCRIPT" "$METRIC_SCRIPT"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing submission target: $f" >&2; exit 2; }
done
if [[ "$SUBMIT_EVAL" == "1" ]]; then
  [[ -f "$EVAL_SCRIPT" ]] || {
    echo "[ERROR] Missing evaluation target: $EVAL_SCRIPT" >&2
    exit 2
  }
fi

sbatch_resource_args() {
  local partition="$1"
  local node="$2"
  local -n out_ref="$3"
  out_ref=(-p "$partition")
  if [[ -n "$node" ]]; then
    out_ref+=(-w "$node")
  fi
}

submit_eval() {
  local preset="$1" partition="$2" node="$3"
  local args=()
  sbatch_resource_args "$partition" "$node" args
  sbatch --parsable "${args[@]}" \
    --export=ALL,HOLDOUT_PRESET="$preset",GPU_ID="$GPU_ID",OVERWRITE_EVAL="$OVERWRITE_EVAL",NO_VISUALS="$NO_VISUALS" \
    "$EVAL_SCRIPT"
}

submit_infer() {
  local preset="$1" partition="$2" node="$3"
  local args=()
  sbatch_resource_args "$partition" "$node" args
  sbatch --parsable "${args[@]}" \
    --export=ALL,HOLDOUT_PRESET="$preset",GPU_ID="$GPU_ID",OVERWRITE="$OVERWRITE_INFER",RESUME="$RESUME_INFER" \
    "$INFER_SCRIPT"
}

submit_metrics_after() {
  local preset="$1" dependency_job="$2"
  sbatch --parsable \
    --dependency="afterok:${dependency_job}" \
    --export=ALL,HOLDOUT_PRESET="$preset",OVERWRITE="$OVERWRITE_METRICS" \
    "$METRIC_SCRIPT"
}

declare -A EVAL_JID INFER_JID METRIC_JID

for preset in CA CO Santiam; do
  case "$preset" in
    CA)      partition="$CA_PARTITION";      node="$CA_NODE" ;;
    CO)      partition="$CO_PARTITION";      node="$CO_NODE" ;;
    Santiam) partition="$SANTIAM_PARTITION"; node="$SANTIAM_NODE" ;;
  esac

  if [[ "$SUBMIT_EVAL" == "1" ]]; then
    EVAL_JID[$preset]=$(submit_eval "$preset" "$partition" "$node")
  else
    EVAL_JID[$preset]="SKIPPED"
  fi

  INFER_JID[$preset]=$(submit_infer "$preset" "$partition" "$node")
  METRIC_JID[$preset]=$(submit_metrics_after "$preset" "${INFER_JID[$preset]}")
done

echo "============================================================"
echo "Corrected prediction-patch-filtered MeterOnly workflow submitted"
for preset in CA CO Santiam; do
  echo "$preset"
  echo "  E045 train/val evaluation : ${EVAL_JID[$preset]}"
  echo "  F045 full-river inference : ${INFER_JID[$preset]}"
  echo "  F047 GT/error metrics     : ${METRIC_JID[$preset]} (afterok:${INFER_JID[$preset]})"
done
echo
echo "Prediction root:"
echo "$RELAX_ROOT/results/FullRiver_Predictions_F044_MeterOnly_D001cAnyVisiblePatch"
echo
echo "GT/error root:"
echo "$RELAX_ROOT/results/FullRiver_GT_Error_F046_MeterOnly_D001cAnyVisiblePatch"
echo "============================================================"
