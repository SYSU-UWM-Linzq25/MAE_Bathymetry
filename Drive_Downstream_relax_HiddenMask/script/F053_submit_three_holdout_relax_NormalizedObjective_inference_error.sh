#!/usr/bin/env bash
set -euo pipefail

# Submit the missing relaxed-mask normalized-objective full-river workflow:
#
#   F050 full-river inference
#      -> F052 GT/error and exact unique-geospatial-pixel metrics
#
# for CA, CO, and Santiam.
#
# This workflow is intentionally placed after the corrected Meter workflow
# F044-F048. It uses the same E001c AnyVisiblePatch tiles and the same
# prediction_patch_mask filtering, but loads checkpoints from results/NormOnly.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
SCRIPT_DIR=${SCRIPT_DIR:-$RELAX_ROOT/script}

INFER_SCRIPT=${INFER_SCRIPT:-$SCRIPT_DIR/F050_run_relax_fullriver_inference_NormalizedObjective_TileAvgVRT.sh}
METRIC_SCRIPT=${METRIC_SCRIPT:-$SCRIPT_DIR/F052_run_relax_fullriver_gt_error_NormalizedObjective_UniquePixel.sh}
H_ANALYSIS_SCRIPT=${H_ANALYSIS_SCRIPT:-$SCRIPT_DIR/H048_submit_analysis_only.sh}

GPU_ID=${GPU_ID:-0}
OVERWRITE_INFER=${OVERWRITE_INFER:-0}
RESUME_INFER=${RESUME_INFER:-1}
OVERWRITE_METRICS=${OVERWRITE_METRICS:-0}

# Optional: after all three F052 jobs succeed, submit the H044-H048
# analysis-only master. Default is off so F053 remains a pure F-series workflow.
SUBMIT_H_ANALYSIS=${SUBMIT_H_ANALYSIS:-0}
OVERWRITE_H_ANALYSIS=${OVERWRITE_H_ANALYSIS:-1}

CA_PARTITION=${CA_PARTITION:-HydroIntel}
CA_NODE=${CA_NODE:-execute-4006}
CO_PARTITION=${CO_PARTITION:-HydroIntel}
CO_NODE=${CO_NODE:-execute-4006}
SANTIAM_PARTITION=${SANTIAM_PARTITION:-HydroIntel}
SANTIAM_NODE=${SANTIAM_NODE:-execute-4006}

for file in "$INFER_SCRIPT" "$METRIC_SCRIPT"; do
  [[ -f "$file" ]] || {
    echo "[ERROR] Missing submission target: $file" >&2
    exit 2
  }
done

if [[ "$SUBMIT_H_ANALYSIS" == "1" ]]; then
  [[ -f "$H_ANALYSIS_SCRIPT" ]] || {
    echo "[ERROR] Missing H analysis submitter: $H_ANALYSIS_SCRIPT" >&2
    exit 2
  }
fi

sbatch_resource_args() {
  local partition="$1"
  local node="$2"
  local -n output_ref="$3"
  output_ref=(-p "$partition")
  if [[ -n "$node" ]]; then
    output_ref+=(-w "$node")
  fi
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

declare -A INFER_JID METRIC_JID

for preset in CA CO Santiam; do
  case "$preset" in
    CA)
      partition="$CA_PARTITION"
      node="$CA_NODE"
      ;;
    CO)
      partition="$CO_PARTITION"
      node="$CO_NODE"
      ;;
    Santiam)
      partition="$SANTIAM_PARTITION"
      node="$SANTIAM_NODE"
      ;;
  esac

  INFER_JID[$preset]=$(submit_infer "$preset" "$partition" "$node")
  METRIC_JID[$preset]=$(submit_metrics_after "$preset" "${INFER_JID[$preset]}")
done

H_ANALYSIS_JID="NOT_SUBMITTED"
if [[ "$SUBMIT_H_ANALYSIS" == "1" ]]; then
  metric_dependency="${METRIC_JID[CA]}:${METRIC_JID[CO]}:${METRIC_JID[Santiam]}"
  H_ANALYSIS_JID=$(sbatch --parsable \
    --dependency="afterok:${metric_dependency}" \
    --export=ALL,OVERWRITE="$OVERWRITE_H_ANALYSIS",RELAX_NORMALIZED_PRED_ROOT="$RELAX_ROOT/results/FullRiver_Predictions_F049_NormalizedObjective_D001cAnyVisiblePatch" \
    "$H_ANALYSIS_SCRIPT")
fi

echo "============================================================"
echo "Relaxed normalized-objective full-river workflow submitted"
for preset in CA CO Santiam; do
  echo "$preset"
  echo "  F050 full-river inference : ${INFER_JID[$preset]}"
  echo "  F052 GT/error metrics     : ${METRIC_JID[$preset]} (afterok:${INFER_JID[$preset]})"
done
echo
echo "Prediction root:"
echo "$RELAX_ROOT/results/FullRiver_Predictions_F049_NormalizedObjective_D001cAnyVisiblePatch"
echo
echo "GT/error root:"
echo "$RELAX_ROOT/results/FullRiver_GT_Error_F051_NormalizedObjective_D001cAnyVisiblePatch"
echo
echo "H044-H048 analysis submitter:"
echo "  $H_ANALYSIS_JID"
echo "============================================================"
