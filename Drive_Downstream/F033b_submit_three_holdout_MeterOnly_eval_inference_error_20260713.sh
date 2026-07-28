#!/usr/bin/env bash
# NUMBER-ALIGNED NAME: F033b_submit_three_holdout_MeterOnly_eval_inference_error_20260713.sh
# ORIGINAL BACKUP NAME: F064_submit_three_holdout_meterMAE_eval_inference_metrics_20260713.sh
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
set -euo pipefail

SCRIPT_DIR=${SCRIPT_DIR:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/script}
EVAL_SCRIPT=${EVAL_SCRIPT:-$SCRIPT_DIR/E031_run_MeterOnly_train_val_core_pixel_eval_20260713.sh}
INFER_SCRIPT=${INFER_SCRIPT:-$SCRIPT_DIR/F031_run_fullriver_inference_MeterOnly_TileAvgVRT_20260713.sh}
METRIC_SCRIPT=${METRIC_SCRIPT:-$SCRIPT_DIR/F033_run_fullriver_gt_error_MeterOnly_UniquePixel_20260713.sh}

GPU_ID=${GPU_ID:-0}
OVERWRITE_EVAL=${OVERWRITE_EVAL:-0}
OVERWRITE_INFER=${OVERWRITE_INFER:-0}
RESUME_INFER=${RESUME_INFER:-1}
OVERWRITE_METRICS=${OVERWRITE_METRICS:-0}
NO_VISUALS=${NO_VISUALS:-0}

for f in "$EVAL_SCRIPT" "$INFER_SCRIPT" "$METRIC_SCRIPT"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing submission target: $f" >&2; exit 2; }
done

submit_eval() {
  local preset="$1"
  sbatch --parsable \
    --export=ALL,HOLDOUT_PRESET="$preset",GPU_ID="$GPU_ID",OVERWRITE_EVAL="$OVERWRITE_EVAL",NO_VISUALS="$NO_VISUALS" \
    "$EVAL_SCRIPT"
}
submit_infer() {
  local preset="$1"
  sbatch --parsable \
    --export=ALL,HOLDOUT_PRESET="$preset",GPU_ID="$GPU_ID",OVERWRITE="$OVERWRITE_INFER",RESUME="$RESUME_INFER" \
    "$INFER_SCRIPT"
}
submit_metrics_after() {
  local preset="$1"
  local dependency_job="$2"
  sbatch --parsable \
    --dependency="afterok:${dependency_job}" \
    --export=ALL,HOLDOUT_PRESET="$preset",OVERWRITE="$OVERWRITE_METRICS" \
    "$METRIC_SCRIPT"
}

declare -A EVAL_JID INFER_JID METRIC_JID

for preset in CA CO Santiam; do
  EVAL_JID[$preset]=$(submit_eval "$preset")
  INFER_JID[$preset]=$(submit_infer "$preset")
  METRIC_JID[$preset]=$(submit_metrics_after "$preset" "${INFER_JID[$preset]}")
done

echo "============================================================"
echo "v4 meter-MAE evaluation and full-river workflow submitted"
for preset in CA CO Santiam; do
  echo "$preset"
  echo "  E060 train/val evaluation : ${EVAL_JID[$preset]}"
  echo "  F060 full-river inference : ${INFER_JID[$preset]}"
  echo "  F062 GT/error metrics     : ${METRIC_JID[$preset]} (afterok:${INFER_JID[$preset]})"
done
echo
echo "Prediction root:"
echo "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Predictions_F060_TileAvgVRT_D003MeterMAE_BaselineEval_D001NoDataSafe"
echo
echo "GT/error root:"
echo "/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_GT_Error_F062_UniquePixel_D003MeterMAE_BaselineEval_D001NoDataSafe"
echo "============================================================"
