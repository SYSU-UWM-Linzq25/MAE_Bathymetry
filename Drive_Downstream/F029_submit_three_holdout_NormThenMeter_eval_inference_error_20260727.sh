#!/usr/bin/env bash
# NUMBER-ALIGNED NAME: F029_submit_three_holdout_NormThenMeter_eval_inference_error_20260727.sh
# Purpose: submit E026 evaluation, F026 full-river inference, and F028 error metrics for CA/CO/Santiam.
set -euo pipefail
SCRIPT_DIR=${SCRIPT_DIR:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/script}
EVAL_SCRIPT=${EVAL_SCRIPT:-$SCRIPT_DIR/E026_run_NormThenMeter_train_val_core_pixel_eval_20260727.sh}
INFER_SCRIPT=${INFER_SCRIPT:-$SCRIPT_DIR/F026_run_fullriver_inference_NormThenMeter_TileAvgVRT_20260727.sh}
ERROR_SCRIPT=${ERROR_SCRIPT:-$SCRIPT_DIR/F028_run_fullriver_gt_error_NormThenMeter_UniquePixel_20260727.sh}
ANALYSIS_SCRIPT=${ANALYSIS_SCRIPT:-$SCRIPT_DIR/G093_run_compare_NormThenMeter_vs_MeterOnly_local_reaches_20260727.sh}
GPU_ID=${GPU_ID:-0}
OVERWRITE_EVAL=${OVERWRITE_EVAL:-0}
OVERWRITE_INFER=${OVERWRITE_INFER:-0}
RESUME_INFER=${RESUME_INFER:-1}
OVERWRITE_ERROR=${OVERWRITE_ERROR:-0}
SUBMIT_ANALYSIS=${SUBMIT_ANALYSIS:-0}
for f in "$EVAL_SCRIPT" "$INFER_SCRIPT" "$ERROR_SCRIPT"; do [[ -f "$f" ]] || { echo "[ERROR] Missing $f" >&2; exit 2; }; done
submit_eval(){ sbatch --parsable --export=ALL,HOLDOUT_PRESET="$1",GPU_ID="$GPU_ID",OVERWRITE_EVAL="$OVERWRITE_EVAL" "$EVAL_SCRIPT"; }
submit_infer(){ sbatch --parsable --export=ALL,HOLDOUT_PRESET="$1",GPU_ID="$GPU_ID",OVERWRITE="$OVERWRITE_INFER",RESUME="$RESUME_INFER" "$INFER_SCRIPT"; }
submit_error(){ sbatch --parsable --dependency="afterok:$2" --export=ALL,HOLDOUT_PRESET="$1",OVERWRITE="$OVERWRITE_ERROR" "$ERROR_SCRIPT"; }
declare -A EVAL_JID INFER_JID ERROR_JID
for preset in CA CO Santiam; do
  EVAL_JID[$preset]=$(submit_eval "$preset")
  INFER_JID[$preset]=$(submit_infer "$preset")
  ERROR_JID[$preset]=$(submit_error "$preset" "${INFER_JID[$preset]}")
done
ANALYSIS_JID=""
if [[ "$SUBMIT_ANALYSIS" == "1" ]]; then
  [[ -f "$ANALYSIS_SCRIPT" ]] || { echo "[ERROR] Missing $ANALYSIS_SCRIPT" >&2; exit 2; }
  dep="afterok:${ERROR_JID[CA]}:${ERROR_JID[CO]}:${ERROR_JID[Santiam]}"
  ANALYSIS_JID=$(sbatch --parsable --dependency="$dep" "$ANALYSIS_SCRIPT")
fi
echo "============================================================"
echo "NormThenMeter workflow submitted"
for preset in CA CO Santiam; do
  echo "$preset E026=${EVAL_JID[$preset]} F026=${INFER_JID[$preset]} F028=${ERROR_JID[$preset]}"
done
[[ -n "$ANALYSIS_JID" ]] && echo "G093=$ANALYSIS_JID"
echo "============================================================"
