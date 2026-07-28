#!/usr/bin/env bash
#SBATCH -J G095_three_model_structfix
#SBATCH -p HydroIntel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v6_Stage2MeterMAE_FromNorm/logs/G095_three_model_structfix_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v6_Stage2MeterMAE_FromNorm/logs/G095_three_model_structfix_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
SCRIPT=${SCRIPT:-$WORK/script/G094_compare_three_models_global_and_local_StructureFix_20260727.py}

NORM_ONLY_PRED_ROOT=${NORM_ONLY_PRED_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Predictions_F010_TileAvgVRT_D001NoDataSafe}
NORM_ONLY_ERROR_ROOT=${NORM_ONLY_ERROR_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_GT_Error_F020_TileVRT_D001NoDataSafe}

METER_ONLY_PRED_ROOT=${METER_ONLY_PRED_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Predictions_F060_TileAvgVRT_D003MeterMAE_BaselineEval_D001NoDataSafe}
METER_ONLY_ERROR_ROOT=${METER_ONLY_ERROR_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_GT_Error_F062_UniquePixel_D003MeterMAE_BaselineEval_D001NoDataSafe}

NORM_THEN_METER_PRED_ROOT=${NORM_THEN_METER_PRED_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Predictions_G001_NormThenMeter_D001NoDataSafe}
NORM_THEN_METER_ERROR_ROOT=${NORM_THEN_METER_ERROR_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_GT_Error_G002_NormThenMeter_D001NoDataSafe}

OUT_DIR=${OUT_DIR:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Analysis_G094_ThreeModels_CommonFootprint_Local_D001NoDataSafe}

MIN_COMMON_PIXELS=${MIN_COMMON_PIXELS:-512}
MIN_SELECTED_CENTER_DISTANCE=${MIN_SELECTED_CENTER_DISTANCE:-400}
GLOBAL_DISPLAY_PERCENTILE=${GLOBAL_DISPLAY_PERCENTILE:-99.5}
OVERWRITE=${OVERWRITE:-0}
MAKE_ZIP=${MAKE_ZIP:-0}

CONDA_SH=""
for candidate in \
  /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh \
  /tank/data/SFS/xinyis/data/bathymetry/miniconda3/etc/profile.d/conda.sh
do
  if [[ -f "$candidate" ]]; then
    CONDA_SH="$candidate"
    break
  fi
done

if [[ -z "$CONDA_SH" ]]; then
  echo "[ERROR] Cannot find conda.sh" >&2
  exit 2
fi

source "$CONDA_SH"
conda activate "$ROOT/conda_envs/mae_zequn"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
mkdir -p "$WORK/cross_validation_v6_Stage2MeterMAE_FromNorm/logs"

for path in \
  "$SCRIPT" \
  "$NORM_ONLY_PRED_ROOT" "$NORM_ONLY_ERROR_ROOT" \
  "$METER_ONLY_PRED_ROOT" "$METER_ONLY_ERROR_ROOT" \
  "$NORM_THEN_METER_PRED_ROOT" "$NORM_THEN_METER_ERROR_ROOT"
do
  [[ -e "$path" ]] || {
    echo "[ERROR] Missing required path: $path" >&2
    exit 2
  }
done

ARGS=(
  --norm_only_pred_root "$NORM_ONLY_PRED_ROOT"
  --norm_only_error_root "$NORM_ONLY_ERROR_ROOT"
  --meter_only_pred_root "$METER_ONLY_PRED_ROOT"
  --meter_only_error_root "$METER_ONLY_ERROR_ROOT"
  --norm_then_meter_pred_root "$NORM_THEN_METER_PRED_ROOT"
  --norm_then_meter_error_root "$NORM_THEN_METER_ERROR_ROOT"
  --out_dir "$OUT_DIR"
  --min_common_pixels "$MIN_COMMON_PIXELS"
  --min_selected_center_distance "$MIN_SELECTED_CENTER_DISTANCE"
  --global_display_percentile "$GLOBAL_DISPLAY_PERCENTILE"
)

[[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" ]] && ARGS+=(--overwrite)
[[ "$MAKE_ZIP" != "1" && "$MAKE_ZIP" != "true" && "$MAKE_ZIP" != "TRUE" ]] && ARGS+=(--no_zip)

echo "================================================================================"
echo "G095 -> G094 three-model exact-common full-river metrics + native local regions"
date
echo "HOST=$(hostname)"
echo "PYTHON=$(command -v python)"
echo "SCRIPT=$SCRIPT"
echo "NORM_ONLY_PRED_ROOT=$NORM_ONLY_PRED_ROOT"
echo "NORM_ONLY_ERROR_ROOT=$NORM_ONLY_ERROR_ROOT"
echo "METER_ONLY_PRED_ROOT=$METER_ONLY_PRED_ROOT"
echo "METER_ONLY_ERROR_ROOT=$METER_ONLY_ERROR_ROOT"
echo "NORM_THEN_METER_PRED_ROOT=$NORM_THEN_METER_PRED_ROOT"
echo "NORM_THEN_METER_ERROR_ROOT=$NORM_THEN_METER_ERROR_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "MIN_COMMON_PIXELS=$MIN_COMMON_PIXELS"
echo "MIN_SELECTED_CENTER_DISTANCE=$MIN_SELECTED_CENTER_DISTANCE"
echo "GLOBAL_DISPLAY_PERCENTILE=$GLOBAL_DISPLAY_PERCENTILE"
echo "COMMON_MASK=CoreLoss AND valid GT AND valid predictions from all three models"
echo "GLOBAL_SCOPE=exact common unique geospatial pixels"
echo "LOCAL_OUTPUT=native PNG + GeoTIFF only"
echo "HTML_OUTPUT=disabled"
echo "FULL_RIVER_WEB_MAP=disabled"
echo "================================================================================"

python -u "$SCRIPT" "${ARGS[@]}"

echo "================================================================================"
echo "DONE G095"
echo "OUT_DIR=$OUT_DIR"
echo "GLOBAL_METRICS=$OUT_DIR/G094_global_common_metrics_wide.csv"
echo "GLOBAL_FIGURE=$OUT_DIR/global_figures/G094_abs_error_density_and_cdf_3x2_three_models.png"
echo "LOCAL_REGIONS=$OUT_DIR/local_regions"
echo "HTML_GENERATED=NO"
date
echo "================================================================================"
