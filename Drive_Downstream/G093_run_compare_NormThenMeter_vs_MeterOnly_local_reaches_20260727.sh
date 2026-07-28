#!/usr/bin/env bash
#SBATCH -J G002_two_model_local
#SBATCH -p HydroIntel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v6_Stage2MeterMAE_FromNorm/logs/G002_two_model_local_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v6_Stage2MeterMAE_FromNorm/logs/G002_two_model_local_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
SCRIPT=${SCRIPT:-$WORK/script/G092_compare_NormThenMeter_vs_MeterOnly_local_reaches_20260727.py}

NORM_METER_PRED_ROOT=${NORM_METER_PRED_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Predictions_G001_NormThenMeter_D001NoDataSafe}
NORM_METER_ERROR_ROOT=${NORM_METER_ERROR_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_GT_Error_G002_NormThenMeter_D001NoDataSafe}
METER_ONLY_PRED_ROOT=${METER_ONLY_PRED_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Predictions_F060_TileAvgVRT_D003MeterMAE_BaselineEval_D001NoDataSafe}
METER_ONLY_ERROR_ROOT=${METER_ONLY_ERROR_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_GT_Error_F062_UniquePixel_D003MeterMAE_BaselineEval_D001NoDataSafe}
OUT_DIR=${OUT_DIR:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_Analysis_G004_NormThenMeter_vs_MeterOnly_D001NoDataSafe}

MIN_COMMON_PIXELS=${MIN_COMMON_PIXELS:-512}
MIN_SELECTED_CENTER_DISTANCE=${MIN_SELECTED_CENTER_DISTANCE:-400}
ROBUST_ERROR_PERCENTILE=${ROBUST_ERROR_PERCENTILE:-98}
OVERWRITE=${OVERWRITE:-0}
MAKE_ZIP=${MAKE_ZIP:-1}

source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
mkdir -p "$WORK/cross_validation_v6_Stage2MeterMAE_FromNorm/logs"

for path in \
  "$SCRIPT" \
  "$NORM_METER_PRED_ROOT" "$NORM_METER_ERROR_ROOT" \
  "$METER_ONLY_PRED_ROOT" "$METER_ONLY_ERROR_ROOT"; do
  [[ -e "$path" ]] || { echo "[ERROR] Missing required path: $path" >&2; exit 2; }
done

ARGS=(
  --norm_meter_pred_root "$NORM_METER_PRED_ROOT"
  --norm_meter_error_root "$NORM_METER_ERROR_ROOT"
  --meter_only_pred_root "$METER_ONLY_PRED_ROOT"
  --meter_only_error_root "$METER_ONLY_ERROR_ROOT"
  --out_dir "$OUT_DIR"
  --min_common_pixels "$MIN_COMMON_PIXELS"
  --min_selected_center_distance "$MIN_SELECTED_CENTER_DISTANCE"
  --robust_error_percentile "$ROBUST_ERROR_PERCENTILE"
)

[[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" ]] && ARGS+=(--overwrite)
[[ "$MAKE_ZIP" != "1" && "$MAKE_ZIP" != "true" && "$MAKE_ZIP" != "TRUE" ]] && ARGS+=(--no_zip)

echo "============================================================"
echo "G002 run G001 two-model native local-reach analysis"
date
echo "HOST=$(hostname)"
echo "SCRIPT=$SCRIPT"
echo "NORM_METER_PRED_ROOT=$NORM_METER_PRED_ROOT"
echo "NORM_METER_ERROR_ROOT=$NORM_METER_ERROR_ROOT"
echo "METER_ONLY_PRED_ROOT=$METER_ONLY_PRED_ROOT"
echo "METER_ONLY_ERROR_ROOT=$METER_ONLY_ERROR_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "MIN_COMMON_PIXELS=$MIN_COMMON_PIXELS"
echo "MIN_SELECTED_CENTER_DISTANCE=$MIN_SELECTED_CENTER_DISTANCE"
echo "ROBUST_ERROR_PERCENTILE=$ROBUST_ERROR_PERCENTILE"
echo "PRIMARY_LOCAL_ERROR_DISPLAY=actual local max absolute error, no clipping"
echo "LOCAL_GRID=native source grid, no EPSG:3857 reprojection, no bilinear resampling"
echo "============================================================"

python -u "$SCRIPT" "${ARGS[@]}"

echo "=== DONE G002 ==="
echo "HTML=$OUT_DIR/G004_local_reach_dashboard.html"
echo "CSV=$OUT_DIR/G004_global_common_metrics.csv"
date
