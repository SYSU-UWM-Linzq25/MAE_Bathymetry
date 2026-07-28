#!/usr/bin/env bash
# NUMBER-ALIGNED NAME: G091_run_compare_fullriver_error_distribution_NormOnly_vs_MeterOnly_20260714.sh
# ORIGINAL BACKUP NAME: F066_run_fullriver_compare_error_distribution_norm_vs_meter_20260714.sh
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
#SBATCH -J F066_norm_meter_dist
#SBATCH -p HydroIntel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v4_meterMAE_BaselineEval/logs/F066_norm_meter_dist_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v4_meterMAE_BaselineEval/logs/F066_norm_meter_dist_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
SCRIPT=${SCRIPT:-$WORK/script/G090_compare_fullriver_error_distribution_NormOnly_vs_MeterOnly_20260714.py}

NORM_ROOT=${NORM_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_GT_Error_F020_TileVRT_D001NoDataSafe}
METER_ROOT=${METER_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_GT_Error_F062_UniquePixel_D003MeterMAE_BaselineEval_D001NoDataSafe}
OUT_DIR=${OUT_DIR:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/FullRiver_ErrorDistribution_F065_NormVsMeter_D001NoDataSafe}

NODATA=${NODATA:--999999.0}
NODATA_THRESHOLD=${NODATA_THRESHOLD:--9999.0}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}
DISPLAY_PERCENTILE=${DISPLAY_PERCENTILE:-99.5}
OVERWRITE_CACHE=${OVERWRITE_CACHE:-0}

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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "$WORK/cross_validation_v4_meterMAE_BaselineEval/logs"

for path in "$SCRIPT" "$NORM_ROOT" "$METER_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 2
  fi
done

ARGS=(
  --norm_root "$NORM_ROOT"
  --meter_root "$METER_ROOT"
  --out_dir "$OUT_DIR"
  --nodata "$NODATA"
  --nodata_threshold "$NODATA_THRESHOLD"
  --progress_every "$PROGRESS_EVERY"
  --display_percentile "$DISPLAY_PERCENTILE"
)

if [[ "$OVERWRITE_CACHE" == "1" || "$OVERWRITE_CACHE" == "true" || "$OVERWRITE_CACHE" == "TRUE" ]]; then
  ARGS+=(--overwrite_cache)
fi

echo "============================================================"
echo "F066 full-river error distribution comparison"
echo "NORMALIZED-LOSS vs METER-LOSS"
date
echo "HOST=$(hostname)"
echo "CONDA_SH=$CONDA_SH"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PYTHON=$(command -v python)"
echo "SCRIPT=$SCRIPT"
echo "NORM_ROOT=$NORM_ROOT"
echo "METER_ROOT=$METER_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "NODATA=$NODATA"
echo "NODATA_THRESHOLD=$NODATA_THRESHOLD"
echo "PROGRESS_EVERY=$PROGRESS_EVERY"
echo "DISPLAY_PERCENTILE=$DISPLAY_PERCENTILE"
echo "OVERWRITE_CACHE=$OVERWRITE_CACHE"
echo "COMPARISON_SCOPE=unique geospatial full-river pixels for both experiments"
echo "ERROR_DEFINITION=Prediction minus GT in meters"
echo "MAIN_FIGURE=3 rows x 2 columns; absolute-error density and CDF"
echo "============================================================"

python -u "$SCRIPT" "${ARGS[@]}"

echo "============================================================"
echo "DONE F066"
echo "OUT_DIR=$OUT_DIR"
echo "MAIN_PNG=$OUT_DIR/figures/F065_abs_error_distribution_and_cdf_3x2_norm_vs_meter.png"
echo "MAIN_PDF=$OUT_DIR/figures/F065_abs_error_distribution_and_cdf_3x2_norm_vs_meter.pdf"
date
echo "============================================================"
