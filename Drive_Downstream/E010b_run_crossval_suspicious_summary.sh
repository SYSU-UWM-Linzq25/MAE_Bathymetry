#!/usr/bin/env bash
# NUMBER-ALIGNED NAME: E010b_run_crossval_suspicious_summary.sh
# ORIGINAL BACKUP NAME: E042_run_crossval_suspicious_summary.sh
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
set -euo pipefail

# ============================================================
# E042: summarize E041 evaluation results and extract suspicious
#      folds / tiles for later source-chain inspection.
#
# Usage:
#   bash E010b_run_crossval_suspicious_summary.sh
#
# Optional:
#   METRIC=rmse_m_full_exact_pixel TOP_N_PER_FOLD=100 GLOBAL_TOP_N=500 \
#     bash E010b_run_crossval_suspicious_summary.sh
#
# By default all folds are included. To skip intentionally:
#   SKIP_FOLDS="OR_SantiamRiverTB_Topobathy_1_D23" bash E010b_run_crossval_suspicious_summary.sh
# ============================================================

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
WORK=$ROOT/Downstream_Task_Bathy
CV_ROOT=$WORK/cross_validation
PY=$WORK/script/E010c_analyze_crossval_evaluations_suspicious_tiles.py

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

[[ -f "$PY" ]] || { echo "[ERROR] Missing: $PY" >&2; exit 2; }

EVAL_ROOT=${EVAL_ROOT:-$CV_ROOT/evaluation}
METRIC=${METRIC:-rmse_m_core_exact_pixel}
TOP_N_PER_FOLD=${TOP_N_PER_FOLD:-50}
GLOBAL_TOP_N=${GLOBAL_TOP_N:-300}
SKIP_FOLDS=${SKIP_FOLDS:-}

echo "=== E042 summarize suspicious cross-val tiles ==="
echo "EVAL_ROOT=$EVAL_ROOT"
echo "METRIC=$METRIC"
echo "TOP_N_PER_FOLD=$TOP_N_PER_FOLD"
echo "GLOBAL_TOP_N=$GLOBAL_TOP_N"
echo "SKIP_FOLDS=$SKIP_FOLDS"
echo "PY=$PY"

python "$PY" \
  --eval_root "$EVAL_ROOT" \
  --metric "$METRIC" \
  --top_n_per_fold "$TOP_N_PER_FOLD" \
  --global_top_n "$GLOBAL_TOP_N" \
  --skip_folds "$SKIP_FOLDS"
