#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
RESULTS_ROOT=${RESULTS_ROOT:-$RELAX_ROOT/results}
SCRIPT_DIR=${SCRIPT_DIR:-$RELAX_ROOT/script}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2_D001c_AnyVisiblePatch/Tiles_1m}
DATA_TAG=${DATA_TAG:-D001cAnyVisiblePatch_D001NoDataSafe}
SPLIT_SCRIPT=${SPLIT_SCRIPT:-$SCRIPT_DIR/A020_relax_prepare_holdout_split.py}

NORM=${NORM:-$SCRIPT_DIR/D041_relax_holdout_norm.sh}
METER=${METER:-$SCRIPT_DIR/D045_relax_holdout_meter.sh}
N2M=${N2M:-$SCRIPT_DIR/D049_relax_holdout_norm2meter.sh}
M2N=${M2N:-$SCRIPT_DIR/D055_relax_holdout_meter2norm.sh}

GPU_ID=${GPU_ID:-0}
FRESH_RUN=${FRESH_RUN:-1}
OVERWRITE_STAGE2=${OVERWRITE_STAGE2:-0}

NORM_EPOCHS=${NORM_EPOCHS:-400}
METER_EPOCHS=${METER_EPOCHS:-400}
N2M_EPOCHS=${N2M_EPOCHS:-120}
M2N_EPOCHS=${M2N_EPOCHS:-120}

NORM_PATIENCE=${NORM_PATIENCE:-60}
METER_PATIENCE=${METER_PATIENCE:-60}
N2M_PATIENCE=${N2M_PATIENCE:-30}
M2N_PATIENCE=${M2N_PATIENCE:-30}

N2M_LR=${N2M_LR:-1e-5}
M2N_LR=${M2N_LR:-1e-5}

# D058 prepares split files on the submit/login node before calling sbatch.
# Therefore it must activate the same Python environment used by training.
module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

PYTHON_BIN=${PYTHON_BIN:-python}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "[ERROR] Python is unavailable after conda activation." >&2
  echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
  exit 2
}

echo "[ENV] CONDA_PREFIX=$CONDA_PREFIX"
echo "[ENV] PYTHON=$($PYTHON_BIN -c 'import sys; print(sys.executable)')"

for f in "$SPLIT_SCRIPT" "$NORM" "$METER" "$N2M" "$M2N"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing $f" >&2; exit 2; }
done
[[ -d "$TILE_ROOT" ]] || { echo "[ERROR] Missing TILE_ROOT: $TILE_ROOT" >&2; exit 2; }

mkdir -p "$RESULTS_ROOT/NormOnly/splits" "$RESULTS_ROOT/MeterOnly" \
  "$RESULTS_ROOT/NormThenMeter" "$RESULTS_ROOT/MeterThenNorm"

prepare_split() {
  local preset="$1" holdout_name="$2" holdout_river="$3"
  local split_dir="$RESULTS_ROOT/NormOnly/splits/holdout_${preset}_${DATA_TAG}"
  "$PYTHON_BIN" "$SPLIT_SCRIPT" \
    --holdout_name "$holdout_name" \
    --holdout_rivers "$holdout_river" \
    --tile_root "$TILE_ROOT" \
    --out_dir "$split_dir"
}

prepare_split CA CA_KlamathRiver_TopoBathy_2018_D18 CA_KlamathRiver_TopoBathy_2018_D18
prepare_split CO CO_UpperColorado_Topobathy_1_2020 CO_UpperColorado_Topobathy_1_2020
prepare_split Santiam OR_SantiamRiverTB_Topobathy_1_D23 OR_SantiamRiverTB_Topobathy_1_D23

submit_norm() {
  sbatch --parsable \
    --export=ALL,HOLDOUT_PRESET="$1",GPU_ID="$GPU_ID",EPOCHS="$NORM_EPOCHS",PATIENCE="$NORM_PATIENCE",FRESH_RUN="$FRESH_RUN",PREPARE_SPLIT=0 \
    "$NORM"
}
submit_meter() {
  sbatch --parsable \
    --export=ALL,HOLDOUT_PRESET="$1",GPU_ID="$GPU_ID",EPOCHS="$METER_EPOCHS",PATIENCE="$METER_PATIENCE",FRESH_RUN="$FRESH_RUN" \
    "$METER"
}
submit_n2m() {
  sbatch --parsable --dependency="afterok:$2" \
    --export=ALL,HOLDOUT_PRESET="$1",GPU_ID="$GPU_ID",EPOCHS="$N2M_EPOCHS",PATIENCE="$N2M_PATIENCE",LR="$N2M_LR",FRESH_RUN="$FRESH_RUN",OVERWRITE_STAGE2="$OVERWRITE_STAGE2" \
    "$N2M"
}
submit_m2n() {
  sbatch --parsable --dependency="afterok:$2" \
    --export=ALL,HOLDOUT_PRESET="$1",GPU_ID="$GPU_ID",EPOCHS="$M2N_EPOCHS",PATIENCE="$M2N_PATIENCE",LR="$M2N_LR",FRESH_RUN="$FRESH_RUN",OVERWRITE_STAGE2="$OVERWRITE_STAGE2" \
    "$M2N"
}

declare -A NORM_JID METER_JID N2M_JID M2N_JID
for p in CA CO Santiam; do
  NORM_JID[$p]=$(submit_norm "$p")
  METER_JID[$p]=$(submit_meter "$p")
  N2M_JID[$p]=$(submit_n2m "$p" "${NORM_JID[$p]}")
  M2N_JID[$p]=$(submit_m2n "$p" "${METER_JID[$p]}")
done

echo "============================================================"
echo "D001c RELAX four-model training submitted"
for p in CA CO Santiam; do
  echo "$p"
  echo "  NormOnly      : ${NORM_JID[$p]}"
  echo "  MeterOnly     : ${METER_JID[$p]}"
  echo "  NormThenMeter : ${N2M_JID[$p]} (afterok:${NORM_JID[$p]})"
  echo "  MeterThenNorm : ${M2N_JID[$p]} (afterok:${METER_JID[$p]})"
done
echo "============================================================"
