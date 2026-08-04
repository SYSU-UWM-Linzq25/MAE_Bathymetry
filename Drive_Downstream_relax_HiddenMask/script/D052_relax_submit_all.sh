#!/usr/bin/env bash
set -euo pipefail

# LEGACY THREE-MODEL SUBMITTER. Use D058_relax_submit_all_four.sh.

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

GPU_ID=${GPU_ID:-0}
FRESH_RUN=${FRESH_RUN:-1}
OVERWRITE_STAGE2=${OVERWRITE_STAGE2:-0}
NORM_EPOCHS=${NORM_EPOCHS:-400}
METER_EPOCHS=${METER_EPOCHS:-400}
N2M_EPOCHS=${N2M_EPOCHS:-120}
NORM_PATIENCE=${NORM_PATIENCE:-60}
METER_PATIENCE=${METER_PATIENCE:-60}
N2M_PATIENCE=${N2M_PATIENCE:-30}
N2M_LR=${N2M_LR:-1e-5}

for f in "$SPLIT_SCRIPT" "$NORM" "$METER" "$N2M"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing $f" >&2; exit 2; }
done
[[ -d "$TILE_ROOT" ]] || { echo "[ERROR] Missing D001c TILE_ROOT: $TILE_ROOT" >&2; exit 2; }
mkdir -p "$RESULTS_ROOT/NormOnly/splits" "$RESULTS_ROOT/MeterOnly" "$RESULTS_ROOT/NormThenMeter"

prepare_split() {
  local preset="$1" holdout_name="$2" holdout_river="$3"
  local split_dir="$RESULTS_ROOT/NormOnly/splits/holdout_${preset}_${DATA_TAG}"
  echo "[SPLIT] $preset -> $split_dir"
  python "$SPLIT_SCRIPT" \
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

declare -A NORM_JID METER_JID N2M_JID
for p in CA CO Santiam; do
  NORM_JID[$p]=$(submit_norm "$p")
  METER_JID[$p]=$(submit_meter "$p")
  N2M_JID[$p]=$(submit_n2m "$p" "${NORM_JID[$p]}")
done

echo "============================================================"
echo "D001c RELAX three-model training submitted"
for p in CA CO Santiam; do
  echo "$p"
  echo "  NormOnly      : ${NORM_JID[$p]}"
  echo "  MeterOnly     : ${METER_JID[$p]} (parallel)"
  echo "  NormThenMeter : ${N2M_JID[$p]} (afterok:${NORM_JID[$p]})"
done
echo "Results root: $RESULTS_ROOT"
echo "============================================================"
