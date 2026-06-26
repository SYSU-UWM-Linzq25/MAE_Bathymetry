#!/usr/bin/env bash
set -euo pipefail

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
WORK=$ROOT/Downstream_Task_Bathy
SCRIPT_DIR=$WORK/script
PY=$SCRIPT_DIR/A010_prepare_LOORO_crossval_splits_and_audits.py

SOURCE_HOLDOUT=${SOURCE_HOLDOUT:-OR_SantiamRiverTB_Topobathy_1_D23}
OVERWRITE=${OVERWRITE:-1}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn

[[ -f "$PY" ]] || { echo "[ERROR] Missing: $PY" >&2; exit 2; }

ARGS=(--work "$WORK" --source_holdout "$SOURCE_HOLDOUT")
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" ]]; then
  ARGS+=(--overwrite)
fi

echo "=== Prepare LOORO cross-validation splits/audits ==="
echo "WORK=$WORK"
echo "SOURCE_HOLDOUT=$SOURCE_HOLDOUT"
echo "OVERWRITE=$OVERWRITE"
echo "PY=$PY"

python "$PY" "${ARGS[@]}"

MANIFEST=$WORK/cross_validation/LOORO_fold_manifest.csv
[[ -f "$MANIFEST" ]] || { echo "[ERROR] Missing manifest: $MANIFEST" >&2; exit 3; }

echo "=== Manifest ==="
column -s, -t "$MANIFEST" || cat "$MANIFEST"
echo "=== DONE ==="
echo "$MANIFEST"
