#!/usr/bin/env bash
# NUMBER-ALIGNED NAME: B010g_prepare_MilwaukeeRiverGroup_holdout.sh
# ORIGINAL BACKUP NAME: B011_prepare_MilwaukeeRiverGroup_holdout.sh
# Compatibility rule: scientific logic and default data/result roots are preserved unless explicitly noted.
set -euo pipefail

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
WORK=$ROOT/Downstream_Task_Bathy
SCRIPT_DIR=$WORK/script
PY=$SCRIPT_DIR/A010g_prepare_grouped_holdout_from_existing_LOORO.py

SOURCE_FOLD=${SOURCE_FOLD:-OR_SantiamRiverTB_Topobathy_1_D23}
GROUP_NAME=${GROUP_NAME:-MilwaukeeRiverGroup}
OVERWRITE=${OVERWRITE:-1}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn

[[ -f "$PY" ]] || { echo "[ERROR] Missing: $PY" >&2; exit 2; }

ARGS=(
  --work "$WORK"
  --group-name "$GROUP_NAME"
  --source-fold "$SOURCE_FOLD"
  --member Kletzch_Combined_UpMax3Null
  --member KewaFix2Null
  --member BadgerFinNull
  --member Estabrook_Combined
)

if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" ]]; then
  ARGS+=(--overwrite)
fi

echo "=== Prepare grouped LOORO holdout ==="
echo "WORK=$WORK"
echo "GROUP_NAME=$GROUP_NAME"
echo "SOURCE_FOLD=$SOURCE_FOLD"
echo "OVERWRITE=$OVERWRITE"
echo "PY=$PY"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"

python "$PY" "${ARGS[@]}"

SPLIT_DIR=$WORK/cross_validation/splits/holdout_${GROUP_NAME}
AUDIT_DIR=$WORK/cross_validation/audits/holdout_${GROUP_NAME}
SUMMARY=$SPLIT_DIR/fold_summary.json

[[ -f "$SPLIT_DIR/train.txt" ]] || { echo "[ERROR] Missing train.txt: $SPLIT_DIR" >&2; exit 3; }
[[ -f "$SPLIT_DIR/val.txt" ]] || { echo "[ERROR] Missing val.txt: $SPLIT_DIR" >&2; exit 4; }
[[ -f "$SPLIT_DIR/holdout_rivers.txt" ]] || { echo "[ERROR] Missing holdout_rivers.txt: $SPLIT_DIR" >&2; exit 5; }
[[ -f "$SUMMARY" ]] || { echo "[ERROR] Missing summary: $SUMMARY" >&2; exit 6; }

echo "=== Holdout rivers ==="
cat "$SPLIT_DIR/holdout_rivers.txt"

echo "=== Fold summary ==="
python - <<PY
import json
from pathlib import Path
summary = json.loads(Path("$SUMMARY").read_text())
keys = ["group_name", "source_fold", "train_total", "train_pass", "val_total", "val_pass"]
for k in keys:
    print(f"{k}: {summary.get(k)}")
print("val_river_counts:")
for k, v in summary.get("val_river_counts", {}).items():
    print(f"  {k}: {v}")
print("val_status:")
for k, v in summary.get("val_status", {}).items():
    print(f"  {k}: {v}")
PY

echo "=== DONE ==="
echo "$SPLIT_DIR"
echo "$AUDIT_DIR"
