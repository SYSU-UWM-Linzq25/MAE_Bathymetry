#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
CODE=${CODE:-$ROOT/mae_Retrain}
OUT_DIR=${OUT_DIR:-$RELAX_ROOT/results/EncoderUnfreeze_Preflight}
mkdir -p "$OUT_DIR"

REPORT="$OUT_DIR/D053_encoder_unfreeze_source_report.txt"
{
  echo "DATE=$(date -Is)"
  echo "CODE=$CODE"
  echo
  echo "=== help matches ==="
  for py in \
    main_pretrain_dem.py \
    main_pretrain_dem_meterMAE_BaselineEval_D030_20260713.py \
    main_pretrain_dem_Stage2NormMeterSelect_D034_20260713.py; do
    if [[ -f "$CODE/$py" ]]; then
      echo "--- $py --help ---"
      python "$CODE/$py" --help 2>&1 | grep -Ei -C 3 'freeze_encoder|freeze_last_n_encoder_blocks|unfreeze' || true
    fi
  done
  echo
  echo "=== source matches ==="
  grep -R -n -C 8 -E 'freeze_last_n_encoder_blocks|freeze_encoder|unfreeze' "$CODE" --include='*.py' || true
} | tee "$REPORT"

echo "Report written: $REPORT"
