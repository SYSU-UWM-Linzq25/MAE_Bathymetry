#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
SCRIPT_DIR=${SCRIPT_DIR:-$RELAX_ROOT/script}
CODE=${CODE:-$ROOT/mae_Retrain}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2_D001c_AnyVisiblePatch/Tiles_1m}
UP_CKPT=${UP_CKPT:-$ROOT/Upstream_Model_ReTrain/runs/Small_tilenorm_viscorr_336/checkpoint-best.pth}

echo "============================================================"
echo "D059 D001c RELAX four-model preflight"
echo "RELAX_ROOT=$RELAX_ROOT"
echo "CODE=$CODE"
echo "TILE_ROOT=$TILE_ROOT"
echo "UP_CKPT=$UP_CKPT"
echo "============================================================"

[[ -d "$TILE_ROOT" ]] || { echo "[ERROR] Missing TILE_ROOT" >&2; exit 2; }
[[ -f "$UP_CKPT" ]] || { echo "[ERROR] Missing upstream checkpoint" >&2; exit 2; }

for f in \
  "$CODE/main_pretrain_dem_unified_relax.py" \
  "$CODE/engine_pretrain_unified_relax.py" \
  "$SCRIPT_DIR/A020_relax_prepare_holdout_split.py" \
  "$SCRIPT_DIR/D040_relax_train_norm.sh" \
  "$SCRIPT_DIR/D041_relax_holdout_norm.sh" \
  "$SCRIPT_DIR/D044_relax_train_meter.sh" \
  "$SCRIPT_DIR/D045_relax_holdout_meter.sh" \
  "$SCRIPT_DIR/D048_relax_train_norm2meter.sh" \
  "$SCRIPT_DIR/D049_relax_holdout_norm2meter.sh" \
  "$SCRIPT_DIR/D054_relax_train_meter2norm.sh" \
  "$SCRIPT_DIR/D055_relax_holdout_meter2norm.sh" \
  "$SCRIPT_DIR/D058_relax_submit_all_four.sh"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing $f" >&2; exit 2; }
done

for f in "$SCRIPT_DIR"/*.sh; do
  bash -n "$f"
done

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE"

python -m py_compile \
  main_pretrain_dem_unified_relax.py \
  engine_pretrain_unified_relax.py

python main_pretrain_dem_unified_relax.py --help > /tmp/D059_relax_help_$$.txt 2>&1

for key in \
  optimization_loss normalized_mse meter_mae \
  best_metric early_stop_metric val_mae_m_mask val_loss \
  baseline_eval_before_training freeze_encoder \
  freeze_last_n_encoder_blocks train_hidden_list train_loss_list; do
  grep -q -- "$key" /tmp/D059_relax_help_$$.txt || {
    echo "[ERROR] Unified trainer help missing: $key" >&2
    exit 3
  }
done
rm -f /tmp/D059_relax_help_$$.txt

echo "[OK] Shell syntax"
echo "[OK] Unified Python compile/import/help"
echo "[OK] D001c input and upstream checkpoint"
echo
echo "Formal definitions:"
echo "  NormOnly      : normalized_mse -> select val_loss"
echo "  MeterOnly     : meter_mae      -> select val_mae_m_mask"
echo "  NormThenMeter : meter_mae      -> select val_mae_m_mask"
echo "  MeterThenNorm : normalized_mse -> select val_mae_m_mask"
echo
echo "Preflight passed."
