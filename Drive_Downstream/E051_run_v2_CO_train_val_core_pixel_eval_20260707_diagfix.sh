#!/usr/bin/env bash
#SBATCH -J e051_v2_diag
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/E051_v2_diag_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation_v2/logs/E051_v2_diag_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
CODE=${CODE:-$ROOT/mae_Retrain}
SCRIPT=${SCRIPT:-$WORK/script/E050_v2_dualmask_evaluate_core_pixel_metrics_20260707_diagfix.py}

TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2/Tiles_1m}
HOLDOUT_NAME=${HOLDOUT_NAME:-CO_UpperColorado_Topobathy_1_2020}
CV_ROOT=${CV_ROOT:-$WORK/cross_validation_v2}

RUN_NAME=${RUN_NAME:-train_holdout_${HOLDOUT_NAME}_v2_dualmask_corePixelLoss_e400_b4_acc4}
RUN_DIR=${RUN_DIR:-$CV_ROOT/runs/holdout_${HOLDOUT_NAME}/$RUN_NAME}
CKPT=${CKPT:-$RUN_DIR/checkpoint-best.pth}
SPLIT_DIR=${SPLIT_DIR:-$CV_ROOT/splits/holdout_${HOLDOUT_NAME}}
OUT_ROOT=${OUT_ROOT:-$RUN_DIR/eval_E050_v2_dualmask_core_pixel_diagnosis}

GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-1}
STD_SCALE=${STD_SCALE:-1.5}
USE_AMP=${USE_AMP:-1}
OVERWRITE_EVAL=${OVERWRITE_EVAL:-0}
RANK_METRIC=${RANK_METRIC:-rmse_m_core_loss_pixel}
WORST_VIS=${WORST_VIS:-20}
MEDIAN_VIS=${MEDIAN_VIS:-10}
BEST_VIS=${BEST_VIS:-10}
VIS_DPI=${VIS_DPI:-180}
NO_VISUALS=${NO_VISUALS:-0}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "$OUT_ROOT" "$CV_ROOT/logs"

echo "=== E051 MAE v2 train/val core pixel diagnosis ==="
date
echo "HOST=$(hostname)"
echo "RUN_DIR=$RUN_DIR"
echo "CKPT=$CKPT"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "SCRIPT=$SCRIPT"
echo "TILE_ROOT=$TILE_ROOT"
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RANK_METRIC=$RANK_METRIC"
echo "WORST_VIS=$WORST_VIS"
echo "MEDIAN_VIS=$MEDIAN_VIS"
echo "BEST_VIS=$BEST_VIS"
echo "VIS_DPI=$VIS_DPI"
echo "NO_VISUALS=$NO_VISUALS"
nvidia-smi || true

for f in \
  "$SCRIPT" "$CKPT" \
  "$SPLIT_DIR/train_tiles.txt" "$SPLIT_DIR/train_hidden.txt" "$SPLIT_DIR/train_loss.txt" \
  "$SPLIT_DIR/val_tiles.txt" "$SPLIT_DIR/val_hidden.txt" "$SPLIT_DIR/val_loss.txt"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required file: $f" >&2
    exit 2
  fi
done

if [[ -d "$OUT_ROOT" ]] && find "$OUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
  if [[ "$OVERWRITE_EVAL" == "1" ]]; then
    rm -rf "$OUT_ROOT"
    mkdir -p "$OUT_ROOT"
  else
    echo "[ERROR] Output is not empty: $OUT_ROOT"
    echo "Set OVERWRITE_EVAL=1 to replace it."
    exit 3
  fi
fi

AMP_ARGS=()
if [[ "$USE_AMP" == "1" || "$USE_AMP" == "true" || "$USE_AMP" == "TRUE" ]]; then
  AMP_ARGS+=(--amp)
fi

VIS_ARGS=()
if [[ "$NO_VISUALS" == "1" || "$NO_VISUALS" == "true" || "$NO_VISUALS" == "TRUE" ]]; then
  VIS_ARGS+=(--no_visuals)
fi

run_split() {
  local split="$1"
  local list_file="$SPLIT_DIR/${split}_tiles.txt"
  local hidden_file="$SPLIT_DIR/${split}_hidden.txt"
  local loss_file="$SPLIT_DIR/${split}_loss.txt"
  local out_dir="$OUT_ROOT/$split"

  echo
  echo "=== Evaluate split=$split ==="
  python -u "$SCRIPT" \
    --code_dir "$CODE" \
    --ckpt "$CKPT" \
    --data_root "$TILE_ROOT" \
    --list "$list_file" \
    --hidden_list "$hidden_file" \
    --loss_list "$loss_file" \
    --output_dir "$out_dir" \
    --split_name "$split" \
    --model mae_vit_large_patch16 \
    --input_size 336 \
    --in_chans 1 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device cuda \
    --nodata -999999 \
    --nodata_threshold -9999 \
    --tile_norm \
    --tile_norm_visible_only \
    --tile_norm_eps 1e-3 \
    --tile_norm_std_scale "$STD_SCALE" \
    --bottleneck_norm inst1d \
    --loss_mode mse \
    --mask_ratio 0.75 \
    --lcc_patch_threshold 0.5 \
    --loss_region_mode core \
    --core_patch_radius 3 \
    --min_valid_visible_patch_ratio 0.70 \
    --min_loss_pixel_count 1 \
    --rank_metric "$RANK_METRIC" \
    --worst_vis "$WORST_VIS" \
    --median_vis "$MEDIAN_VIS" \
    --best_vis "$BEST_VIS" \
    --vis_dpi "$VIS_DPI" \
    "${VIS_ARGS[@]}" \
    "${AMP_ARGS[@]}"
}

run_split train
run_split val

python - "$OUT_ROOT/train/summary.json" "$OUT_ROOT/val/summary.json" "$OUT_ROOT/compare_train_val_summary.json" <<'PY'
import json
import sys
from pathlib import Path

train = json.loads(Path(sys.argv[1]).read_text())
val = json.loads(Path(sys.argv[2]).read_text())

def get(d, path):
    cur = d
    for p in path:
        cur = cur[p]
    return cur

summary = {
    "train": {
        "n_tiles": train["n_tiles"],
        "rmse_m_core_loss_pixel": get(train, ["global_pixel_weighted", "rmse_m_core_loss_pixel"]),
        "rmse_norm_core_loss_pixel": get(train, ["global_pixel_weighted", "rmse_norm_core_loss_pixel"]),
        "tile_std_safe": train["per_tile_summary"]["tile_std_safe"],
    },
    "val": {
        "n_tiles": val["n_tiles"],
        "rmse_m_core_loss_pixel": get(val, ["global_pixel_weighted", "rmse_m_core_loss_pixel"]),
        "rmse_norm_core_loss_pixel": get(val, ["global_pixel_weighted", "rmse_norm_core_loss_pixel"]),
        "tile_std_safe": val["per_tile_summary"]["tile_std_safe"],
    },
    "interpretation_hint": (
        "If train rmse_norm is small but train rmse_m is large, inspect train/worst_by_tile_std_safe.csv "
        "and train/per_river_summary.csv. Meter RMSE scales with tile_std_safe under tile-wise normalization."
    )
}
Path(sys.argv[3]).write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "=== DONE E051 ==="
echo "$OUT_ROOT"
date
