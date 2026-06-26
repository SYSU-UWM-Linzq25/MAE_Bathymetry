#!/bin/bash
#SBATCH -J e031_s4_predOnly
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/E031_eval_s4_predOnly_coreBox_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/E031_eval_s4_predOnly_coreBox_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 <VAL_RIVER_NAME>"
  echo "Example: sbatch $0 OR_SantiamRiverTB_Topobathy_1_D23"
  exit 1
fi

VAL_RIVER="$1"
SAFE_VAL=$(echo "$VAL_RIVER" | sed 's/[^A-Za-z0-9_]/_/g')

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn

GPU_ID=${GPU_ID:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
CODE=$ROOT/mae_Retrain
WORK=$ROOT/Downstream_Task_Bathy
SCRIPT=$WORK/script/E030_evaluate_stage4_predictionOnly_coreBox.py
SPLIT=$WORK/splits/bathy_lcc_1m_holdout_${SAFE_VAL}
BATH=$ROOT/Data/Tiles_for_Training_1m/1m_Tiles
MASK=$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles

STD_SCALE=${STD_SCALE:-1.5}
END_EPOCH=${END_EPOCH:-400}
LOSS_REGION_MODE=${LOSS_REGION_MODE:-core}
CORE_PATCH_RADIUS=${CORE_PATCH_RADIUS:-3}
MIN_CORE_VALID_PATCH_RATIO=${MIN_CORE_VALID_PATCH_RATIO:-0.85}
MIN_CORE_PREDICTION_PATCH_RATIO=${MIN_CORE_PREDICTION_PATCH_RATIO:-0.02}
MAX_CORE_PREDICTION_PATCH_RATIO=${MAX_CORE_PREDICTION_PATCH_RATIO:-0.90}
DATA_FIX_TAG=${DATA_FIX_TAG:-allRiverCanonicalND}

RUN_NAME=stage4_bathy_finalmask_holdout_${SAFE_VAL}_exact_freeze_decoder_tilenormVis_${DATA_FIX_TAG}_nodataSafe_${LOSS_REGION_MODE}Loss_r${CORE_PATCH_RADIUS}_cv${MIN_CORE_VALID_PATCH_RATIO//./p}_cp${MIN_CORE_PREDICTION_PATCH_RATIO//./p}-${MAX_CORE_PREDICTION_PATCH_RATIO//./p}_std${STD_SCALE//./p}_1m_e${END_EPOCH}
RUN=$WORK/runs/$RUN_NAME
CKPT=${CKPT:-$RUN/checkpoint-best.pth}
OUT=${OUT:-$RUN/eval_E030_val_predictionOnly_coreBox_exact_pixel}

mkdir -p "$WORK/logs"

for f in "$SCRIPT" "$CKPT" "$SPLIT/val.txt" "$SPLIT/val_masks.txt"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required file: $f"
    exit 2
  fi
done

if [[ -d "$OUT" ]] && find "$OUT" -mindepth 1 -print -quit | grep -q .; then
  if [[ "${OVERWRITE_EVAL:-0}" == "1" ]]; then
    rm -rf -- "$OUT"
  else
    echo "[ERROR] Evaluation output is not empty: $OUT"
    echo "Set OVERWRITE_EVAL=1 only when you intentionally want to replace it."
    exit 3
  fi
fi
mkdir -p "$OUT"

echo "=== E030 STAGE4 PREDICTION-ONLY CORE-BOX EVALUATION ==="
echo "HOST=$(hostname)"
echo "VAL_RIVER=$VAL_RIVER"
echo "DATA_FIX_TAG=$DATA_FIX_TAG"
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RUN=$RUN"
echo "CKPT=$CKPT"
echo "SCRIPT=$SCRIPT"
echo "OUT=$OUT"
nvidia-smi || true

python -u "$SCRIPT" \
  --code_dir "$CODE" \
  --ckpt "$CKPT" \
  --data_root "$BATH" \
  --list "$SPLIT/val.txt" \
  --lcc_mask_path "$MASK" \
  --lcc_list "$SPLIT/val_masks.txt" \
  --output_dir "$OUT" \
  --split_name "val_${SAFE_VAL}_${DATA_FIX_TAG}" \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --in_chans 1 \
  --batch_size 4 \
  --num_workers 8 \
  --device cuda \
  --amp \
  --nodata -999999 \
  --nodata_threshold -9999 \
  --output_nodata -999999 \
  --tile_norm \
  --tile_norm_visible_only \
  --tile_norm_eps 1e-3 \
  --tile_norm_std_scale "$STD_SCALE" \
  --bottleneck_norm inst1d \
  --loss_mode mse \
  --mask_ratio 0.75 \
  --lcc_priority 10.0 \
  --lcc_patch_threshold 0.5 \
  --min_prediction_patch_ratio 0.0001 \
  --max_prediction_patch_ratio 0.80 \
  --min_valid_visible_patch_ratio 0.70 \
  --loss_region_mode "$LOSS_REGION_MODE" \
  --core_patch_radius "$CORE_PATCH_RADIUS" \
  --min_core_valid_patch_ratio "$MIN_CORE_VALID_PATCH_RATIO" \
  --min_core_prediction_patch_ratio "$MIN_CORE_PREDICTION_PATCH_RATIO" \
  --max_core_prediction_patch_ratio "$MAX_CORE_PREDICTION_PATCH_RATIO" \
  --rank_metric rmse_m_core_exact_pixel \
  --worst_vis 30 \
  --median_vis 10 \
  --best_vis 10

echo "=== SUMMARY ==="
cat "$OUT/summary.json"
echo "=== DONE ==="
echo "$OUT"
