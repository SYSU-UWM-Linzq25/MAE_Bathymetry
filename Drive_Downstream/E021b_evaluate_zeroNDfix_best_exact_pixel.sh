#!/usr/bin/env bash
#SBATCH -J eval_zeroND_best
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/eval_zeroND_best_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/eval_zeroND_best_%j.err

set -euo pipefail

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
CODE=$ROOT/mae_Retrain
WORK=$ROOT/Downstream_Task_Bathy
SPLIT=$WORK/splits/bathy_lcc_1m_holdout_OR_SantiamRiverTB_Topobathy_1_D23

RUN=$WORK/runs/stage4_bathy_finalmask_holdout_OR_SantiamRiverTB_Topobathy_1_D23_exact_freeze_decoder_tilenormVis_zeroNDfix_nodataSafe_coreLoss_r3_cv0p85_cp0p02-0p90_std1p5_1m_e400
CKPT=$RUN/checkpoint-best.pth
OUT=$RUN/eval_val_zeroNDfix_best_exact_pixel

DATA=$ROOT/Data/Tiles_for_Training_1m/1m_Tiles
MASK=$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles

EVAL_PY=$WORK/script/E020_evaluate_stage4_nodata_core_per_tile.py
if [[ -f "$WORK/script/E020_evaluate_stage4_nodata_core_per_tile_fixed.py" ]]; then
    EVAL_PY=$WORK/script/E020_evaluate_stage4_nodata_core_per_tile_fixed.py
fi

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn

for f in \
    "$CKPT" \
    "$SPLIT/val.txt" \
    "$SPLIT/val_masks.txt" \
    "$EVAL_PY"; do
    [[ -f "$f" ]] || { echo "[ERROR] Missing file: $f" >&2; exit 2; }
done

mkdir -p "$WORK/logs"

echo "=== ZERO-ND VALIDATION EXACT-PIXEL EVALUATION ==="
echo "HOST=$(hostname)"
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CKPT=$CKPT"
echo "EVAL_PY=$EVAL_PY"
echo "OUT=$OUT"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA is unavailable.")
free_b, total_b = torch.cuda.mem_get_info(0)
print(
    f"[GPU CHECK] device=0 name={torch.cuda.get_device_name(0)} "
    f"free={free_b/1024**3:.2f} GiB total={total_b/1024**3:.2f} GiB"
)
if free_b < 6 * 1024**3:
    raise SystemExit("[ERROR] Assigned GPU has less than 6 GiB free.")
PY

# Avoid mixing a partial prior evaluation with this run.
if [[ -d "$OUT" ]] && find "$OUT" -mindepth 1 -print -quit | grep -q .; then
    echo "[ERROR] Evaluation output already exists and is not empty: $OUT" >&2
    echo "Rename or remove it before rerunning." >&2
    exit 3
fi
mkdir -p "$OUT"

python "$EVAL_PY" \
  --code_dir "$CODE" \
  --ckpt "$CKPT" \
  --data_root "$DATA" \
  --list "$SPLIT/val.txt" \
  --lcc_mask_path "$MASK" \
  --lcc_list "$SPLIT/val_masks.txt" \
  --output_dir "$OUT" \
  --split_name val_zeroNDfix_best \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --in_chans 1 \
  --batch_size 4 \
  --num_workers 4 \
  --device cuda \
  --amp \
  --nodata -999999 \
  --nodata_threshold -9999 \
  --output_nodata -999999 \
  --tile_norm \
  --tile_norm_visible_only \
  --tile_norm_eps 0.001 \
  --tile_norm_std_scale 1.5 \
  --bottleneck_norm inst1d \
  --loss_mode mse \
  --mask_ratio 0.75 \
  --lcc_priority 10.0 \
  --lcc_patch_threshold 0.5 \
  --min_prediction_patch_ratio 0.0001 \
  --max_prediction_patch_ratio 0.80 \
  --min_valid_visible_patch_ratio 0.70 \
  --loss_region_mode core \
  --core_patch_radius 3 \
  --min_core_valid_patch_ratio 0.85 \
  --min_core_prediction_patch_ratio 0.02 \
  --max_core_prediction_patch_ratio 0.90 \
  --worst_vis 30 \
  --median_vis 10 \
  --best_vis 10 \
  --rank_metric rmse_m_core_exact_pixel

echo "=== SUMMARY ==="
cat "$OUT/summary.json"
echo "=== DONE ==="
echo "$OUT"
