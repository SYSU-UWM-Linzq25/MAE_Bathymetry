#!/bin/bash
#SBATCH -J cv_s4_core
#SBATCH -p gpu
#SBATCH -w execute-3000
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 7-00:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation/logs/train_LOORO_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/cross_validation/logs/train_LOORO_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  sbatch $0 <VAL_RIVER_NAME>"
  echo
  echo "Example:"
  echo "  sbatch $0 MD_PotomacRiver_Bathy_2019"
  echo
  echo "This script expects:"
  echo "  Downstream_Task_Bathy/cross_validation/splits/holdout_<VAL_RIVER_NAME>"
  echo "  Downstream_Task_Bathy/cross_validation/audits/holdout_<VAL_RIVER_NAME>"
  exit 1
fi
VAL_RIVER="$1"
SAFE_VAL=$(echo "$VAL_RIVER" | sed 's/[^A-Za-z0-9_]/_/g')

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
CODE=$ROOT/mae_Retrain
WORK=$ROOT/Downstream_Task_Bathy
CV_ROOT=$WORK/cross_validation

SPLIT=$CV_ROOT/splits/holdout_${SAFE_VAL}
AUDIT_DIR=$CV_ROOT/audits/holdout_${SAFE_VAL}
TRAIN_AUDIT=$AUDIT_DIR/train_core_loss_audit.csv
VAL_AUDIT=$AUDIT_DIR/val_core_loss_audit.csv

BATH=$ROOT/Data/Tiles_for_Training_1m/1m_Tiles
MASK=$ROOT/Data/TilesMask_for_Training_1m/1m_Tiles
UP_CKPT=$ROOT/Upstream_Model_ReTrain/runs/Small_tilenorm_viscorr_336/checkpoint-best.pth
ENTRY=$CODE/main_pretrain_dem.py

STD_SCALE=${STD_SCALE:-1.5}
END_EPOCH=${END_EPOCH:-400}
NODATA=${NODATA:-"-999999"}
NODATA_THRESHOLD=${NODATA_THRESHOLD:-"-9999"}
MIN_VISIBLE_PATCH_RATIO=${MIN_VISIBLE_PATCH_RATIO:-0.70}
LOSS_REGION_MODE=${LOSS_REGION_MODE:-core}
CORE_PATCH_RADIUS=${CORE_PATCH_RADIUS:-3}
MIN_CORE_VALID_PATCH_RATIO=${MIN_CORE_VALID_PATCH_RATIO:-0.85}
MIN_CORE_PREDICTION_PATCH_RATIO=${MIN_CORE_PREDICTION_PATCH_RATIO:-0.02}
MAX_CORE_PREDICTION_PATCH_RATIO=${MAX_CORE_PREDICTION_PATCH_RATIO:-0.90}
DATA_FIX_TAG=${DATA_FIX_TAG:-allRiverCanonicalND}
CV_TAG=${CV_TAG:-LOORO_v1}

RUN_NAME=stage4_${CV_TAG}_holdout_${SAFE_VAL}_exact_freeze_decoder_tilenormVis_${DATA_FIX_TAG}_nodataSafe_${LOSS_REGION_MODE}Loss_r${CORE_PATCH_RADIUS}_cv${MIN_CORE_VALID_PATCH_RATIO//./p}_cp${MIN_CORE_PREDICTION_PATCH_RATIO//./p}-${MAX_CORE_PREDICTION_PATCH_RATIO//./p}_std${STD_SCALE//./p}_1m_e${END_EPOCH}
OUT=$CV_ROOT/runs/holdout_${SAFE_VAL}/$RUN_NAME

mkdir -p "$OUT" "$CV_ROOT/logs"

echo "=== LOORO CROSS-VALIDATION TRAINING ==="
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOST=$(hostname)"
echo "VAL_RIVER=$VAL_RIVER"
# Define GPU_ID before any reference because set -u is enabled.
GPU_ID=${GPU_ID:-0}
echo "GPU_ID=$GPU_ID"
echo "OUT=$OUT"
date

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

# Keep the same GPU behavior as D005:
#   node fixed by #SBATCH -w execute-3000
#   GPU_ID defaults to 0
# Because Slurm does not expose GPU GRES on this system, do not submit several
# folds at the same time unless you intentionally change node/GPU settings.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available inside this Slurm job.")
    sys.exit(10)

free_b, total_b = torch.cuda.mem_get_info()
free_gb = free_b / (1024 ** 3)
total_gb = total_b / (1024 ** 3)
print(f"[GPU CHECK] device={torch.cuda.current_device()} "
      f"name={torch.cuda.get_device_name(0)} "
      f"free={free_gb:.2f} GiB total={total_gb:.2f} GiB")

if free_gb < 6.0:
    print("[ERROR] Assigned GPU has less than 6 GiB free. "
          "This usually means the GPU was not reserved correctly or is shared with another job.")
    sys.exit(11)
PY

for f in \
  "$ENTRY" \
  "$SPLIT/train.txt" \
  "$SPLIT/val.txt" \
  "$SPLIT/train_masks.txt" \
  "$SPLIT/val_masks.txt" \
  "$TRAIN_AUDIT" \
  "$VAL_AUDIT" \
  "$UP_CKPT"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing required file: $f" >&2; exit 3; }
done

echo "=== Fold split counts ==="
wc -l "$SPLIT/train.txt" "$SPLIT/train_masks.txt"
wc -l "$SPLIT/val.txt" "$SPLIT/val_masks.txt"

echo "=== Holdout sanity check ==="
if grep -q "$VAL_RIVER" "$SPLIT/train.txt"; then
  echo "[ERROR] Holdout river appears in train.txt." >&2
  grep "$VAL_RIVER" "$SPLIT/train.txt" | head
  exit 4
fi
if grep -v "$VAL_RIVER" "$SPLIT/val.txt" | grep -q .; then
  echo "[ERROR] val.txt contains a different river." >&2
  grep -v "$VAL_RIVER" "$SPLIT/val.txt" | head
  exit 5
fi

python - "$TRAIN_AUDIT" "$VAL_AUDIT" <<'PY'
import csv
import sys
from collections import Counter
for label, filename in zip(("train", "val"), sys.argv[1:]):
    counts = Counter()
    total = 0
    with open(filename, newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            counts[(row.get("status") or "UNKNOWN").strip()] += 1
    print(
        f"[AUDIT] {label}: total={total} "
        f"pass={counts.get('PASS', 0)} drop={total-counts.get('PASS', 0)} "
        f"status={dict(sorted(counts.items()))}"
    )
PY

cp -pf "$SPLIT"/{train.txt,val.txt,train_masks.txt,val_masks.txt} "$OUT/"
cp -pf "$TRAIN_AUDIT" "$OUT/input_train_core_loss_audit.csv"
cp -pf "$VAL_AUDIT" "$OUT/input_val_core_loss_audit.csv"

python "$ENTRY" --help > "$OUT/help.txt" 2>&1 || true

CKPT=$(find "$OUT" -maxdepth 1 -type f -name 'checkpoint-*.pth' ! -name 'checkpoint-best.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1{print $2}' || true)
if [[ -z "${CKPT:-}" && -f "$OUT/checkpoint-best.pth" ]]; then
  CKPT="$OUT/checkpoint-best.pth"
fi

RESUME_ARGS=()
if [[ -n "${CKPT:-}" ]]; then
  echo "[RESUME] Found fold checkpoint: $CKPT"
  RESUME_ARGS=(--resume "$CKPT")
else
  echo "[RESUME] Fresh fold; initialize from upstream checkpoint only."
fi

PYTHONUNBUFFERED=1 python -u "$ENTRY" \
  --device cuda \
  --data_root "$BATH" \
  --train_list "$SPLIT/train.txt" \
  --val_list "$SPLIT/val.txt" \
  --lcc_mask_path "$MASK" \
  --train_lcc_list "$SPLIT/train_masks.txt" \
  --val_lcc_list "$SPLIT/val_masks.txt" \
  --output_dir "$OUT" \
  --log_dir "$OUT/tb" \
  --model mae_vit_large_patch16 \
  --input_size 336 \
  --in_chans 1 \
  --nodata "$NODATA" \
  --nodata_threshold "$NODATA_THRESHOLD" \
  --batch_size 4 \
  --accum_iter 4 \
  --epochs "$END_EPOCH" \
  --num_workers 8 \
  --bottleneck_norm inst1d \
  --pin_mem \
  --tile_norm \
  --tile_norm_visible_only \
  --tile_norm_eps 1e-3 \
  --tile_norm_std_scale "$STD_SCALE" \
  --init_ckpt "$UP_CKPT" \
  --freeze_encoder \
  --freeze_last_n_encoder_blocks 0 \
  --lcc_mask_mode exact \
  --loss_on_lcc_only \
  --eval_rmse \
  --best_metric val_rmse_m_mask \
  --early_stop_metric val_rmse_m_mask \
  --early_stop_patience 60 \
  --early_stop_min_delta 0.001 \
  --early_stop_warmup_epochs 20 \
  --plot_every 1 \
  --vis_every 20 \
  --vis_n 10 \
  --stats_max_files 1000 \
  --min_lcc_patch_ratio 0.0001 \
  --max_lcc_patch_ratio 0.80 \
  --min_valid_visible_patch_ratio "$MIN_VISIBLE_PATCH_RATIO" \
  --loss_region_mode "$LOSS_REGION_MODE" \
  --core_patch_radius "$CORE_PATCH_RADIUS" \
  --min_core_valid_patch_ratio "$MIN_CORE_VALID_PATCH_RATIO" \
  --min_core_prediction_patch_ratio "$MIN_CORE_PREDICTION_PATCH_RATIO" \
  --max_core_prediction_patch_ratio "$MAX_CORE_PREDICTION_PATCH_RATIO" \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --warmup_epochs 5 \
  "${RESUME_ARGS[@]}"

date
echo "=== LOORO FOLD DONE ==="
echo "VAL_RIVER=$VAL_RIVER"
echo "RUN=$OUT"
