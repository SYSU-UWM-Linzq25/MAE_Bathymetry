#!/usr/bin/env bash
#SBATCH -J F050_relax_norm_fullriver
#SBATCH -p HydroIntel
#SBATCH -w execute-4006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask
#SBATCH --mail-user=zequnlin@uwm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# F050: corrected full-river reconstruction for the D001c AnyVisiblePatch normalized-objective model.
# Input tiles MUST come from E001c AnyVisiblePatch, not the older E001 branch.

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
RELAX_ROOT=${RELAX_ROOT:-$ROOT/Downstream_Task_Bathy_relax_HiddenMask}
RESULTS_ROOT=${RESULTS_ROOT:-$RELAX_ROOT/results}
CODE=${CODE:-$ROOT/mae_Retrain}
SCRIPT=${SCRIPT:-$RELAX_ROOT/script/F049_relax_fullriver_predict_NormalizedObjective_TileAvgVRT_CoreFinalLossOnly.py}

MODEL_ROOT=${MODEL_ROOT:-$RESULTS_ROOT/NormOnly}
RUN_TAG=${RUN_TAG:-D001cAnyVisiblePatch_D001NoDataSafe}
TILE_ROOT=${TILE_ROOT:-/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_FullRiver_E001c_AnyVisiblePatch/Tiles_1m}
OUT_BASE=${OUT_BASE:-$RESULTS_ROOT/FullRiver_Predictions_F049_NormalizedObjective_D001cAnyVisiblePatch}

HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}
RIVERS=${RIVERS:-}
GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-4}
STD_SCALE=${STD_SCALE:-1.5}
USE_AMP=${USE_AMP:-1}
OVERWRITE=${OVERWRITE:-0}
RESUME=${RESUME:-1}
RUN_DIR=${RUN_DIR:-}
CKPT=${CKPT:-}
OUT_DIR=${OUT_DIR:-}

RUNTIME_LOG_DIR="$OUT_BASE/logs"
mkdir -p "$RUNTIME_LOG_DIR"
RUNTIME_JOB_ID=${SLURM_JOB_ID:-local_$$}
exec >"$RUNTIME_LOG_DIR/F050_relax_normalized_fullriver_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.out" \
     2>"$RUNTIME_LOG_DIR/F050_relax_normalized_fullriver_${HOLDOUT_PRESET}_${RUNTIME_JOB_ID}.err"

safe_name() { echo "$1" | sed 's/[^A-Za-z0-9_]/_/g'; }

latest_formal_run_with_ckpt() {
  local parent="$1"
  local hit=""
  [[ -d "$parent" ]] || { echo ""; return 0; }

  hit=$(find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth \
    -printf '%T@ %h\n' 2>/dev/null \
    | awk '$2 ~ /train_holdout_/ && $2 !~ /smoke/' \
    | sort -nr | awk 'NR==1{print $2}')

  if [[ -z "$hit" ]]; then
    hit=$(find "$parent" -mindepth 2 -maxdepth 2 -type f -name checkpoint-best.pth \
      -printf '%T@ %h\n' 2>/dev/null \
      | sort -nr | awk 'NR==1{print $2}')
  fi
  echo "$hit"
}

case "$HOLDOUT_PRESET" in
  CO)      DEFAULT_RIVERS="CO_UpperColorado_Topobathy_1_2020" ;;
  CA)      DEFAULT_RIVERS="CA_KlamathRiver_TopoBathy_2018_D18" ;;
  Santiam) DEFAULT_RIVERS="OR_SantiamRiverTB_Topobathy_1_D23" ;;
  *) echo "[ERROR] F050 formal workflow supports CA, CO, Santiam. Got $HOLDOUT_PRESET" >&2; exit 2 ;;
esac
RIVERS=${RIVERS:-$DEFAULT_RIVERS}

SAFE_PRESET=$(safe_name "$HOLDOUT_PRESET")
RUN_PARENT="$MODEL_ROOT/runs/holdout_${SAFE_PRESET}_${RUN_TAG}"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR=$(latest_formal_run_with_ckpt "$RUN_PARENT")
fi
if [[ -z "$RUN_DIR" ]]; then
  echo "[ERROR] Could not find formal normalized-objective checkpoint under: $RUN_PARENT" >&2
  exit 2
fi

CKPT=${CKPT:-$RUN_DIR/checkpoint-best.pth}
OUT_DIR=${OUT_DIR:-$OUT_BASE/holdout_${SAFE_PRESET}_${RUN_TAG}}

module purge || true
source /home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/conda_envs/mae_zequn"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

for f in \
  "$SCRIPT" "$CKPT" \
  "$TILE_ROOT/FullRiver_tile" "$TILE_ROOT/Hidden_Mask" \
  "$TILE_ROOT/Loss_Mask_Pixel" "$TILE_ROOT/Core_Loss_Mask_Pixel"; do
  [[ -e "$f" ]] || { echo "[ERROR] Missing required E001c path: $f" >&2; exit 2; }
done

if [[ "$TILE_ROOT" != *"E001c_AnyVisiblePatch"* ]]; then
  echo "[ERROR] TILE_ROOT is not the E001c AnyVisiblePatch branch:" >&2
  echo "        $TILE_ROOT" >&2
  exit 2
fi

if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$OUT_DIR"
  elif [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "TRUE" ]]; then
    echo "[RESUME] $OUT_DIR"
  else
    echo "[ERROR] Output is not empty: $OUT_DIR" >&2
    exit 3
  fi
fi
mkdir -p "$OUT_DIR"

AMP_ARGS=()
[[ "$USE_AMP" == "1" || "$USE_AMP" == "true" || "$USE_AMP" == "TRUE" ]] && AMP_ARGS+=(--amp)
RESUME_ARGS=()
[[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "TRUE" ]] && RESUME_ARGS+=(--resume)

read -r -a RIVER_ARRAY <<< "$RIVERS"
[[ ${#RIVER_ARRAY[@]} -gt 0 ]] || { echo "[ERROR] RIVERS is empty." >&2; exit 2; }

echo "============================================================"
echo "F050 D001c/E001c AnyVisiblePatch normalized objective full-river inference"
date
echo "JOB=${SLURM_JOB_ID:-local}"
echo "HOST=$(hostname)"
echo "HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "RUN_DIR=$RUN_DIR"
echo "CKPT=$CKPT"
echo "TILE_ROOT=$TILE_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "RIVERS=$RIVERS"
echo "MODEL_BRANCH=D001c AnyVisiblePatch NormOnly / normalized objective"
echo "FULLRIVER_BRANCH=E001c AnyVisiblePatch"
echo "FINAL_FOOTPRINT=Core_Loss_Mask_Pixel AND model prediction_patch_mask"
echo "SCIENTIFIC_FIX=exclude E001c newly-visible patches from reconstructed output"
echo "OVERLAP=exact georeferenced pixel averaging"
echo "============================================================"
nvidia-smi || true

python -u "$SCRIPT" \
  --code_dir "$CODE" \
  --ckpt "$CKPT" \
  --tile_root "$TILE_ROOT" \
  --output_dir "$OUT_DIR" \
  --rivers "${RIVER_ARRAY[@]}" \
  --batch_size "$BATCH_SIZE" \
  --device cuda \
  --tile_norm_std_scale "$STD_SCALE" \
  --bottleneck_norm inst1d \
  --loss_region_mode core \
  --core_patch_radius 3 \
  --require_e001c_anyvisiblepatch_root \
  "${RESUME_ARGS[@]}" \
  "${AMP_ARGS[@]}"

echo "=== DONE F050 ==="
echo "$OUT_DIR"
date
