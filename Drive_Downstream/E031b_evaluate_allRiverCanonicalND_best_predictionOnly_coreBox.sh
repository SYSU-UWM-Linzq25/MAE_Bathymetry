#!/usr/bin/env bash
#SBATCH -J e031b_allND_best
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/E031b_eval_allND_best_predOnly_%j.out
#SBATCH -e /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy/logs/E031b_eval_allND_best_predOnly_%j.err
#SBATCH --chdir=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography

set -euo pipefail

ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography
WORK=$ROOT/Downstream_Task_Bathy
GENERIC=$WORK/script/E031_eval_stage4_best_val_predictionOnly_coreBox.sh
VAL_RIVER=OR_SantiamRiverTB_Topobathy_1_D23

[[ -x "$GENERIC" ]] || {
  echo "[ERROR] Missing or non-executable generic evaluator: $GENERIC" >&2
  exit 2
}

# Run the generic evaluator inside this already allocated Slurm job.
# The generic file contains SBATCH headers, but bash ignores those comments.
export DATA_FIX_TAG=${DATA_FIX_TAG:-allRiverCanonicalND}
export GPU_ID=${GPU_ID:-0}
export OVERWRITE_EVAL=${OVERWRITE_EVAL:-0}

bash "$GENERIC" "$VAL_RIVER"
