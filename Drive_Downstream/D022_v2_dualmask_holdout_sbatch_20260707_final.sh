#!/usr/bin/env bash
set -euo pipefail

# D022 v2 final: submit one MAE v2 dual-mask holdout fold to Slurm.
#
# It submits:
#   D021_v2_dualmask_holdout_onefold_runner_20260707_final.sh

ROOT=${ROOT:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography}
WORK=${WORK:-$ROOT/Downstream_Task_Bathy}
SCRIPT=${SCRIPT:-$WORK/script/D021_v2_dualmask_holdout_onefold_runner_20260707_final.sh}

PARTITION=${PARTITION:-HydroIntel}
NODE=${NODE:-execute-4006}
LOG_DIR=${LOG_DIR:-$WORK/cross_validation_v2/logs}

HOLDOUT_PRESET=${HOLDOUT_PRESET:-CO}
EPOCHS=${EPOCHS:-400}
BATCH_SIZE=${BATCH_SIZE:-4}
ACCUM_ITER=${ACCUM_ITER:-4}
GPU_ID=${GPU_ID:-0}
NUM_WORKERS=${NUM_WORKERS:-1}
PATIENCE=${PATIENCE:-60}

# Slurm resources. Keep conservative defaults for one frozen-encoder fold.
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-48G}
TIME_LIMIT=${TIME_LIMIT:-7-00:00:00}

if [[ -z "${RUN_STAGE:-}" ]]; then
  if [[ "$EPOCHS" -le 5 ]]; then
    RUN_STAGE="smoke"
  else
    RUN_STAGE="train"
  fi
fi

JOB_NAME=${JOB_NAME:-${RUN_STAGE}_holdout_${HOLDOUT_PRESET}_v2_e${EPOCHS}_b${BATCH_SIZE}_acc${ACCUM_ITER}}

mkdir -p "$LOG_DIR"

echo "Submitting job:"
echo "  JOB_NAME=$JOB_NAME"
echo "  HOLDOUT_PRESET=$HOLDOUT_PRESET"
echo "  RUN_STAGE=$RUN_STAGE"
echo "  EPOCHS=$EPOCHS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  ACCUM_ITER=$ACCUM_ITER"
echo "  GPU_ID=$GPU_ID"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo "  PATIENCE=$PATIENCE"
echo "  PARTITION=$PARTITION"
echo "  NODE=$NODE"
echo "  CPUS_PER_TASK=$CPUS_PER_TASK"
echo "  MEM=$MEM"
echo "  TIME_LIMIT=$TIME_LIMIT"
echo "  SCRIPT=$SCRIPT"
echo "  LOG_DIR=$LOG_DIR"

sbatch \
  -p "$PARTITION" \
  -w "$NODE" \
  -J "$JOB_NAME" \
  -n 1 \
  -c "$CPUS_PER_TASK" \
  --mem="$MEM" \
  -t "$TIME_LIMIT" \
  -o "$LOG_DIR/${JOB_NAME}_%j.out" \
  -e "$LOG_DIR/${JOB_NAME}_%j.err" \
  --export=ALL \
  "$SCRIPT"
