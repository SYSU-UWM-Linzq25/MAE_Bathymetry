#!/usr/bin/env bash
# Shared runtime setup for the Slurm entry points.
#
# Do not derive CODE_ROOT from BASH_SOURCE inside an sbatch job. Slurm executes a
# copied script under /var/spool/slurmd, so BASH_SOURCE points at the spool copy
# rather than at this project.

conus_setup_paths() {
  local requested_root=${1:-${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}}

  if [[ ! -d "$requested_root" ]]; then
    echo "[runtime] project directory does not exist: $requested_root" >&2
    exit 2
  fi

  CODE_ROOT=$(cd "$requested_root" && pwd -P)
  SCRIPT_DIR="$CODE_ROOT/scripts"
  if [[ ! -d "$SCRIPT_DIR" || ! -f "$SCRIPT_DIR/C080_slurm_runtime.sh" ]]; then
    echo "[runtime] cannot find project scripts under: $CODE_ROOT" >&2
    echo "[runtime] submit from the package root, or export CODE_ROOT=/absolute/package/path" >&2
    exit 2
  fi

  cd "$CODE_ROOT"
  mkdir -p "$CODE_ROOT/logs"
  export CODE_ROOT SCRIPT_DIR

  echo "[runtime] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-not-set}"
  echo "[runtime] CODE_ROOT=$CODE_ROOT"
  echo "[runtime] HOST=$(hostname) JOB_ID=${SLURM_JOB_ID:-not-slurm}"
}

conus_activate_python() {
  local env_name=${CONDA_ENV:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/conus_sampling_gdal}
  local conda_init=${CONDA_SH:-/home/uwm/zequnlin/miniconda3/etc/profile.d/conda.sh}
  local conda_base

  # Optional cluster module, for installations where conda is exposed by a module.
  if [[ -n "${CONDA_MODULE:-}" ]]; then
    if ! type module >/dev/null 2>&1; then
      echo "[runtime] CONDA_MODULE is set, but the module command is unavailable" >&2
      exit 2
    fi
    module load "$CONDA_MODULE"
  fi

  if [[ -z "$conda_init" && -n "${CONDA_EXE:-}" ]]; then
    conda_base=$(cd "$(dirname "$CONDA_EXE")/.." && pwd -P)
    conda_init="$conda_base/etc/profile.d/conda.sh"
  fi
  if [[ -z "$conda_init" ]] && command -v conda >/dev/null 2>&1; then
    conda_base=$(conda info --base)
    conda_init="$conda_base/etc/profile.d/conda.sh"
  fi
  if [[ -z "$conda_init" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    conda_init="$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
  if [[ -z "$conda_init" && -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    conda_init="$HOME/anaconda3/etc/profile.d/conda.sh"
  fi

  if [[ -z "$conda_init" || ! -f "$conda_init" ]]; then
    echo "[runtime] cannot locate conda.sh" >&2
    echo "[runtime] set CONDA_SH=/absolute/path/to/etc/profile.d/conda.sh" >&2
    echo "[runtime] CONDA_ENV may be an environment name or an absolute environment path" >&2
    exit 2
  fi

  # Some conda releases reference unset shell variables during activation.
  set +u
  # shellcheck disable=SC1090
  source "$conda_init"
  conda activate "$env_name"
  set -u

  PYTHON_BIN=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "[runtime] Python was not found after conda activation: $env_name" >&2
    exit 2
  fi
  export PYTHON_BIN

  echo "[runtime] CONDA_ENV=$env_name"
  echo "[runtime] PYTHON=$PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY'
import importlib
import sys

required = (
    "requests",
    "numpy",
    "rasterio",
    "osgeo.gdal",
    "osgeo.ogr",
    "osgeo.osr",
)
failures = []
for module_name in required:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # Also report binary/ABI import failures.
        failures.append((module_name, type(exc).__name__, str(exc)))

if failures:
    print("[runtime] Python dependency check failed:", file=sys.stderr)
    for module_name, error_type, message in failures:
        print(f"  - {module_name}: {error_type}: {message}", file=sys.stderr)
    print(
        "[runtime] Required conda packages: requests numpy rasterio gdal",
        file=sys.stderr,
    )
    raise SystemExit(2)

from osgeo import gdal

print(f"[runtime] Python={sys.version.split()[0]} GDAL={gdal.VersionInfo()}")
PY
}
