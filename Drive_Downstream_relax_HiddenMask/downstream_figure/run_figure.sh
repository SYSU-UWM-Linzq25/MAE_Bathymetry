#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MAE_ENV=${MAE_ENV:-/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/conda_envs/mae_zequn}
PYTHON_BIN=${PYTHON_BIN:-$MAE_ENV/bin/python}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: MAE Python was not found or is not executable:" >&2
  echo "  $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN to the correct interpreter and rerun." >&2
  exit 2
fi

# Use the same Python 3.12 environment as the MAE training/evaluation workflow.
# Calling its interpreter directly avoids silently falling back to an old
# system Python on Mortimer compute nodes.
export PATH="$MAE_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mae_bathymetry_matplotlib_${USER:-user}}
mkdir -p "$MPLCONFIGDIR"

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 9):
    raise SystemExit(
        "ERROR: Python >= 3.9 is required; actual version is "
        + sys.version.split()[0]
    )

try:
    import matplotlib
    import numpy
    import scipy
except ImportError as exc:
    raise SystemExit("ERROR: missing plotting dependency: {}".format(exc))

print("Python executable : {}".format(sys.executable))
print("Python version    : {}".format(sys.version.split()[0]))
print("NumPy             : {}".format(numpy.__version__))
print("SciPy             : {}".format(scipy.__version__))
print("Matplotlib        : {}".format(matplotlib.__version__))
PY

"$PYTHON_BIN" "$SCRIPT_DIR/plot_representative_bathymetry.py" \
  --data-dir "$SCRIPT_DIR/data/H054_AGU_SelectedReach_DataBundle" \
  --output-dir "$SCRIPT_DIR/output"
