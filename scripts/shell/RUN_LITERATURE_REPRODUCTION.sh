#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python3}" scripts/python/reproduce_pymatching_timing.py \
  --shots "${SHOTS:-10000}" --distances 5 7 9 --output-dir outputs/literature
