#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python3}" -m qec_decoder_lab sweep \
  --distances 3 5 7 --ps 0.001 0.005 --shots "${SHOTS:-200}" \
  --seed 20261103 --bayesian auto --approximate on \
  --bp-iterations "${BP_ITERATIONS:-25}" \
  --output-dir outputs/bp_scaling --plot
