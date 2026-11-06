#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m qec_decoder_lab sweep \
  --distances 3 5 7 --ps 0.001 0.003 0.005 0.01 \
  --shots "${SHOTS:-10000}" --seed 20261101 \
  --basis x --profile circuit_level --bayesian auto \
  --output-dir outputs/baseline --plot
