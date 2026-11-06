#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m qec_decoder_lab run \
  --distance 3 --rounds 1 --p 0.01 --shots "${SHOTS:-10000}" \
  --basis x --profile circuit_level --seed 20261101 \
  --bayesian on --output outputs/bayesian_d3_r1.json
