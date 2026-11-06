#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python3}" -m qec_decoder_lab run \
  --distance 3 --rounds 1 --p 0.01 --shots "${SHOTS:-2000}" \
  --seed 20261103 --bayesian on --approximate on \
  --bp-iterations 30 --bp-damping 0.3 \
  --output outputs/bp_exact_crosscheck.json
