#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python3}" -m qec_decoder_lab hardware-run \
  --calibration "${CALIBRATION:-data/examples/synthetic_calibration.json}" \
  --distance 3 --rounds 3 --shots "${SHOTS:-1000}" \
  --output outputs/hardware_surrogate.json
