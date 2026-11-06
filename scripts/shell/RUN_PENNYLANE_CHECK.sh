#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python3}" -m qec_decoder_lab.pennylane_check \
  --p 0.09 --shots "${SHOTS:-20000}" --output outputs/pennylane/check_extraction.json
