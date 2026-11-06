#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python3}" -m qec_decoder_lab.exact_oracle_suite \
  --random-models "${RANDOM_MODELS:-24}" \
  --output "${OUTPUT_DIR:-outputs/exact_oracle}"
