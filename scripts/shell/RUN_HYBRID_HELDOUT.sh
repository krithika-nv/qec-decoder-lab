#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hybrid_heldout}"
if [[ "$OUTPUT_DIR" == 'outputs/optimized_heldout' || "$OUTPUT_DIR" == 'outputs/heldout' ]]; then
  echo 'Use a NEW output directory; never mix archived held-out results.' >&2
  exit 2
fi
# Default uses the widely compatible NumPy engine. Optional JIT use requires
# a Python interpreter with a compatible Numba wheel and a separate frozen run.
OUTPUT_DIR="$OUTPUT_DIR" \
MODE="${MODE:-expanded}" \
SHOTS="${SHOTS:-25000}" \
PAIRED_SHOTS="${PAIRED_SHOTS:-4096}" \
CHECK_UPDATE="${CHECK_UPDATE:-vectorized}" \
BASIS_UPDATE="${BASIS_UPDATE:-python}" \
BP_ITERATIONS="${BP_ITERATIONS:-20}" \
DECODER_STRATEGY=hybrid \
HYBRID_GATE_FRACTION="${HYBRID_GATE_FRACTION:-0.10}" \
PYTHON="${PYTHON:-python3}" \
bash scripts/shell/RUN_HELDOUT_VALIDATION.sh
