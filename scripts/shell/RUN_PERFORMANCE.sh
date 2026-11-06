#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
DIR="${OUTPUT_DIR:-outputs/heldout}"
"$PYTHON" -u -m qec_decoder_lab.performance run --root "$DIR" \
  --shots "${TIMING_SHOTS:-64}" --repetitions "${REPEATS:-5}" 2>&1 | tee "$DIR/performance_execution.log"
"$PYTHON" -m qec_decoder_lab.performance audit --root "$DIR"
