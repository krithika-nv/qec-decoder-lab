#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python3}" -u -m qec_decoder_lab.convergence_study \
  --archive-root "${ARCHIVE_ROOT:-outputs/stage4}" \
  --output "${OUTPUT_DIR:-outputs/bp_convergence}" \
  --shots "${SHOTS:-48}" --limit "${LIMIT:-4}"
