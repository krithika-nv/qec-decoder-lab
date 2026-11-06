#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
: "${ARCHIVE_ROOT:?Set ARCHIVE_ROOT to the extracted previously reviewed heldout/ archive}"
"${PYTHON:-python3}" -m qec_decoder_lab.ablation \
  --archive-root "$ARCHIVE_ROOT" \
  --output "${OUTPUT_DIR:-outputs/performance_development/ablation}" \
  --shots "${SHOTS:-32}"
