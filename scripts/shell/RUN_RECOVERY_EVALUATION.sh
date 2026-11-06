#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-outputs/stage4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/recovery_eval}"
set -- --archive-root "$ARCHIVE_ROOT" --output "$OUTPUT_DIR" \
  --shots "${SHOTS:-256}" --iterations "${BP_ITERATIONS:-20}" \
  --projected-detectors "${PROJECTED_DETECTORS:-4}"
if [[ "${LIMIT:-0}" != "0" ]]; then set -- "$@" --limit "$LIMIT"; fi
mkdir -p "$OUTPUT_DIR"
"${PYTHON:-python3}" -u -m qec_decoder_lab.archive_eval "$@" 2>&1 | tee "$OUTPUT_DIR/benchmark.log"
