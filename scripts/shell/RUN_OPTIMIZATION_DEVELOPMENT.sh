#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
if [[ -z "${ARCHIVE_ROOT:-}" ]]; then
  echo "Set ARCHIVE_ROOT to extracted original heldout evidence directory (containing cases/)." >&2
  exit 2
fi
"$PYTHON" -m qec_decoder_lab.optimization_study run \
  --archive-root "$ARCHIVE_ROOT" --output "${OUTPUT_DIR:-outputs/optimization_development}" \
  --shots "${SHOTS:-64}" --repeats "${REPEATS:-5}" --profile-shots "${PROFILE_SHOTS:-16}"
