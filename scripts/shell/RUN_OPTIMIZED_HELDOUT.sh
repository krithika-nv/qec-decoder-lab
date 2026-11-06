#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# Fresh prospective post-optimization study. This output must never be populated
# with previously inspected Stage 4 or pre-optimization held-out samples.
if [[ "${OUTPUT_DIR:-outputs/optimized_heldout}" == 'outputs/heldout' ]]; then
  echo 'Refusing to overwrite previous held-out evidence. Choose a new OUTPUT_DIR.' >&2
  exit 2
fi
MODE="${MODE:-expanded}" \
SHOTS="${SHOTS:-25000}" \
PAIRED_SHOTS="${PAIRED_SHOTS:-4096}" \
CHECK_UPDATE=vectorized \
OUTPUT_DIR="${OUTPUT_DIR:-outputs/optimized_heldout}" \
PYTHON="${PYTHON:-python3}" \
bash scripts/shell/RUN_HELDOUT_VALIDATION.sh
