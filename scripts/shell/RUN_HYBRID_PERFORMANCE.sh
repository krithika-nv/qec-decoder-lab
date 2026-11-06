#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# Same-worker timing includes MWPM on every shot and optional recovery only
# for selected shots. Construction and JIT warmup are excluded from timings.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hybrid_heldout}" \
TIMING_SHOTS="${TIMING_SHOTS:-256}" \
REPEATS="${REPEATS:-5}" \
PYTHON="${PYTHON:-python3}" \
bash scripts/shell/RUN_PERFORMANCE.sh
