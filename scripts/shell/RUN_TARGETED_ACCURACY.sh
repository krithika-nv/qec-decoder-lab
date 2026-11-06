#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# Separate, locked large-sample study. Keep different from primary held-out data.
MODE="${MODE:-targeted}" OUTPUT_DIR="${OUTPUT_DIR:-outputs/large_sample_study}" \
SHOTS="${SHOTS:-25000}" PAIRED_SHOTS="${PAIRED_SHOTS:-8192}" \
PYTHON="${PYTHON:-python3}" bash scripts/shell/RUN_HELDOUT_VALIDATION.sh
