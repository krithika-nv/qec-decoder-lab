#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hybrid_heldout}" \
ZIP_PATH="${ZIP_PATH:-qec-hybrid-heldout-evidence.zip}" \
PYTHON="${PYTHON:-python3}" \
bash scripts/shell/PACK_VALIDATION_EVIDENCE.sh
