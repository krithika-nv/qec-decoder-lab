#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
set -- --root "${OUTPUT_DIR:-outputs/heldout}" --zip "${ZIP_PATH:-qec-heldout-validation-evidence.zip}"
if [[ "${WITHOUT_PERFORMANCE:-0}" == "1" ]]; then set -- "$@" --no-performance; fi
"${PYTHON:-python3}" -m qec_decoder_lab.validation_bundle "$@"
