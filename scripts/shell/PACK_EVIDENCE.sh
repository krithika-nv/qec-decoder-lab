#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ARCHIVE="qec-stages-3-5-evidence.zip"
set --
if [[ "${STAGE5_ONLY:-0}" == "1" ]]; then
  set -- --stage5-only
  ARCHIVE="qec-stage5-full-results.zip"
  "${PYTHON:-python3}" -m qec_decoder_lab.evidence_gate "$@"
  zip -q -r "$ARCHIVE" outputs/stage5_paper outputs/final_evidence
else
  "${PYTHON:-python3}" -m qec_decoder_lab.evidence_gate "$@"
  zip -q -r "$ARCHIVE" outputs/stage3 outputs/stage4 outputs/stage5_paper outputs/final_evidence
fi
echo "Created $ARCHIVE. Attach this ZIP for scientific review."
