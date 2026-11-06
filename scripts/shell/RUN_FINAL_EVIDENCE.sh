#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p outputs/final_evidence
{
  echo "=== STAGE 3: SCIENTIFIC INVARIANTS ==="
  bash scripts/shell/RUN_STAGE3_VERIFY.sh
  echo "=== STAGE 4: MATCHED SHOT COMPARISON ==="
  MODE=full bash scripts/shell/RUN_STAGE4_COMPARISON.sh
  echo "=== STAGE 5: ALL NINE PAPER DISTANCES ==="
  STRICT_ALL_NINE=1 REDO_DECODE=1 bash scripts/shell/RUN_STAGE5_PAPER.sh
  echo "=== EVIDENCE COMPLETE (CHECK EACH MANIFEST FOR SCIENTIFIC LIMITS) ==="
} 2>&1 | tee outputs/final_evidence/master_execution.log
