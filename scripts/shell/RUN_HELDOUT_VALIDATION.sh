#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
DIR="${OUTPUT_DIR:-outputs/heldout}"
if [[ ! -f "$DIR/locked_protocol.json" ]]; then
  set -- plan --root "$DIR" --profile "${MODE:-targeted}" \
    --shots "${SHOTS:-10000}" --paired-shots "${PAIRED_SHOTS:-2048}" \
    --iterations "${BP_ITERATIONS:-20}" --damping "${BP_DAMPING:-0.3}" \
    --refinement-budget "${REFINEMENT_BUDGET:-64}" \
    --check-update "${CHECK_UPDATE:-vectorized}" \
    --basis-update "${BASIS_UPDATE:-python}" \
    --strategy "${DECODER_STRATEGY:-recovery}"
  if [[ -n "${HYBRID_GATE_FRACTION:-}" ]]; then
    set -- "$@" --gate-fraction "$HYBRID_GATE_FRACTION"
  fi
  "$PYTHON" -m qec_decoder_lab.frozen_validation "$@"
fi
mkdir -p "$DIR"
"$PYTHON" -u -m qec_decoder_lab.frozen_validation run --root "$DIR" 2>&1 | tee -a "$DIR/heldout_execution.log"
set -- --root "$DIR"
if [[ "${REDO_MWPM:-1}" == "1" ]]; then set -- "$@" --redo-mwpm; fi
if [[ "${REDO_RECOVERY:-0}" == "1" ]]; then set -- "$@" --redo-recovery; fi
if [[ "${REPLAY_SEED:-0}" == "1" ]]; then set -- "$@" --replay-seed; fi
"$PYTHON" -m qec_decoder_lab.frozen_validation audit "$@"
