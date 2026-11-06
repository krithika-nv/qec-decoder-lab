#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export QEC_PROGRESS=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
MODE="${MODE:-quick}"
if [[ "$MODE" == "quick" ]]; then
  # A short functional check; NOT a publishable decoder comparison.
  OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage4_quick}"
  set -- --distances 3 --ps 0.005 --bases x --profiles circuit_level \
    --shots "${SHOTS:-100}" --bp-shots "${BP_SHOTS:-8}" \
    --bp-iterations "${BP_ITERATIONS:-8}" \
    --timing-shots "${TIMING_SHOTS:-4}" --timing-repeats "${TIMING_REPEATS:-1}"
elif [[ "$MODE" == "full" ]]; then
  # A 72-case research study. BP is deliberately an unoptimized Python prototype.
  OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage4}"
  set -- --distances 3 5 7 --ps 0.001 0.003 0.005 0.01 --bases x z \
    --profiles circuit_level data_only measurement_only \
    --shots "${SHOTS:-2000}" --bp-shots "${BP_SHOTS:-256}" \
    --bp-iterations "${BP_ITERATIONS:-20}" \
    --timing-shots "${TIMING_SHOTS:-32}" --timing-repeats "${TIMING_REPEATS:-3}"
else
  echo "MODE must be quick or full (received '$MODE')" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
if [[ "${RESUME:-1}" == "1" ]]; then set -- "$@" --resume; fi
if [[ "${REPLAY:-1}" == "1" ]]; then set -- "$@" --replay; fi
if [[ "${REDO_MWPM:-1}" == "1" ]]; then set -- "$@" --redo-mwpm; fi
if [[ "${NO_PLOT:-0}" == "1" ]]; then set -- "$@" --no-plot; fi
if [[ "${RECOVERY:-0}" == "1" ]]; then set -- "$@" --recovery; fi
printf 'Stage 4 mode: %s | output: %s | progress follows case-by-case\n' "$MODE" "$OUTPUT_DIR"
"${PYTHON:-python3}" -u -m qec_decoder_lab.studies benchmark \
  "$@" --seed "${SEED:-20261103}" --approximate on \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee -a "$OUTPUT_DIR/benchmark.log"
