#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Do not expand empty Bash arrays under `set -u`: macOS ships Bash 3.2,
# where even an initialized empty array produces an "unbound variable" error.
read -r -a DISTANCE_ARRAY <<< "${DISTANCES:-5 7 9 13 17 23 29 39 50}"
if (( ${#DISTANCE_ARRAY[@]} == 0 )); then
  echo "DISTANCES must contain at least one distance" >&2
  exit 2
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage5_paper}"
mkdir -p "$OUTPUT_DIR"
{
  set -- --shots "${SHOTS:-10000}" --distances "${DISTANCE_ARRAY[@]}" \
    --repeats "${REPEATS:-7}" --seed "${SEED:-20261103}" --output-dir "$OUTPUT_DIR"
  if [[ "${RESUME:-0}" == "1" ]]; then set -- "$@" --resume; fi
  "${PYTHON:-python3}" -u scripts/python/reproduce_pymatching_timing.py "$@"
  set -- "$OUTPUT_DIR"
  if [[ "${STRICT_ALL_NINE:-0}" == "1" ]]; then set -- "$@" --all-nine; fi
  if [[ "${REDO_DECODE:-0}" == "1" ]]; then set -- "$@" --redo-decode; fi
  if [[ "${REPLAY_SAMPLING:-0}" == "1" ]]; then set -- "$@" --replay; fi
  "${PYTHON:-python3}" -u -m qec_decoder_lab.stage5_audit "$@"
} 2>&1 | tee "$OUTPUT_DIR/benchmark.log"
