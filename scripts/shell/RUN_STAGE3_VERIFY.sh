#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
mkdir -p outputs/stage3
# Fail loudly if integrations are missing, never silently count test skips as passes.
"$PYTHON" -c "import stim, pymatching, pennylane; print('Quantum dependencies loaded')"
"$PYTHON" -m pytest -q | tee outputs/stage3/pytest.txt
"$PYTHON" -m qec_decoder_lab.validate_bp --output outputs/stage3/bp_synthetic_exact_check.json
"$PYTHON" -m qec_decoder_lab.pennylane_check --shots 3000 --output outputs/stage3/pennylane_check.json
# Small-code full exact inference is exercised, with saved circuit, DEM and shot archive.
"$PYTHON" -m qec_decoder_lab run --distance 3 --rounds 1 --p 0.01 \
  --shots "${SHOTS:-400}" --seed 20261103 --bayesian on --approximate on \
  --output outputs/stage3/exact_small.json
# Two bases of zero-noise circuits establish a hard invariant.
for basis in x z; do
  "$PYTHON" -m qec_decoder_lab run --distance 3 --rounds 3 --p 0 --shots 100 \
    --seed 20261103 --basis "$basis" --bayesian off \
    --output "outputs/stage3/noiseless_${basis}.json"
done
"$PYTHON" -m qec_decoder_lab.studies verify \
  outputs/stage3/exact_small.json outputs/stage3/noiseless_x.json outputs/stage3/noiseless_z.json \
  --replay --output outputs/stage3/verification.json
