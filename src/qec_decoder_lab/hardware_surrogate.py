"""Explicit, auditable **surrogate** mapping of calibration metrics to Stim's uniform channels.

This is not a calibrated QEC prediction or an IBM device circuit. In particular,
Stim's after_clifford_depolarization applies to all Clifford gates, whereas we
map a median measured 2q gate error to it as a stated modelling approximation.
The real coupling graph, 1q/2q heterogeneity, T1/T2 and crosstalk are not
represented. Missing reset/idle probabilities are *never silently guessed*.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .model import MemoryConfig


def _prob(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f'{label} must be numeric')
    x = float(value)
    if not np.isfinite(x) or not 0 <= x <= 1:
        raise ValueError(f'{label} must be a finite probability in [0, 1]')
    return x


def surrogate_from_calibration(
    path: Path, *, distance: int, rounds: int, basis: str = 'x',
) -> tuple[MemoryConfig, dict]:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get('schema_version') != 1:
        raise ValueError('calibration must declare schema_version=1')
    kind = data.get('source_kind')
    if kind not in ('measured_calibration_export', 'synthetic_example'):
        raise ValueError('source_kind must be measured_calibration_export or synthetic_example')
    if not isinstance(data.get('source_uri'), str) or not data['source_uri'].strip():
        raise ValueError('source_uri (traceable source identifier) required')
    if not isinstance(data.get('backend'), str) or not data['backend'].strip():
        raise ValueError('backend required')
    stamp = data.get('calibration_timestamp_utc')
    try:
        dt = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
        if dt.tzinfo is None or dt.utcoffset() != timezone.utc.utcoffset(dt):
            raise ValueError('timestamp must be UTC')
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('calibration_timestamp_utc must be ISO 8601 UTC') from exc
    qubits = data.get('qubits')
    gates = data.get('two_qubit_gates')
    assumptions = data.get('assumptions')
    if not isinstance(qubits, list) or not qubits or not isinstance(gates, list) or not gates:
        raise ValueError('qubits and two_qubit_gates must be nonempty arrays')
    if not isinstance(assumptions, dict):
        raise ValueError('assumptions including measured/assumed reset and idle probabilities required')
    readouts = [_prob(q['readout_error'], 'readout_error') for q in qubits]
    two_qubit = [_prob(g['error'], 'two_qubit_gates.error') for g in gates]
    idle = _prob(assumptions['before_round_data_depolarization'], 'idle assumption')
    reset = _prob(assumptions['after_reset_flip_probability'], 'reset assumption')
    gate_proxy = float(np.median(two_qubit))
    readout_proxy = float(np.median(readouts))
    cfg = MemoryConfig(
        distance=distance, rounds=rounds, basis=basis, noise_profile='custom',
        p=0.0, after_clifford_depolarization=gate_proxy,
        before_round_data_depolarization=idle,
        before_measure_flip_probability=readout_proxy,
        after_reset_flip_probability=reset,
    )
    provenance = {
        'status': 'surrogate_only_not_hardware_validated',
        'source_kind': kind, 'source_uri': data['source_uri'],
        'backend': data['backend'], 'calibration_timestamp_utc': stamp,
        'input_sha256': hashlib.sha256(raw).hexdigest(),
        'num_qubits_characterized': len(qubits), 'num_two_qubit_gates_characterized': len(gates),
        'readout_error_range': [float(min(readouts)), float(max(readouts))],
        'two_qubit_error_range': [float(min(two_qubit)), float(max(two_qubit))],
        'channel_mappings': {
            'after_clifford_depolarization': 'median(two_qubit_gates[*].error): proxy for all Clifford gates',
            'before_measure_flip_probability': 'median(qubits[*].readout_error): symmetric flip approximation',
            'before_round_data_depolarization': 'explicit user-provided assumption, not derived from T1/T2',
            'after_reset_flip_probability': 'explicit user-provided assumption, not inferred',
        },
        'limitations': [
            'No layout- or gate-specific calibration assignment',
            'Readout confusion and gate infidelity are not identical to symmetric Pauli errors',
            'No T1/T2, leakage, correlations, coherent errors, crosstalk, or calibration drift',
            'No match to physical schedule, hardware syndrome measurements, or held-out logical outcomes',
        ],
    }
    return cfg, provenance
