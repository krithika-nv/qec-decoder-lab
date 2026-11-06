"""Independent PennyLane density-matrix vs Stim sample parity-check validation.

A single Z_0 Z_1 ancilla extraction is NOT a full surface code or logical QEC.
The noise is exactly the symmetric 1q depolarization channel before extraction.
The expected ancilla parity-flip probability is 2p/3, not p.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def validate_zz_check(*, p: float = .09, shots: int = 20000, seed: int = 20261103) -> dict:
    import pennylane as qml
    import stim

    if not 0 <= p <= 1 or shots < 100:
        raise ValueError('p must be in [0,1] and shots >= 100')
    dev = qml.device('default.mixed', wires=3, shots=None)

    @qml.qnode(dev)
    def parity_circuit(a: int, b: int):
        if a:
            qml.PauliX(wires=0)
        if b:
            qml.PauliX(wires=1)
        qml.DepolarizingChannel(p, wires=0)
        qml.CNOT(wires=[0, 2])
        qml.CNOT(wires=[1, 2])
        return qml.probs(wires=[2])

    records = []
    for a in (0, 1):
        for b in (0, 1):
            observed = np.asarray(parity_circuit(a, b), dtype=float)
            expected_flip = 2.0 * p / 3.0
            parity = a ^ b
            theoretical_odd = expected_flip if parity == 0 else 1 - expected_flip
            # Independently construct and sample matching Stim stabilizer circuit.
            stim_source = 'R 0 1 2\n'
            if a:
                stim_source += 'X 0\n'
            if b:
                stim_source += 'X 1\n'
            stim_source += f'DEPOLARIZE1({p}) 0\nCX 0 2\nCX 1 2\nM 2\n'
            counts = stim.Circuit(stim_source).compile_sampler(seed=seed+10*a+b).sample(shots)
            empirical_odd = float(np.mean(counts[:, 0]))
            tolerance = 6 * np.sqrt(theoretical_odd*(1-theoretical_odd)/shots) + 1/shots
            qml_pass = bool(np.isclose(observed[1], theoretical_odd, atol=1e-8, rtol=0))
            stim_pass = bool(abs(empirical_odd - theoretical_odd) <= tolerance)
            records.append({
                'data_bits': [a, b], 'ideal_parity': parity,
                'expected_odd_probability': theoretical_odd,
                'pennylane_odd_probability': float(observed[1]),
                'stim_sampled_odd_probability': empirical_odd,
                'stim_tolerance': float(tolerance),
                'pennylane_matches_analytical': qml_pass,
                'stim_matches_analytical_with_6sigma_bound': stim_pass,
            })
    result = {
        'schema_version': 1,
        'status': 'passed' if all(r['pennylane_matches_analytical'] and
                                  r['stim_matches_analytical_with_6sigma_bound']
                                  for r in records) else 'failed',
        'protocol': 'one_Z0Z1_ancilla_check_not_surface_code',
        'noise': 'single_qubit_depolarizing_channel_before_two_CNOTs',
        'p': p, 'shots': shots, 'seed': seed, 'cases': records,
        'versions': {'pennylane': qml.__version__, 'stim': stim.__version__},
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_limit': 'Does not establish equivalence of full surface-code schedules or decoders',
    }
    if result['status'] != 'passed':
        raise AssertionError('Stim/PennyLane parity-check validation failed: ' + repr(records))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=float, default=.09)
    parser.add_argument('--shots', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--output', type=Path, default=Path('outputs/pennylane/check_extraction.json'))
    a = parser.parse_args()
    result = validate_zz_check(p=a.p, shots=a.shots, seed=a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"Wrote {a.output} ({result['status']})")


if __name__ == '__main__':
    main()
