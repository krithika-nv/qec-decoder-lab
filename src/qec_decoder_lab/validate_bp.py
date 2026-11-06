"""Enumerated synthetic DEM study: compare loopy BP to exact logical marginal MAP.

All detector syndromes are evaluated, weighted by their exact model evidence.
This is a model-level benchmark, *not* measured hardware or a surface code run.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .belief_propagation import BeliefPropagationDEMDecoder
from .exact import ExactBayesianDEMDecoder, Fault


def validation_report():
    scenarios = {
        'tree_single_check': {
            'detectors': 1,
            'faults': [Fault(.10, 0b11), Fault(.05, 0b01)],
        },
        'loopy_three_check': {
            'detectors': 3,
            'faults': [
                Fault(.055, 0b0011),    # D0 D1
                Fault(.045, 0b0110),    # D1 D2
                Fault(.045, 0b1101),    # D0 D2 L0
                Fault(.013, 0b1001),    # D0 L0
                Fault(.025, 0b0010),    # D1
                Fault(.020, 0b0100),    # D2
            ],
        },
    }
    output = {}
    for name, sc in scenarios.items():
        d, faults = sc['detectors'], sc['faults']
        exact = ExactBayesianDEMDecoder(d, 1, faults)
        bp = BeliefPropagationDEMDecoder(d, 1, faults, iterations=100, damping=.35)
        records = []
        weights, deviations, agreements, converged = [], [], [], []
        for syndrome_id in range(1 << d):
            y = np.array([(syndrome_id >> j) & 1 for j in range(d)], dtype=np.uint8)
            evidence = float(exact.evidence[syndrome_id])
            if evidence <= 0:
                continue
            posterior = float(exact.joint[1, syndrome_id] / evidence)
            result = bp.decode(y)
            estimate = float(result.logical_one_probabilities[0])
            exact_decision = int(exact.decode_batch(y[None, :])[0, 0])
            bp_decision = int(result.prediction[0])
            weights.append(evidence)
            deviations.append(abs(estimate - posterior))
            agreements.append(bp_decision == exact_decision)
            converged.append(result.converged)
            records.append({
                'detector_mask': int(syndrome_id), 'syndrome': y.tolist(),
                'exact_syndrome_evidence': evidence,
                'exact_p_logical_one': posterior, 'bp_p_logical_one': estimate,
                'exact_map': exact_decision, 'bp_prediction': bp_decision,
                'bp_converged': result.converged,
                'bp_iterations': result.iterations,
            })
        weights = np.array(weights)
        weights /= weights.sum()
        output[name] = {
            'detectors': d, 'observables': 1, 'faults': [f.__dict__ for f in faults],
            'num_syndromes_with_evidence': len(records),
            'exact_evidence_normalized': bool(np.isclose(sum(exact.evidence), 1.0)),
            'weighted_mean_absolute_posterior_error': float(np.dot(weights, deviations)),
            'weighted_exact_map_agreement': float(np.dot(weights, agreements)),
            'weighted_bp_convergence_rate': float(np.dot(weights, converged)),
            'per_syndrome': records,
        }
    return {
        'schema_version': 1, 'model': 'synthetic_independent_DEM_not_surface_code',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'metrics': output,
        'limitations': [
            'Exact oracle is conditioned on independent synthetic DEM, not measured data',
            'Loops mean BP and independent fault logical marginals are not exact',
            'Numerical closeness here is not proof of practical advantage over MWPM',
        ],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/validation/bp_exact_synthetic.json'))
    args = parser.parse_args()
    report = validation_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, item in report['metrics'].items():
        print(f"{name}: weighted posterior MAE={item['weighted_mean_absolute_posterior_error']:.6g}, "
              f"weighted MAP agreement={item['weighted_exact_map_agreement']:.5f}")
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
