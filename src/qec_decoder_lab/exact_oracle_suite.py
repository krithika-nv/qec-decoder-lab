"""Independent-error exact logical-MAP oracle on genuinely small models.

No surface-code optimality is claimed. Comparing weighted *logical* error risk
keeps the oracle separate from minimum-prior-energy fault recovery.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .belief_propagation import BeliefPropagationDEMDecoder
from .exact import ExactBayesianDEMDecoder, Fault
from .recovery import SyndromeRecoveryDecoder


def evaluate_small_model(detectors: int, observables: int, faults: list[Fault],
                         *, iterations: int = 20, damping: float = 0.3) -> dict:
    if detectors + observables > 12:
        raise ValueError('oracle suite refuses more than 12 joint bits')
    exact = ExactBayesianDEMDecoder(detectors, observables, faults, max_bits=12)
    reachable = np.flatnonzero(exact.evidence > 1e-15)
    if not len(reachable):
        raise ValueError('no reachable syndrome')
    syndromes = np.array([[(int(s) >> j) & 1 for j in range(detectors)]
                          for s in reachable], dtype=np.uint8)
    oracle = exact.decode_batch(syndromes)
    bp = BeliefPropagationDEMDecoder(detectors, observables, faults,
                                     iterations=iterations, damping=damping)
    recovery = SyndromeRecoveryDecoder(detectors, observables, faults,
                                       iterations=iterations, damping=damping)
    bpp = bp.decode_batch(syndromes)
    rcp = recovery.decode_batch(syndromes)
    weights = exact.evidence[reachable]
    weights = weights / weights.sum()

    def metrics(prediction: np.ndarray) -> dict:
        frames = (prediction.astype(np.int64) *
                  (1 << np.arange(observables, dtype=np.int64))).sum(axis=1)
        # The joint P(syndrome, true logical frame) is ground truth.
        success = float(np.sum(exact.joint[frames, reachable]))
        agreement = float(np.sum(weights * np.all(prediction == oracle, axis=1)))
        return {'exact_probability_of_correct_frame': success,
                'exact_expected_logical_failure': float(1.0 - success),
                'evidence_weighted_MAP_agreement': agreement}

    optimal = metrics(oracle)
    bp_metrics = metrics(bpp)
    recovery_metrics = metrics(rcp)
    for other in (bp_metrics, recovery_metrics):
        other['excess_expected_failure_over_oracle'] = max(
            0.0, other['exact_expected_logical_failure'] -
            optimal['exact_expected_logical_failure'])
    return {
        'detectors': detectors, 'observables': observables,
        'fault_mechanisms': len(faults), 'reachable_syndromes': len(reachable),
        'model': [{'probability': float(f.probability), 'mask': int(f.mask)} for f in faults],
        'exact_logical_MAP': optimal, 'original_loopy_bp': bp_metrics,
        'syndrome_recovery': recovery_metrics,
        'scope': 'exact independent-error DEM on small models; not full physical surface-code MAP',
    }


def fixture_models(seed: int = 8301, random_models: int = 24):
    # Hand-authored regressions for parity ambiguity, logical-only faults,
    # correlated syndrome/logical mechanisms, and redundant checks.
    yield 'single_fault', 2, 1, [Fault(0.13, 1 | (1 << 2)),
                                  Fault(0.08, 2), Fault(0.03, 1 | 2)]
    yield 'logical_only', 2, 1, [Fault(0.09, 1), Fault(0.17, 2), Fault(0.21, 1 << 2)]
    yield 'shared_syndrome', 3, 1, [Fault(0.05, 1 | (1 << 3)),
                                    Fault(0.09, 1), Fault(0.07, 2 | 4), Fault(0.12, 3)]
    yield 'redundant_faults', 3, 1, [Fault(0.12, 1 | 2), Fault(0.15, 1 | 2 | 8),
                                     Fault(0.05, 4), Fault(0.11, 2), Fault(0.04, 1)]
    rng = np.random.default_rng(seed)
    for number in range(random_models):
        detectors = int(rng.integers(2, 6))
        observables = 1
        faults = [Fault(float(rng.choice([0.01, 0.06, 0.14, 0.28])), 1 << k)
                  for k in range(detectors)]
        for _ in range(int(rng.integers(4, 13))):
            mask = 0
            for k in rng.choice(detectors, size=int(rng.integers(1, min(4, detectors + 1))),
                                replace=False):
                mask ^= 1 << int(k)
            if rng.random() < 0.35:
                mask ^= 1 << detectors
            faults.append(Fault(float(rng.choice([0.01, 0.04, 0.1, 0.2, 0.38])), mask))
        yield f'random_{number:03d}', detectors, observables, faults


def run(output: Path, *, seed: int, count: int, iterations: int, damping: float) -> dict:
    if count < 0:
        raise ValueError('random model count must be nonnegative')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, detectors, observables, faults in fixture_models(seed, count):
        item = evaluate_small_model(detectors, observables, faults,
                                    iterations=iterations, damping=damping)
        item['case_id'] = name
        records.append(item)
    result = {
        'status': 'PASS', 'seed': seed, 'random_models': count,
        'iterations': iterations, 'damping': damping, 'cases': records,
        'warning': 'Designed and random small DEMs, not held-out surface-code experiments.',
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + '\n'
    (output / 'exact_oracle_results.json').write_text(payload)
    with (output / 'exact_oracle_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'case_id', 'detectors', 'fault_mechanisms', 'reachable_syndromes',
            'exact_failure', 'bp_failure', 'recovery_failure', 'bp_excess', 'recovery_excess',
            'bp_map_agreement', 'recovery_map_agreement'])
        writer.writeheader()
        for x in records:
            bp, rc, ex = x['original_loopy_bp'], x['syndrome_recovery'], x['exact_logical_MAP']
            writer.writerow({
                'case_id': x['case_id'], 'detectors': x['detectors'],
                'fault_mechanisms': x['fault_mechanisms'],
                'reachable_syndromes': x['reachable_syndromes'],
                'exact_failure': ex['exact_expected_logical_failure'],
                'bp_failure': bp['exact_expected_logical_failure'],
                'recovery_failure': rc['exact_expected_logical_failure'],
                'bp_excess': bp['excess_expected_failure_over_oracle'],
                'recovery_excess': rc['excess_expected_failure_over_oracle'],
                'bp_map_agreement': bp['evidence_weighted_MAP_agreement'],
                'recovery_map_agreement': rc['evidence_weighted_MAP_agreement'],
            })
    result['results_sha256'] = hashlib.sha256(payload.encode()).hexdigest()
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/exact_oracle'))
    p.add_argument('--seed', type=int, default=8301)
    p.add_argument('--random-models', type=int, default=24)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--damping', type=float, default=0.3)
    args = p.parse_args(argv)
    result = run(args.output, seed=args.seed, count=args.random_models,
                 iterations=args.iterations, damping=args.damping)
    print(f'Exact-oracle suite: {len(result["cases"])} models; reports in {args.output}')


if __name__ == '__main__':
    main()
