"""Development-only BP convergence and sensitivity analysis.

This tunes or diagnoses old Stage 4 data and MUST NOT be used to tune a decoder
after a fresh held-out protocol has been committed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

from .archive_eval import parse_flattened_dem
from .belief_propagation import BeliefPropagationDEMDecoder
from .recovery import SyndromeRecoveryDecoder


def evaluate_variants(detectors: int, observables: int, faults,
                      syndromes: np.ndarray, truth: np.ndarray,
                      *, dampings=(0.0, 0.3, 0.6), iterations=(10, 20, 40)) -> list[dict]:
    if not 0 <= len(syndromes) or truth.shape != (len(syndromes), observables):
        raise ValueError('syndrome/truth shape mismatch')
    results = []
    for damping, limit in product(dampings, iterations):
        decoder = BeliefPropagationDEMDecoder(detectors, observables, faults,
                                              iterations=int(limit), damping=float(damping))
        prediction, _, details = decoder.decode_batch(syndromes, with_details=True)
        recovery = SyndromeRecoveryDecoder(detectors, observables, faults,
                                          iterations=int(limit), damping=float(damping))
        recovered, recovery_info = recovery.decode_batch(syndromes, with_details=True)
        results.append({
            'damping': float(damping), 'max_iterations': int(limit),
            'shots': len(syndromes), 'logical_errors': int(np.any(prediction != truth, axis=1).sum()),
            'logical_failure_rate': float(np.mean(np.any(prediction != truth, axis=1))),
            'fraction_converged': details['fraction_converged'],
            'mean_iterations': details['mean_iterations'],
            'fraction_hard_fault_syndrome_consistent':
                details['fraction_hard_fault_syndrome_consistent'],
            'unique_syndromes': details['unique_syndromes'],
            'recovery_logical_errors': int(np.any(recovered != truth, axis=1).sum()),
            'fraction_recovery_syndrome_consistent': recovery_info['fraction_syndrome_consistent'],
            'fraction_recovery_bp_converged': recovery_info['fraction_bp_converged'],
        })
    return results


def run(archive_root: Path, output: Path, *, shots: int = 48, limit: int = 4,
        dampings=(0.0, 0.3, 0.6), iterations=(10, 20, 40),
        case_ids: tuple[str, ...] = ()) -> dict:
    if shots <= 0 or limit < 0:
        raise ValueError('shots must be positive; limit must be nonnegative')
    candidates = sorted((archive_root / 'runs').glob('*.json'))
    if case_ids:
        candidates = [p for p in candidates if p.stem in case_ids]
    if limit:
        candidates = candidates[:limit]
    if not candidates:
        raise ValueError('no archived cases found')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, case in enumerate(candidates, 1):
        metadata = json.loads(case.read_text())
        raw_path, dem_path = case.with_suffix('.npz'), case.with_suffix('.dem')
        hash_ = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        expected = metadata.get('model', {}).get('raw_shots_sha256')
        if expected and expected != hash_:
            raise ValueError('archived raw-shot hash mismatch')
        with np.load(raw_path, allow_pickle=False) as arr:
            syn = arr['detection_events'][:shots].copy()
            truth = arr['actual_observables'][:shots].copy()
        faults = parse_flattened_dem(dem_path, syn.shape[1], truth.shape[1])
        variants = evaluate_variants(syn.shape[1], truth.shape[1], faults,
                                     syn, truth, dampings=dampings, iterations=iterations)
        rows.extend([{'case_id': case.stem, **v} for v in variants])
        print(f'BP sensitivity {i}/{len(candidates)}: {case.stem}', flush=True)
    result = {
        'status': 'PASS', 'cases': len(candidates), 'variants_per_case': len(dampings)*len(iterations),
        'rows': rows,
        'scope': 'DEVELOPMENT ONLY: recorded Stage 4 samples; not blind or held-out validation.',
        'limitations': 'LLR fixed-point convergence does not imply accurate logical parity or syndrome consistency.',
    }
    (output / 'bp_convergence.json').write_text(json.dumps(result, indent=2) + '\n')
    if rows:
        with (output / 'bp_convergence.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root', type=Path, default=Path('outputs/stage4'))
    p.add_argument('--output', type=Path, default=Path('outputs/bp_convergence'))
    p.add_argument('--shots', type=int, default=48)
    p.add_argument('--limit', type=int, default=4)
    p.add_argument('--case-id', action='append', default=[])
    p.add_argument('--dampings', nargs='+', type=float, default=[0, .3, .6])
    p.add_argument('--iterations', nargs='+', type=int, default=[10, 20, 40])
    args = p.parse_args(argv)
    print(run(args.archive_root, args.output, shots=args.shots, limit=args.limit,
              case_ids=tuple(args.case_id), dampings=args.dampings,
              iterations=args.iterations)['scope'])


if __name__ == '__main__':
    main()
