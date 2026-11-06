"""Development-only fault-recovery ablation using *previously inspected* shots.

Ablation modes are diagnostic and must never be described as prospective held-out
results. Each mode uses the same sampled syndromes and true observables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .optimization_study import _case_dir, _load_case
from .recovery import SyndromeRecoveryDecoder, _basis, _bits, _refine, _score, _solve

MODES = ('prior_raw', 'prior_refined', 'bp_guided_raw', 'bp_guided_refined', 'full')


def _logical_prediction(frame: int, masks: list[int], observables: int) -> np.ndarray:
    logical = 0
    for j in _bits(frame):
        logical ^= masks[j]
    return np.array([(logical >> k) & 1 for k in range(observables)], dtype=np.uint8)


class RecoveryAblator:
    """One model, decomposed to expose the price and marginal utility of BP."""

    def __init__(self, decoder: SyndromeRecoveryDecoder):
        self.decoder = decoder

    def decode_all(self, syndrome: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        d = self.decoder
        y = np.asarray(syndrome)
        if y.shape != (d.detectors,) or np.any((y != 0) & (y != 1)):
            raise ValueError('invalid detector syndrome')
        target = sum(int(b) << i for i, b in enumerate(y))
        t0 = perf_counter()
        prior_raw = _solve(d._prior_pivots, target)
        t1 = perf_counter()
        prior_refined = _refine(prior_raw, d._prior_pivots, d.det_masks,
                                d.prior_weights, d._prior_order, d.refinement_budget)
        t2 = perf_counter()
        result = d.bp.decode(y)
        t3 = perf_counter()
        q = result.fault_marginals
        hard = sum(1 << int(j) for j in np.flatnonzero(q > 0.5))
        hard_syndrome = 0
        for j in _bits(hard):
            hard_syndrome ^= d.det_masks[j]
        reliability = np.abs(np.log1p(-np.clip(q, 1e-12, 1 - 1e-12)) -
                             np.log(np.clip(q, 1e-12, 1 - 1e-12)))
        order = np.argsort(reliability, kind='stable')
        pivots = _basis(d.det_masks, order, target_rank=len(d._prior_pivots))
        guided_raw = hard ^ _solve(pivots, target ^ hard_syndrome)
        t4 = perf_counter()
        guided_refined = _refine(guided_raw, pivots, d.det_masks,
                                 d.prior_weights, order, d.refinement_budget)
        t5 = perf_counter()
        guided_wins = _score(guided_refined, d.prior_weights) < _score(prior_refined, d.prior_weights)
        full = guided_refined if guided_wins else prior_refined
        frames = {
            'prior_raw': prior_raw,
            'prior_refined': prior_refined,
            'bp_guided_raw': guided_raw,
            'bp_guided_refined': guided_refined,
            'full': full,
        }
        pred = {mode: _logical_prediction(frame, d.logical_masks, d.observables)
                for mode, frame in frames.items()}
        # Unlike the original BP's marginal-parity heuristic, all candidates
        # are explicit fault patterns satisfying the detector syndrome.
        for mode, frame in frames.items():
            observed = 0
            for j in _bits(frame):
                observed ^= d.det_masks[j]
            if observed != target:
                raise ArithmeticError(f'{mode}: syndrome-inconsistent recovery')
        return pred, {'bp_converged': result.converged,
                      'guided_wins_prior_score': bool(guided_wins),
                      'logical_changed_by_bp': bool(np.any(pred['full'] != pred['prior_refined'])),
                      'times_seconds': {
                          'prior_solve': t1-t0,
                          'prior_refine': t2-t1,
                          'bp_messages': t3-t2,
                          'guided_basis_and_solve': t4-t3,
                          'guided_refine': t5-t4,
                      }}


def evaluate_archive(archive: Path, output: Path, *, cases: list[str], shots: int = 32,
                     iterations: int | None = None) -> dict:
    if not cases or len(set(cases)) != len(cases) or shots <= 0:
        raise ValueError('unique case identifiers and positive shot count required')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('ablation output must be a fresh directory')
    output.mkdir(parents=True, exist_ok=True)
    rows=[]
    for case_id in cases:
        meta, syndromes, historical, faults, hashes = _load_case(archive, case_id, shots)
        # Original source has 20 BP iterations, 0.3 damping, 64 refinement.
        d = SyndromeRecoveryDecoder(meta['detectors'], meta['observables'], faults,
                                    iterations=20 if iterations is None else iterations,
                                    damping=0.3, refinement_budget=64, check_update='vectorized')
        analyzer = RecoveryAblator(d)
        predictions={m: [] for m in MODES}
        times={k: [] for k in ('prior_solve','prior_refine','bp_messages','guided_basis_and_solve','guided_refine')}
        changes=0; guided_wins=0; converged=0
        for syn in syndromes:
            pp, details = analyzer.decode_all(syn)
            for mode in MODES: predictions[mode].append(pp[mode])
            for k,v in details['times_seconds'].items(): times[k].append(v)
            changes += details['logical_changed_by_bp']
            guided_wins += details['guided_wins_prior_score']
            converged += details['bp_converged']
        predictions={m: np.stack(v) for m,v in predictions.items()}
        if iterations is None or iterations == 20:
            if not np.array_equal(predictions['full'], historical):
                raise AssertionError(f'{case_id}: ablation changed the historical frozen decoder outputs')
        # The archive is an already evaluated historical held-out dataset;
        # this use is explicitly and permanently DEVELOPMENT_ONLY.
        with np.load(_case_dir(archive)/(case_id+'.npz'), allow_pickle=False) as data:
            truth = data['actual_observables'][:shots]
            mwpm = data['mwpm_predictions'][:shots]
        if truth.shape != predictions['full'].shape or mwpm.shape != truth.shape:
            raise ValueError('truth/predictions shape mismatch')
        errors={mode:int(np.count_nonzero(np.any(pred != truth, axis=1)))
                for mode,pred in predictions.items()}
        errors['mwpm'] = int(np.count_nonzero(np.any(mwpm != truth, axis=1)))
        count=shots
        row={'case_id':case_id,'shots':count,'faults':len(faults),
             'detectors':meta['detectors'], 'model_sha256':hashes,
             'logical_changed_by_bp':changes,
             'bp_guided_won_prior_score':guided_wins,
             'bp_converged':converged,'errors':errors,
             'timing_component_median_ms':{k:1000*statistics.median(v) for k,v in times.items()},
             'timing_component_total_seconds':{k:float(sum(v)) for k,v in times.items()}}
        rows.append(row)
        print(f'{case_id}: errors ' + ' '.join(f'{m}={errors[m]}' for m in ('mwpm','prior_raw','prior_refined','full'))
              + f'; BP changes {changes}/{shots}; BP time {row["timing_component_median_ms"]["bp_messages"]:.2f} ms/shot',flush=True)
    report={'status':'DEVELOPMENT_ONLY','cases':rows,'shots_per_case':shots,
            'notes':'Historical held-out data repurposed for ablation tuning; not independent held-out verification. '
                    'Component timings include Python measurement overhead and do not substitute standalone benchmarks.',
            'machine':platform.platform()}
    (output/'ablation.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    with (output/'ablation.csv').open('w',newline='') as f:
        fields=['case_id','shots','mwpm_errors']+[m+'_errors' for m in MODES]+['bp_changed_predictions','bp_guided_won_prior_score','bp_converged','bp_median_ms']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            out={'case_id':row['case_id'],'shots':row['shots'],'mwpm_errors':row['errors']['mwpm'],
                 'bp_changed_predictions':row['logical_changed_by_bp'],
                 'bp_guided_won_prior_score':row['bp_guided_won_prior_score'],
                 'bp_converged':row['bp_converged'],
                 'bp_median_ms':row['timing_component_median_ms']['bp_messages']}
            out.update({m+'_errors':row['errors'][m] for m in MODES})
            w.writerow(out)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/performance_development/ablation'))
    p.add_argument('--shots',type=int,default=32)
    p.add_argument('--iterations',type=int)
    p.add_argument('--case',action='append',dest='cases')
    args=p.parse_args(argv)
    cases=args.cases or [
        'circuit_level_x_d3_r3_p0.005','circuit_level_x_d5_r5_p0.01',
        'circuit_level_x_d7_r7_p0.01','circuit_level_z_d3_r3_p0.005',
        'circuit_level_z_d5_r5_p0.01','circuit_level_z_d7_r7_p0.01']
    result=evaluate_archive(args.archive_root,args.output,cases=cases,shots=args.shots,iterations=args.iterations)
    print(json.dumps({'status':result['status'],'cases':len(result['cases'])},indent=2))

if __name__=='__main__':main()
