"""Development-only old-versus-optimized recovery regression and timing audit.

Uses previously inspected shots to verify logical predictions remain unchanged.
It is NOT a new held-out accuracy study and cannot establish decoder superiority.
Requires NumPy but not Stim, PyMatching, or PennyLane.
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import csv
import json
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from .archive_eval import parse_flattened_dem
from .frozen_validation import _canonical_bytes, _digest, _source_manifest
from .recovery import SyndromeRecoveryDecoder

DEFAULT_CASES = (
    'circuit_level_x_d3_r3_p0.01',
    'circuit_level_z_d5_r5_p0.01',
)


def _case_dir(root: Path) -> Path:
    if (root / 'cases').is_dir():
        return root / 'cases'
    if (root / 'heldout' / 'cases').is_dir():
        return root / 'heldout' / 'cases'
    if root.name == 'cases' and root.is_dir():
        return root
    raise FileNotFoundError('expected an extracted heldout/cases directory')


def _load_case(root: Path, case_id: str, shots: int):
    if '/' in case_id or '\\' in case_id or case_id.startswith('.'):
        raise ValueError('case must be a simple archive case identifier')
    folder = _case_dir(root)
    paths = {ext: folder / (case_id + '.' + ext) for ext in ('json', 'npz', 'dem')}
    if any(not p.is_file() for p in paths.values()):
        raise FileNotFoundError(f'missing archived case: {case_id}')
    meta = json.loads(paths['json'].read_text())
    if meta.get('case_id') != case_id:
        raise ValueError('case identifier mismatch')
    for ext in ('npz', 'dem'):
        if meta.get('artifact_sha256', {}).get(ext) != _digest(paths[ext]):
            raise ValueError(f'{case_id}: {ext} hash differs from original evidence')
    with np.load(paths['npz'], allow_pickle=False) as raw:
        required = ('detection_events', 'recovery_predictions')
        if any(n not in raw.files for n in required):
            raise ValueError('missing original shot archive arrays')
        syn = raw['detection_events'][:shots].copy()
        historical = raw['recovery_predictions'][:shots].copy()
    if len(syn) != shots or syn.ndim != 2 or historical.ndim != 2 or not 0 < shots <= meta['n_paired']:
        raise ValueError('invalid regression sample dimensions')
    if syn.shape[1] != meta['detectors'] or historical.shape[1] != meta['observables']:
        raise ValueError('incompatible model dimensions')
    if any(np.any((x != 0) & (x != 1)) for x in (syn, historical)):
        raise ValueError('nonbinary shot archive')
    faults = parse_flattened_dem(paths['dem'], meta['detectors'], meta['observables'])
    evidence_hashes = {ext: _digest(paths[ext]) for ext in ('json','npz','dem')}
    return meta, syn, historical, faults, evidence_hashes


def benchmark_archive(archive: Path, output: Path, *, cases=DEFAULT_CASES,
                      shots: int = 64, repeats: int = 5,
                      profile_shots: int = 16) -> dict:
    if shots < 1 or repeats < 2 or profile_shots < 0:
        raise ValueError('shots must be >=1, repeats >=2, profile_shots >= 0')
    if len(set(cases)) != len(cases) or not cases:
        raise ValueError('cases must be nonempty and unique')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('development benchmark destination is not empty; never overwrite evidence')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in cases:
        meta, syndromes, historical, faults, hashes = _load_case(archive, case_id, shots)
        kwargs = dict(iterations=meta.get('decoder_parameters', {}).get('iterations',20),
                      damping=meta.get('decoder_parameters', {}).get('damping',0.3),
                      refinement_budget=meta.get('decoder_parameters', {}).get('refinement_budget',64))
        legacy = SyndromeRecoveryDecoder(meta['detectors'], meta['observables'], faults,
                                         check_update='legacy', **kwargs)
        optimized = SyndromeRecoveryDecoder(meta['detectors'], meta['observables'], faults,
                                            check_update='vectorized', **kwargs)
        # Both must reproduce the originally archived predictions, not just each other.
        old_pred = legacy.decode_batch(syndromes)
        new_pred = optimized.decode_batch(syndromes)
        if not np.array_equal(old_pred, historical):
            raise AssertionError(f'{case_id}: legacy replay differs from archived predictions')
        if not np.array_equal(new_pred, historical):
            raise AssertionError(f'{case_id}: optimized replay changed archived predictions')
        measurements = {'legacy':[], 'vectorized':[]}
        # Alternate order to limit CPU-cache or load drift bias.
        for trial in range(repeats):
            for engine, decoder in ((('legacy',legacy),('vectorized',optimized)) if trial % 2 == 0 else
                                    (('vectorized',optimized),('legacy',legacy))):
                start = perf_counter()
                predictions = decoder.decode_batch(syndromes)
                measurements[engine].append(perf_counter()-start)
                if not np.array_equal(predictions, historical):
                    raise AssertionError(f'{case_id}: unexpected prediction drift in {engine}')
        a = statistics.median(measurements['legacy'])
        b = statistics.median(measurements['vectorized'])
        profiles = {}
        for engine, decoder in (('legacy', legacy), ('vectorized', optimized)):
            if profile_shots:
                profiler = cProfile.Profile()
                profiler.enable()
                decoder.decode_batch(syndromes[:min(shots, profile_shots)])
                profiler.disable()
                stats = pstats.Stats(profiler)
                profiles[engine] = [
                    {'function': Path(fn[0]).name + ':' + fn[2],
                     'calls': int(v[1]), 'self_seconds': float(v[2]),
                     'cumulative_seconds': float(v[3])}
                    for fn, v in sorted(stats.stats.items(), key=lambda kv: kv[1][3], reverse=True)[:16]
                ]
        rows.append({'case_id':case_id, 'shots':shots, 'repeats':repeats,
                     'cprofile_instrumented_shots': min(shots, profile_shots),
                     'cprofile_hotspots': profiles,
                     'detectors':int(meta['detectors']), 'faults':len(faults),
                     'legacy_trials_seconds':measurements['legacy'],
                     'optimized_trials_seconds':measurements['vectorized'],
                     'legacy_median_us_shot':a*1e6/shots,
                     'optimized_median_us_shot':b*1e6/shots,
                     'speedup_legacy_over_optimized':a/b,
                     'predictions_match_prior_evidence':True, 'input_sha256':hashes,
                     'decoder_parameters':kwargs})
        print(f'{case_id}: {a*1e6/shots:.1f} -> {b*1e6/shots:.1f} us/shot; '
              f'{a/b:.2f}x speedup; {shots} archived predictions unchanged',flush=True)
    result = {'status':'DEVELOPMENT_ONLY', 'cases':rows,
              'source_sha256':_source_manifest(),
              'machine': {'platform': platform.platform(), 'python':sys.version.split()[0],
                          'processor': platform.processor(), 'architecture':platform.machine()},
              'scope':'Same-process comparison of legacy vs vectorized BP on previously reviewed shots. '
                      'Not a held-out test, not a comparison with MWPM, not controlled CPU benchmarking.'}
    (output/'optimization_benchmark.json').write_bytes(_canonical_bytes(result))
    with (output/'optimization_summary.csv').open('w',newline='') as f:
        fields=['case_id','shots','detectors','faults','legacy_median_us_shot',
                'optimized_median_us_shot','speedup_legacy_over_optimized',
                'predictions_match_prior_evidence']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for case in rows:w.writerow({k:case[k] for k in fields})
    (output/'input_source_manifest.json').write_bytes(_canonical_bytes(result['source_sha256']))
    return result


def audit_archive(archive: Path, output: Path) -> dict:
    result=json.loads((output/'optimization_benchmark.json').read_text())
    if result['status']!='DEVELOPMENT_ONLY' or result['source_sha256']!=_source_manifest():
        raise ValueError('source changed or benchmark incorrectly labeled')
    rows=[]
    for row in result['cases']:
        _, _, _, _, hashes = _load_case(archive,row['case_id'],row['shots'])
        if hashes != row['input_sha256'] or not row['predictions_match_prior_evidence']:
            raise ValueError('archived evidence changed')
        for engine, key in [('legacy','legacy_median_us_shot'),
                            ('optimized','optimized_median_us_shot')]:
            a = row['legacy_trials_seconds'] if engine=='legacy' else row['optimized_trials_seconds']
            if len(a)!=row['repeats'] or any(not np.isfinite(v) or v<=0 for v in a):
                raise ValueError('invalid individual timing trials')
            if not np.isclose(statistics.median(a)*1e6/row['shots'],row[key],rtol=1e-12):
                raise ValueError('reported timing does not match trials')
        if not np.isclose(row['legacy_median_us_shot']/row['optimized_median_us_shot'],
                          row['speedup_legacy_over_optimized'],rtol=1e-12):
            raise ValueError('reported speedup does not reconcile')
        rows.append(row)
    with (output/'optimization_summary.csv').open(newline='') as f:
        csv_rows=list(csv.DictReader(f))
    if len(csv_rows)!=len(rows):
        raise ValueError('summary CSV count mismatch')
    for a,b in zip(rows,csv_rows):
        for key in ('case_id','shots','detectors','faults','legacy_median_us_shot',
                    'optimized_median_us_shot','speedup_legacy_over_optimized',
                    'predictions_match_prior_evidence'):
            if str(a[key]) != b[key]:
                raise ValueError(f'altered CSV summary: {key}')
    return {'status':'PASS','cases_verified':len(rows),'scope':result['scope']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    run=commands.add_parser('run')
    run.add_argument('--archive-root',type=Path,required=True)
    run.add_argument('--output',type=Path,default=Path('outputs/optimization_development'))
    run.add_argument('--shots',type=int,default=64)
    run.add_argument('--repeats',type=int,default=5)
    run.add_argument('--profile-shots',type=int,default=16)
    run.add_argument('--case',action='append')
    audit=commands.add_parser('audit')
    audit.add_argument('--archive-root',type=Path,required=True)
    audit.add_argument('--output',type=Path,default=Path('outputs/optimization_development'))
    args=p.parse_args(argv)
    if args.command=='run':
        benchmark_archive(args.archive_root,args.output,cases=args.case or DEFAULT_CASES,
                          shots=args.shots,repeats=args.repeats,
                          profile_shots=args.profile_shots)
    print(json.dumps(audit_archive(args.archive_root,args.output),indent=2))

if __name__=='__main__':
    main()
