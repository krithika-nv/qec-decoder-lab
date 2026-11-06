"""Development-only decoder engine/iteration experiment on archived samples.

All variants use identical input shots. JIT compilation, decoder construction,
and file I/O are excluded from timed batches. CPU environment is uncontrolled.
This is model selection/tuning data, NOT an independent accuracy validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from .optimization_study import _case_dir, _load_case
from .recovery import SyndromeRecoveryDecoder

CANDIDATES = {
    'baseline_numpy_20_python': ('vectorized',20,'python'),
    'compiled_bp_20_python': ('compiled',20,'python'),
    'numpy_bp_20_packed': ('vectorized',20,'compiled'),
    'compiled_bp_20_packed': ('compiled',20,'compiled'),
    'compiled_bp_10_packed': ('compiled',10,'compiled'),
    'compiled_bp_5_packed': ('compiled',5,'compiled'),
    'minsum_20_packed': ('minsum',20,'compiled'),
    'minsum_5_packed': ('minsum',5,'compiled'),
}

DEFAULT_CASES = ('circuit_level_x_d3_r3_p0.005',
                 'circuit_level_z_d3_r3_p0.01',
                 'circuit_level_x_d5_r5_p0.01',
                 'circuit_level_z_d5_r5_p0.01',
                 'circuit_level_x_d7_r7_p0.01',
                 'circuit_level_z_d7_r7_p0.01')


def run_engine_study(archive:Path, output:Path, *, cases=DEFAULT_CASES,
                     shots:int=32, repeats:int=3, candidates=CANDIDATES):
    if shots<1 or repeats<2 or not cases or not candidates:
        raise ValueError('need shots >=1, repeats >=2, cases and candidates')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('study output exists; do not overwrite tuning evidence')
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for case in cases:
        meta, syn, historical, faults, hashes=_load_case(archive,case,shots)
        with np.load(_case_dir(archive)/(case+'.npz'),allow_pickle=False) as data:
            truth=data['actual_observables'][:shots].copy()
            mwpm=data['mwpm_predictions'][:shots].copy()
        mwerr=int(np.count_nonzero(np.any(mwpm!=truth,axis=1)))
        for name,(engine,iters,basis) in candidates.items():
            decoder=SyndromeRecoveryDecoder(meta['detectors'],meta['observables'],faults,
                                            iterations=iters,damping=.3,
                                            refinement_budget=64,
                                            check_update=engine,basis_update=basis)
            # JIT warm-up and output verification are excluded from timings.
            decoder.decode(syn[0])
            candidate=decoder.decode_batch(syn)
            historical_same=bool(np.array_equal(candidate,historical))
            if iters==20 and engine in ('compiled','vectorized') and not historical_same:
                raise AssertionError(f'{case}/{name}: numerical preservation violated')
            individual=[]
            for i in range(repeats):
                t=perf_counter()
                observed=decoder.decode_batch(syn)
                elapsed=perf_counter()-t
                if not np.array_equal(observed,candidate):
                    raise ValueError('decoder output changed between repeated runs')
                individual.append(elapsed)
            err=int(np.count_nonzero(np.any(candidate!=truth,axis=1)))
            row={'case_id':case,'candidate':name,'engine':engine,'iterations':iters,
                 'basis':basis,'shots':shots,'mwpm_errors':mwerr,
                 'candidate_errors':err,'historical_predictions_identical':historical_same,
                 'disagreements_with_historical':int(np.count_nonzero(np.any(candidate!=historical,axis=1))),
                 'median_us_per_shot':statistics.median(individual)*1e6/shots,
                 'trial_seconds':individual,'input_sha256':hashes}
            records.append(row)
            print(f'{case} | {name}: {row["median_us_per_shot"]:.1f} us/shot; '
                  f'logical errors {err}/{shots}; previous outputs identical={historical_same}',flush=True)
    result={'status':'DEVELOPMENT_ONLY', 'records':records,
            'machine':{'platform':platform.platform(),'python':sys.version.split()[0]},
            'method':'warmed same-machine decode_batch on same archived historical syndromes',
            'warning':'Tuning on previously inspected held-out shots. Timings are not PyMatching speed '
                      'comparisons. Do not treat changes in observed error counts as independent validation.'}
    (output/'engine_study.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    keys=('case_id','candidate','engine','iterations','basis','shots','mwpm_errors',
          'candidate_errors','historical_predictions_identical','disagreements_with_historical',
          'median_us_per_shot')
    with (output/'engine_study.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in records:w.writerow({k:row[k] for k in keys})
    return result


def audit_engine_study(archive:Path, output:Path):
    data=json.loads((output/'engine_study.json').read_text())
    if data['status']!='DEVELOPMENT_ONLY':
        raise ValueError('incorrect scientific study status')
    from .frozen_validation import _digest
    for row in data['records']:
        folder=_case_dir(archive)
        for ext,digest in row['input_sha256'].items():
            if _digest(folder/(row['case_id']+'.'+ext))!=digest:
                raise ValueError('original evidence hashes changed')
        trials=row['trial_seconds']
        if len(trials)<2 or any(t<=0 or not np.isfinite(t) for t in trials):
            raise ValueError('invalid timing evidence')
        if not np.isclose(statistics.median(trials)*1e6/row['shots'],
                          row['median_us_per_shot'],rtol=1e-12):
            raise ValueError('timing arithmetic inconsistent')
    with (output/'engine_study.csv').open(newline='') as f: csv_records=list(csv.DictReader(f))
    if len(csv_records)!=len(data['records']):
        raise ValueError('summary CSV incomplete')
    for a,b in zip(data['records'],csv_records):
        for k in ('case_id','candidate','engine','basis','historical_predictions_identical'):
            if str(a[k])!=b[k]:
                raise ValueError('summary CSV mismatch')
    return {'status':'PASS','records_verified':len(data['records'])}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/performance_development/engines'))
    p.add_argument('--shots',type=int,default=32)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--case',action='append',dest='cases')
    p.add_argument('--candidate',action='append',dest='candidates')
    args=p.parse_args(argv)
    chosen={k:CANDIDATES[k] for k in (args.candidates or CANDIDATES)}
    run_engine_study(args.archive_root,args.output,shots=args.shots,repeats=args.repeats,
                     cases=args.cases or DEFAULT_CASES,candidates=chosen)
    print(json.dumps(audit_engine_study(args.archive_root,args.output),indent=2))

if __name__=='__main__':main()
