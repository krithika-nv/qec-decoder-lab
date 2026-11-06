"""Retrospective hybrid gate study from the archived optimized held-out results.

CAUTION: This archive was already inspected. Selecting a gate based on these
results converts it into DEVELOPMENT DATA. Prospective validation is mandatory.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .hybrid import select_complex_syndromes
from .optimization_study import _case_dir

DEFAULT_THRESHOLDS = (0.00, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_gates(archive:Path, output:Path, *, thresholds=DEFAULT_THRESHOLDS):
    thresholds = tuple(float(t) for t in thresholds)
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(t<0 or t>1 for t in thresholds):
        raise ValueError('gate fractions must be unique and in [0,1]')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('never overwrite retrospective evidence')
    folder = _case_dir(archive)
    ids=sorted(p.stem for p in folder.glob('*.json'))
    if not ids:
        raise FileNotFoundError('missing archived case records')
    rows=[]
    for case_id in ids:
        meta_path=folder/(case_id+'.json');raw_path=folder/(case_id+'.npz')
        meta=json.loads(meta_path.read_text())
        if meta['case_id'] != case_id or meta['artifact_sha256']['npz'] != _sha(raw_path):
            raise ValueError(f'{case_id}: invalid original evidence checksum')
        for ext in ('dem', 'stim'):
            recorded = meta.get('artifact_sha256', {}).get(ext)
            if recorded is not None:
                archived_file=folder/(case_id+'.'+ext)
                if not archived_file.is_file() or _sha(archived_file) != recorded:
                    raise ValueError(f'{case_id}: {ext} checksum changed')
        with np.load(raw_path, allow_pickle=False) as data:
            syn=data['detection_events'][:meta['n_paired']]
            truth=data['actual_observables'][:meta['n_paired']]
            mwpm=data['mwpm_predictions'][:meta['n_paired']]
            rec=data['recovery_predictions'][:meta['n_paired']]
        if (syn.shape!=(meta['n_paired'],meta['detectors']) or
                truth.shape!=(meta['n_paired'],meta['observables']) or
                mwpm.shape!=truth.shape or rec.shape!=truth.shape):
            raise ValueError('mismatched archived sample dimensions')
        if any(np.any((x!=0)&(x!=1)) for x in (syn,truth,mwpm,rec)):
            raise ValueError('nonbinary archive')
        mw_wrong=np.any(mwpm!=truth,axis=1)
        rec_wrong=np.any(rec!=truth,axis=1)
        for threshold in thresholds:
            mask=select_complex_syndromes(syn,threshold)
            selected_errors=int(np.sum(np.where(mask,rec_wrong,mw_wrong)))
            rows.append({'case_id':case_id,'threshold':threshold,'shots':len(mask),
                         'selected':int(mask.sum()),'mwpm_errors':int(mw_wrong.sum()),
                         'full_recovery_errors':int(rec_wrong.sum()),'hybrid_errors':selected_errors,
                         'raw_sha256':_sha(raw_path)})
    summary=[]
    for threshold in thresholds:
        subset=[r for r in rows if r['threshold']==threshold]
        total=sum(r['shots'] for r in subset)
        mw_errors=sum(r['mwpm_errors'] for r in subset)
        full_errors=sum(r['full_recovery_errors'] for r in subset)
        hybrid_errors=sum(r['hybrid_errors'] for r in subset)
        gain=mw_errors-full_errors
        summary.append({'threshold':threshold,'cases':len(subset),'paired_shots':total,
                        'selected_fraction':sum(r['selected'] for r in subset)/total,
                        'mwpm_errors':mw_errors,'full_recovery_errors':full_errors,
                        'hybrid_errors':hybrid_errors,
                        'fraction_of_observed_recovery_gain_retained':
                            (mw_errors-hybrid_errors)/gain if gain else None})
    result={'status':'DEVELOPMENT_ONLY','source':'already-reviewed optimized held-out archive',
            'summary':summary,'per_case':rows,
            'warning':'Gate tuned to existing outputs; its apparent accuracy and compute savings are '
                      'retrospective and cannot be used as held-out validation. The selected '
                      'fraction is NOT measured latency; MWPM still runs on every shot. '
                      'Implementation overhead and batch effects are not included.'}
    output.mkdir(parents=True, exist_ok=True)
    (output/'hybrid_gate_study.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    for filename,records in [('hybrid_gate_summary.csv',summary),('hybrid_gate_per_case.csv',rows)]:
        with (output/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/performance_development/gate'))
    p.add_argument('--threshold',type=float,action='append',dest='thresholds')
    args=p.parse_args(argv)
    result=evaluate_gates(args.archive_root,args.output,thresholds=args.thresholds or DEFAULT_THRESHOLDS)
    for x in result['summary']:
        print(f"gate={x['threshold']:.2f}: selected={100*x['selected_fraction']:.1f}% "
              f"hybrid_errors={x['hybrid_errors']} / {x['paired_shots']}; "
              f"historical improvement retained={100*x['fraction_of_observed_recovery_gain_retained']:.1f}%")

if __name__=='__main__':main()
