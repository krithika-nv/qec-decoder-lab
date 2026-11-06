"""Numerical regression tests: optimization is not a new decoding algorithm."""
import json
from pathlib import Path

import numpy as np
import pytest

from qec_decoder_lab.belief_propagation import BeliefPropagationDEMDecoder
from qec_decoder_lab.exact import Fault
from qec_decoder_lab.frozen_validation import create_protocol, load_locked_protocol
from qec_decoder_lab.optimization_study import benchmark_archive, audit_archive
from qec_decoder_lab.recovery import SyndromeRecoveryDecoder, _basis
from qec_decoder_lab.frozen_validation import _digest


def _legacy_check_update(decoder, inputs, y):
    value = np.zeros(decoder.edges)
    for i in range(decoder.detectors):
        start, end = decoder.indptr[i], decoder.indptr[i+1]
        if start == end:
            continue
        t = np.tanh(inputs[start:end] / 2)
        left = np.concatenate([[1.], np.cumprod(t[:-1])])
        right = np.concatenate([np.cumprod(t[:0:-1])[::-1], [1.]])
        prod = np.clip((1 if y[i] == 0 else -1) * left * right, -1 + 1e-12, 1 - 1e-12)
        value[start:end] = 2*np.arctanh(prod)
    return value


def test_vectorized_parity_update_matches_reference_with_zeros():
    rng=np.random.default_rng(5401)
    for trial in range(12):
        det=5
        faults=[]
        for j in range(14):
            mask=sum((1 << i) for i in range(det) if rng.random() < .35)
            faults.append(Fault(.08, mask | ((j%3==0)<<det)))
        dec=BeliefPropagationDEMDecoder(det,1,faults)
        x=rng.normal(size=dec.edges)*2
        if dec.edges:
            x[rng.integers(dec.edges)] = 0.0
        y=rng.integers(0,2,size=det,dtype=np.uint8)
        np.testing.assert_allclose(dec._vectorized_check_messages(x,y),
                                   _legacy_check_update(dec,x,y),rtol=1e-12,atol=1e-10)


def test_optimized_recovery_unchanged_from_legacy_on_synthetic_model():
    faults=[Fault(.03,1),Fault(.05,2),Fault(.1,4), Fault(.07,3|8),
            Fault(.02,5),Fault(.04,6|8),Fault(.08,7), Fault(.015,8)]
    dec_old=SyndromeRecoveryDecoder(3,1,faults,iterations=15,check_update='legacy')
    dec_new=SyndromeRecoveryDecoder(3,1,faults,iterations=15,check_update='vectorized')
    shots=np.array([[(x>>i)&1 for i in range(3)] for x in range(8)],dtype=np.uint8)
    for row in shots:
        old=dec_old.decode(row); new=dec_new.decode(row)
        np.testing.assert_array_equal(old.prediction,new.prediction)
        assert old.candidate_method==new.candidate_method
        assert old.syndrome_consistent and new.syndrome_consistent
    with pytest.raises(ValueError, match='check_update'):
        SyndromeRecoveryDecoder(3,1,faults,check_update='incorrect')


def test_early_rank_stop_preserves_complete_gaussian_basis():
    rng=np.random.default_rng(991)
    for _ in range(30):
        masks=[int(x) for x in rng.integers(0, 1<<7,size=25)]
        order=rng.permutation(len(masks))
        full=_basis(masks,order)
        early=_basis(masks,order,target_rank=len(full))
        assert early==full


def test_post_optimization_protocol_has_engine_locked_and_distance_seven(tmp_path):
    study=create_protocol(tmp_path/'post',profile='focused', shots=15000, paired_shots=4096)
    assert study['check_update']=='vectorized'
    assert len(study['case_specs'])==12
    assert 7 in {item['distance'] for item in study['case_specs']}
    assert load_locked_protocol(tmp_path/'post')['check_update']=='vectorized'
    p=tmp_path/'post'/'locked_protocol.json'
    changed=json.loads(p.read_text());changed['check_update']='legacy'
    p.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='altered'):
        load_locked_protocol(tmp_path/'post')


def test_archived_benchmark_refuses_hash_tampering_and_reports_correct_metrics(tmp_path):
    folder=tmp_path/'original'/'cases';folder.mkdir(parents=True)
    name='unit_case'
    dem=folder/(name+'.dem');dem.write_text('error(0.03) D0 L0\nerror(0.05) D1\n')
    arr=folder/(name+'.npz')
    shots=np.array([[0,0],[1,0],[0,1],[1,1]],dtype=np.uint8)
    expected=SyndromeRecoveryDecoder(2,1,[Fault(.03,1|4),Fault(.05,2)],
                                     check_update='legacy').decode_batch(shots)
    np.savez_compressed(arr,detection_events=shots,recovery_predictions=expected)
    meta={'case_id':name,'detectors':2,'observables':1,'n_paired':4,
          'decoder_parameters':{'iterations':20,'damping':.3,'refinement_budget':64},
          'artifact_sha256': {'npz':_digest(arr),'dem':_digest(dem)}}
    (folder/(name+'.json')).write_text(json.dumps(meta))
    output=tmp_path/'report'
    benchmark_archive(tmp_path/'original',output,cases=[name],shots=4,repeats=2)
    assert audit_archive(tmp_path/'original',output)['status']=='PASS'
    record=json.loads((output/'optimization_benchmark.json').read_text())
    assert record['cases'][0]['predictions_match_prior_evidence']
    csv_path=output/'optimization_summary.csv'
    csv_path.write_text(csv_path.read_text().replace('True','False'))
    with pytest.raises(ValueError,match='CSV summary'):
        audit_archive(tmp_path/'original',output)
    arr.write_bytes(arr.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='hash'):
        audit_archive(tmp_path/'original',output)
