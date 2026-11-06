"""Unit tests for optional compiled kernels, ablations and gated decoding."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from qec_decoder_lab.exact import Fault
from qec_decoder_lab.recovery import SyndromeRecoveryDecoder, _basis
from qec_decoder_lab.ablation import RecoveryAblator
from qec_decoder_lab.hybrid import SelectiveHybridDecoder, select_complex_syndromes
from qec_decoder_lab.hybrid_study import evaluate_gates
from qec_decoder_lab import frozen_validation as fv


def _faults():
    return [Fault(.07, 0b01001), Fault(.11, 0b00011), Fault(.06, 0b00110),
            Fault(.03, 0b10100), Fault(.02, 0b01011), Fault(.05, 0b00001)]


def test_recovery_ablation_full_is_unchanged_and_all_syndromes_consistent():
    d=SyndromeRecoveryDecoder(4,1,_faults(),iterations=10)
    ab=RecoveryAblator(d)
    for x in range(16):
        s=np.array([(x>>i)&1 for i in range(4)],dtype=np.uint8)
        predictions,details=ab.decode_all(s)
        assert set(predictions)=={'prior_raw','prior_refined','bp_guided_raw','bp_guided_refined','full'}
        assert np.array_equal(predictions['full'],d.decode(s).prediction)
        assert all(y.shape==(1,) for y in predictions.values())
        assert all(v>=0 for v in details['times_seconds'].values())


def test_compiled_basis_exactly_preserves_python_pivot_representation():
    pytest.importorskip('numba')
    from qec_decoder_lab.compiled_basis import pack_detector_masks, packed_basis
    rng=np.random.default_rng(7)
    for detectors in (1,7,63,64,65,121):
        masks=[int(''.join('1' if b else '0' for b in rng.integers(0,2,detectors)),2)
               for _ in range(40)]
        packed=pack_detector_masks(masks,detectors)
        for seed in range(3):
            order=np.random.default_rng(seed).permutation(len(masks))
            reference=_basis(masks,order)
            assert packed_basis(packed,order,detectors,len(reference))==reference


def test_compiled_bp_matches_vectorized_marginals_including_zero_messages():
    pytest.importorskip('numba')
    from qec_decoder_lab.belief_propagation import BeliefPropagationDEMDecoder
    models=[(4,1,_faults()),(1,1,[Fault(.5,1),Fault(.5,3)]),
            (0,1,[Fault(.1,1)]),(1,1,[Fault(.01,3)])]
    for detectors,obs,faults in models:
        a=BeliefPropagationDEMDecoder(detectors,obs,faults,iterations=10,check_update='vectorized')
        b=BeliefPropagationDEMDecoder(detectors,obs,faults,iterations=10,check_update='compiled')
        all_y=np.array([[(x>>i)&1 for i in range(detectors)] for x in range(1<<detectors)],dtype=np.uint8)
        for y in all_y:
            try: old=a.decode(y)
            except ValueError:
                with pytest.raises(ValueError):b.decode(y)
                continue
            new=b.decode(y)
            assert np.allclose(old.fault_marginals,new.fault_marginals,atol=1e-9,rtol=1e-9)
            assert np.array_equal(old.prediction,new.prediction)
            assert old.converged==new.converged


def test_compiled_recovery_can_replay_reference_outputs():
    pytest.importorskip('numba')
    base=SyndromeRecoveryDecoder(4,1,_faults(),iterations=10)
    compiled=SyndromeRecoveryDecoder(4,1,_faults(),iterations=10,
                                     check_update='compiled',basis_update='compiled')
    y=np.array([[(x>>i)&1 for i in range(4)] for x in range(16)],dtype=np.uint8)
    assert np.array_equal(base.decode_batch(y),compiled.decode_batch(y))


def test_min_sum_is_opt_in_non_equivalent_approximation():
    pytest.importorskip('numba')
    from qec_decoder_lab.belief_propagation import BeliefPropagationDEMDecoder
    x=BeliefPropagationDEMDecoder(4,1,_faults(),iterations=5,check_update='minsum')
    r=x.decode(np.array([0,1,0,0],dtype=np.uint8))
    assert np.all(np.isfinite(r.fault_marginals))
    assert np.all((r.fault_marginals>=0)&(r.fault_marginals<=1))


def test_gate_logic_and_hybrid_select_only_hard_cases():
    s=np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0]],dtype=np.uint8)
    assert select_complex_syndromes(s,.5).tolist()==[False,False,True,True]
    with pytest.raises(ValueError):select_complex_syndromes(s,1.1)
    class Matcher:
        def decode_batch(self, a): return np.zeros((len(a),1),dtype=np.uint8)
    class Recovery:
        observables=1
        def decode_batch(self,a):return np.ones((len(a),1),dtype=np.uint8)
    x=SelectiveHybridDecoder(Matcher(),Recovery(),threshold=.5)
    y,d=x.decode_batch(s,with_details=True)
    assert y[:,0].tolist()==[0,0,1,1]
    assert d['selected_shots']==2


def test_frozen_hybrid_protocol_refuses_missing_or_modified_gate(tmp_path):
    with pytest.raises(ValueError):
        fv.create_protocol(tmp_path/'bad',profile='smoke',strategy='hybrid',shots=8,paired_shots=4)
    p=fv.create_protocol(tmp_path/'frozen',profile='smoke',strategy='hybrid',gate_fraction=.1,
                         shots=8,paired_shots=4,bootstrap_repeats=100)
    assert p['strategy']=='hybrid'
    assert fv.load_locked_protocol(tmp_path/'frozen')['gate_fraction']==.1
    path=tmp_path/'frozen'/fv.PROTOCOL_FILE
    bad=json.loads(path.read_text());bad['gate_fraction']=.2
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='altered'):
        fv.load_locked_protocol(tmp_path/'frozen')


def test_frozen_hybrid_calculation_preserves_gate_off_mwpm():
    from qec_decoder_lab.frozen_validation import _decode_experimental
    class FakeDecoder:
        def __init__(self):self.observables=1
        @classmethod
        def from_dem(cls,*args,**kwargs):return cls()
        def decode_batch(self,s,**kwargs):
            return np.ones((len(s),1),dtype=np.uint8), {'unique_syndromes':len(s)}
    import qec_decoder_lab.recovery as rec
    old=rec.SyndromeRecoveryDecoder
    try:
        rec.SyndromeRecoveryDecoder=FakeDecoder
        protocol={'iterations':5,'damping':.3,'refinement_budget':4,
                  'check_update':'vectorized','strategy':'hybrid','gate_fraction':.75}
        s=np.array([[0,0],[1,0],[1,1]],dtype=np.uint8)
        mwpm=np.zeros((3,1),dtype=np.uint8)
        y,details=_decode_experimental(protocol,None,s,mwpm)
        assert y[:,0].tolist()==[0,0,1]
        assert details['selected_shots']==1
    finally:rec.SyndromeRecoveryDecoder=old


def test_hybrid_archive_checks_hash_and_counts(tmp_path):
    root=tmp_path/'historical';folder=root/'cases';folder.mkdir(parents=True)
    case_id='circuit_level_x_d3_r3_p0.005'
    syn=np.array([[0,0],[1,1],[1,0],[0,1]],dtype=np.uint8)
    true=np.array([[0],[1],[1],[0]],dtype=np.uint8)
    mwpm=np.array([[0],[0],[1],[0]],dtype=np.uint8)
    recovery=np.array([[0],[1],[0],[0]],dtype=np.uint8)
    raw=folder/(case_id+'.npz')
    np.savez_compressed(raw,detection_events=syn,actual_observables=true,
                        mwpm_predictions=mwpm,recovery_predictions=recovery)
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    meta={'case_id':case_id,'detectors':2,'observables':1,'n_paired':4,
          'artifact_sha256':{'npz':digest}}
    (folder/(case_id+'.json')).write_text(json.dumps(meta))
    result=evaluate_gates(root,tmp_path/'out',thresholds=[0,.5])
    assert len(result['per_case'])==2
    assert result['summary'][1]['selected_fraction']==.75
    with pytest.raises(FileExistsError):evaluate_gates(root,tmp_path/'out')
    raw.write_bytes(raw.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='checksum'):
        evaluate_gates(root,tmp_path/'out2')


def test_hybrid_full_mock_study_reconciles_gate_metadata(tmp_path, monkeypatch):
    """End-to-end frozen study accounting without pretending to run Stim."""
    root=tmp_path/'hybrid'
    fv.create_protocol(root,profile='smoke',shots=16,paired_shots=8,
                       strategy='hybrid',gate_fraction=.5,
                       bootstrap_repeats=100)
    def fake_capture(root,protocol,spec,seed,progress):
        cid=fv._case_id(spec);paths=fv._case_paths(root,cid)
        paths['npz'].parent.mkdir(parents=True,exist_ok=True)
        sy=np.tile(np.array([[0,0,0,0],[1,1,0,0]],dtype=np.uint8),(8,1))
        truth=np.zeros((16,1),dtype=np.uint8)
        mwpm=np.zeros((16,1),dtype=np.uint8)
        selected=select_complex_syndromes(sy[:8],.5)
        hybrid=mwpm[:8].copy();hybrid[selected]=1
        np.savez_compressed(paths['npz'],detection_events=sy,actual_observables=truth,
                            mwpm_predictions=mwpm,recovery_predictions=hybrid)
        paths['stim'].write_text('fake circuit\n');paths['dem'].write_text('fake DEM\n')
        item={'schema':fv.SCHEMA,'case_id':cid,'spec':spec,'seed':seed,
              'protocol_sha256':protocol['protocol_sha256'],'n_shots':16,
              'n_paired':8,'detectors':4,'observables':1,'status':'COMPLETE',
              'metrics':fv.summarize_case(sy,truth,mwpm,hybrid,seed=seed,bootstrap_repeats=100),
              'recovery_diagnostics':{'selected_shots':int(selected.sum()),'gate_fraction':.5},
              'artifact_sha256':{x:fv._digest(paths[x]) for x in ('npz','stim','dem')}}
        paths['json'].write_bytes(fv._canonical_bytes(item));return item
    monkeypatch.setattr(fv,'_capture_case',fake_capture)
    assert fv.run_study(root,progress=False)['status']=='PASS'
    protocol=fv.load_locked_protocol(root)
    c=fv._case_paths(root,fv._case_id(protocol['case_specs'][0]))['json']
    data=json.loads(c.read_text());data['recovery_diagnostics']['selected_shots']=1
    c.write_bytes(fv._canonical_bytes(data))
    report=fv.audit_study(root)
    assert report['status']=='INCOMPLETE'
    assert any('metadata' in x['error'] for x in report['failures'])
